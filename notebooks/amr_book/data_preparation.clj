;; # Data Preparation
;;
;; Before we can train a classifier, each raw
;; [MALDI-TOF](https://en.wikipedia.org/wiki/Matrix-assisted_laser_desorption/ionization)
;; spectrum must be preprocessed and converted into a
;; fixed-length feature vector.
;;
;; This notebook walks through every step of that
;; transformation for **one spectrum**, then shows how
;; the full dataset is assembled for machine learning.

(ns amr-book.data-preparation
  (:require
   ;; AMR ML pipeline (prepare, train, predict, measure):
   [scicloj.amr.learning :as learning]
   ;; AMR data loading utilities:
   [scicloj.amr.data.ingestion :as ingestion]
   ;; Bacterial species definitions and antibiotic lists:
   [scicloj.amr.data.bacteria :as bacteria]
   ;; Ripple MALDI signal processing (https://scicloj.github.io/ripple):
   [scicloj.ripple.maldi :as ripple]
   ;; Table processing (https://scicloj.github.io/tablecloth/):
   [tablecloth.api :as tc]
   ;; Column-level operations:
   [tablecloth.column.api :as tcc]
   ;; Interactive plotting via Plotly (https://scicloj.github.io/tableplot/):
   [scicloj.tableplot.v1.plotly :as plotly]
   ;; Annotating kinds of visualizations (https://scicloj.github.io/kindly-noted/):
   [scicloj.kindly.v4.kind :as kind]))

;; ## From the paper
;;
;; [Weis et al.](https://doi.org/10.1038/s41591-021-01619-9) describe
;; the preprocessing as follows
;; ([preprint](https://www.biorxiv.org/content/10.1101/2020.07.30.228411v2.full.pdf), pp 28–29):
;;
;; > The following preprocessing steps are performed using the R package
;; > MaldiQuant version 1.19: (1) the measured intensity is transformed
;; > with a square-root method to stabilize the variance, (2) smoothing
;; > using the Savitzky-Golay algorithm with half-window-size 10 is
;; > applied, (3) an estimate of the baseline is removed in 20
;; > iterations of the SNIP algorithm, (4) the intensity is calibrated
;; > using the total-ion-current (TIC), and (5) the spectra are trimmed
;; > to values in a 2,000 to 20,000 Da range.
;; >
;; > After preprocessing, each spectrum is represented by a set of
;; > measurements, each of them described by its corresponding
;; > mass-to-charge ratio and intensity. However, this representation
;; > results in each sample having potentially a different
;; > dimensionality [...]. Since the machine learning methods used in
;; > this manuscript require their input to be a feature vector of fixed
;; > dimensionality, intensity measurements are binned using the bin
;; > size of 3 Da. [...] each sample is represented by a vector
;; > containing 6,000 features.
;;
;; Examining the
;; [actual R code](https://github.com/BorgwardtLab/maldi_amr/blob/master/amr_maldi_ml/DRIAMS_preprocessing/DRIAMS-A_2018_preprocessed.r)
;; reveals two details not mentioned in the text:
;;
;; - The [Savitzky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) smoothing uses
;;   [MALDIquant](https://strimmerlab.github.io/software/maldiquant/)'s
;;   default polynomial order of **3** (not specified explicitly in
;;   their script).
;; - The [SNIP](https://doi.org/10.1016/0168-583X(88)90063-8) baseline
;;   removal is applied **twice**:
;;   `removeBaseline(removeBaseline(spectra, method="SNIP", iterations=20))`.
;;
;; We now walk through each of these steps.

;; ## Choosing a scenario
;;
;; We pick a single species / antibiotic / site / year
;; combination — **[*E. coli*](https://en.wikipedia.org/wiki/Escherichia_coli)
;; vs [Cefepime](https://en.wikipedia.org/wiki/Cefepime)**
;; from DRIAMS-A 2018:

(def params
  {:site :A
   :year 2018
   :species bacteria/E-coli
   :antibiotic :Cefepime})

;; `prepare-raw-data` loads the DRIAMS metadata, joins it
;; with available spectrum files, and filters to our scenario.
;; The result has a `:path` column (spectrum file) and a
;; `:ri` column (true = resistant or intermediate).

(def raw-data
  (learning/prepare-raw-data params))

(tc/row-count raw-data)

(tc/select-columns raw-data [:code :species :ri :path])

;; ### Resistance distribution
;;
;; How many spectra are resistant vs susceptible?

(def resistance-counts
  (-> raw-data
      (tc/group-by [:ri])
      (tc/aggregate {:count tc/row-count})))

resistance-counts

(-> resistance-counts
    (tc/map-columns :label [:ri] #(if % "Resistant/Intermediate" "Susceptible"))
    (plotly/base {:=x :label
                  :=y :count
                  :=title "Resistance distribution — E. coli / Cefepime"
                  :=x-title ""
                  :=y-title "Number of spectra"})
    (plotly/layer-bar)
    plotly/plot)

;; ## Loading a single raw spectrum
;;
;; Each spectrum is a gzipped text file with space-separated
;; mass/intensity pairs. Let's pick the first one:

(def spectrum-path
  (-> raw-data :path first))

(def raw-spectrum
  (ingestion/load-raw-spectrum spectrum-path))

raw-spectrum

{:points (tc/row-count raw-spectrum)
 :mass-min (-> raw-spectrum :mass first)
 :mass-max (-> raw-spectrum :mass last)}

(-> raw-spectrum
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=title "Raw spectrum"
                  :=x-title "m/z (Da)"
                  :=y-title "Intensity (a.u.)"})
    (plotly/layer-line)
    plotly/plot)

;; ## Step-by-step preprocessing
;;
;; The steps below follow the paper's R code exactly:
;;
;; 1. Square root transform (variance stabilization)
;; 2. [Savitzky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) smoothing (half-window-size 10, polynomial order 3)
;; 3. [SNIP](https://doi.org/10.1016/0168-583X(88)90063-8) baseline removal (20 iterations, **applied twice**)
;; 4. TIC normalization (total area → 1.0)
;;
;; We apply each one separately so we can see its effect.

;; ### Step 1 — Square root transform
;;
;; Taking the square root compresses the dynamic range,
;; giving lower-intensity peaks more weight relative to
;; dominant ones.

(def masses (:mass raw-spectrum))
(def raw-intensities (:intensity raw-spectrum))

(def sqrt-intensities
  (ripple/sqrt-transform raw-intensities))

(tc/dataset {:mass masses
             :raw-intensity raw-intensities
             :sqrt-intensity sqrt-intensities})

(-> (tc/dataset {:mass masses :intensity sqrt-intensities})
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=title "After square root transform"
                  :=x-title "m/z (Da)"
                  :=y-title "√Intensity"})
    (plotly/layer-line)
    plotly/plot)

;; ### Step 2 — Savitzky-Golay smoothing
;;
;; A polynomial fit in a sliding window removes
;; high-frequency noise while preserving peak shapes.
;; The paper uses half-window-size 10 (= full window of
;; 21 points) with MALDIquant's default polynomial order
;; of 3.

(def smoothed-intensities
  (ripple/savitzky-golay-smooth sqrt-intensities
                                {:window-size 21
                                 :polynomial-order 3}))

(tc/dataset {:mass masses
             :sqrt sqrt-intensities
             :smoothed smoothed-intensities})

(-> (tc/dataset {:mass masses
                 :sqrt sqrt-intensities
                 :smoothed smoothed-intensities})
    (tc/pivot->longer [:sqrt :smoothed]
                      {:target-columns :step
                       :value-column-name :intensity})
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=color :step
                  :=title "Smoothing effect (zoom in to see detail)"
                  :=x-title "m/z (Da)"
                  :=y-title "Intensity"})
    (plotly/layer-line)
    plotly/plot)

;; ### Step 3 — SNIP baseline removal (twice)
;;
;; The [SNIP algorithm](https://doi.org/10.1016/0168-583X(88)90063-8)
;; estimates and subtracts the slowly varying baseline
;; caused by chemical matrix effects.
;;
;; The paper's R code applies SNIP **twice** in succession:
;; `removeBaseline(removeBaseline(spectra, method="SNIP", iterations=20))`.
;; The second pass catches any residual baseline that the
;; first pass left behind.

(def baseline-corrected-once
  (ripple/snip-baseline-removal smoothed-intensities
                                {:iterations 20}))

(def baseline-corrected
  (ripple/snip-baseline-removal baseline-corrected-once
                                {:iterations 20}))

(tc/dataset {:mass masses
             :smoothed smoothed-intensities
             :after-1st-snip baseline-corrected-once
             :after-2nd-snip baseline-corrected})

(-> (tc/dataset {:mass masses
                 :smoothed smoothed-intensities
                 :after-first-snip baseline-corrected-once
                 :after-second-snip baseline-corrected})
    (tc/pivot->longer [:smoothed :after-first-snip :after-second-snip]
                      {:target-columns :step
                       :value-column-name :intensity})
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=color :step
                  :=title "Baseline removal — single vs double SNIP"
                  :=x-title "m/z (Da)"
                  :=y-title "Intensity"})
    (plotly/layer-line)
    plotly/plot)

;; ### Step 4 — TIC normalization
;;
;; [Total Ion Current](https://en.wikipedia.org/wiki/Total_ion_current)
;; normalization scales the spectrum so its area
;; (by the [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule))
;; equals 1.0. This makes spectra from different
;; measurements comparable.

(def normalized-intensities
  (ripple/tic-normalize masses baseline-corrected
                        {:target-area 1.0}))

(tc/dataset {:mass masses
             :baseline-corrected baseline-corrected
             :normalized normalized-intensities})

(-> (tc/dataset {:mass masses :intensity normalized-intensities})
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=title "After TIC normalization"
                  :=x-title "m/z (Da)"
                  :=y-title "Intensity (normalized)"})
    (plotly/layer-line)
    plotly/plot)

;; ## Trimming and binning
;;
;; For machine learning we need a fixed-length feature vector.
;; The DRIAMS protocol trims to [2000, 20000] Da and bins
;; into 3 Da wide bins, producing 6,000 features.

;; ### Trimming

(def preprocessed
  (tc/dataset {:mass masses :intensity normalized-intensities}))

(def trimmed
  (ripple/trim-spectrum preprocessed {:range [2000 20000]}))

{:points-before (tc/row-count preprocessed)
 :points-after (tc/row-count trimmed)}

(-> trimmed
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=title "Preprocessed spectrum (trimmed to 2000–20000 Da)"
                  :=x-title "m/z (Da)"
                  :=y-title "Intensity (normalized)"})
    (plotly/layer-line)
    plotly/plot)

;; ### Binning
;;
;; The m/z axis is partitioned into 3 Da bins and the
;; intensities falling into each bin are **summed**,
;; producing a vector of 6,000 features.

(def binned
  (ripple/bin-spectrum preprocessed {:range [2000 20000] :step 3}))

(count binned)

(kind/test-last
 #(= % 6000))

;; Here are the first few values:

(vec (take 10 binned))

;; Visualized as a line chart (bin index on the x-axis):

(-> (tc/dataset {:bin (range (count binned))
                 :intensity (vec binned)})
    (plotly/base {:=x :bin
                  :=y :intensity
                  :=title "Binned feature vector (6000 features)"
                  :=x-title "Bin index"
                  :=y-title "Summed intensity"})
    (plotly/layer-line)
    plotly/plot)

;; ## Processing all spectra
;;
;; `learning/prepare-ml-data` applies preprocessing and
;; binning to every spectrum in the dataset, producing
;; a table with one row per spectrum and 6,000 feature
;; columns (`:x0` through `:x5999`) plus the `:ri` target.
;;
;; **Note:** the current pipeline applies SNIP baseline
;; removal once (not twice as in the paper's R code).
;; This is a known simplification; a future version of
;; [Ripple](https://scicloj.github.io/ripple) will support
;; the double-SNIP pattern.

(def preprocessing-params
  {:smooth-window 21
   :smooth-polynomial 3})

(def ml-params
  {:preprocessing-params preprocessing-params
   :binning-params {:range [2000 20000] :step 3}})

(def ml-data
  (learning/prepare-ml-data raw-data ml-params))

{:rows (tc/row-count ml-data)
 :columns (count (tc/column-names ml-data))}

;; The first few columns:

(-> ml-data
    (tc/select-columns (into [:ri] (map #(keyword (str "x" %)) (range 5)))))

;; ### A few spectra overlaid
;;
;; Plotting the binned features for the first five spectra
;; shows the typical variation across samples:

(def n-overlay 5)

(def overlay-data
  (let [bins (range 6000)]
    (->> (range n-overlay)
         (mapcat (fn [i]
                   (let [row (-> ml-data (tc/select-rows [i]))]
                     (map (fn [b]
                            {:bin b
                             :intensity (first (row (keyword (str "x" b))))
                             :spectrum (str "spectrum-" i)})
                          bins))))
         tc/dataset)))

(-> overlay-data
    (plotly/base {:=x :bin
                  :=y :intensity
                  :=color :spectrum
                  :=title "Binned features — first 5 spectra"
                  :=x-title "Bin index"
                  :=y-title "Summed intensity"})
    (plotly/layer-line {:=mark-opacity 0.5})
    plotly/plot)

;; ## Train/test split
;;
;; As a final check, we split the dataset and verify
;; it is ready for modeling:

(def split-data
  (learning/split ml-data {:seed 1}))

{:train-rows (tc/row-count (:train split-data))
 :test-rows (tc/row-count (:test split-data))
 :train-resistance-rate (-> split-data :train :ri tcc/mean)
 :test-resistance-rate (-> split-data :test :ri tcc/mean)}

(kind/test-last
 #(and (> (:train-rows %) 0) (> (:test-rows %) 0)))

;; Resistance distribution in train and test sets:

(-> (concat
     (map (fn [ri] {:split "train" :label (if ri "R/I" "S")}) ((:train split-data) :ri))
     (map (fn [ri] {:split "test" :label (if ri "R/I" "S")}) ((:test split-data) :ri)))
    tc/dataset
    (tc/group-by [:split :label])
    (tc/aggregate {:count tc/row-count})
    (plotly/base {:=x :split
                  :=y :count
                  :=color :label
                  :=title "Resistance distribution — train vs test"
                  :=x-title ""
                  :=y-title "Count"})
    (plotly/layer-bar)
    plotly/plot)

;; The dataset is ready. Each row is a preprocessed,
;; binned MALDI spectrum with a resistance label —
;; exactly what the classifier needs.
