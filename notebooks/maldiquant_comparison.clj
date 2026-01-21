(ns maldiquant-comparison
  "Comparing Clojure signal processing implementations with R MALDIquant.
   
   This notebook demonstrates our pure-JVM implementations against
   the reference MALDIquant R package."
  (:require [maldi.data.ingestion :as ingestion]
            [maldi.data.signal :as signal]
            [tablecloth.api :as tc]
            [scicloj.tableplot.v1.plotly :as plotly]
            [clojisr.v1.r :as r :refer [r r->clj clj->r]]
            [scicloj.kindly.v4.kind :as kind]))

^{:kindly/hide-code true
  :kindly/kind :kind/hidden}
(import
 ;; Source - https://stackoverflow.com/a
 ;; Posted by henryw374
 ;; Retrieved 2026-01-21, License - CC BY-SA 3.0
 ch.qos.logback.classic.Logger
 ch.qos.logback.classic.Level)
^{:kindly/hide-code true
  :kindly/kind :kind/hidden}
(.setLevel 
 (org.slf4j.LoggerFactory/getLogger (Logger/ROOT_LOGGER_NAME)) Level/INFO)



;; # MALDIquant Comparison
;;
;; This notebook validates our Clojure implementations against the reference
;; MALDIquant R package.

;; ## Setup

;; Load MALDIquant in R
(r "library(MALDIquant)")

;; ## Test Data
;;
;; We'll use a simple synthetic spectrum with known peaks.

(def test-intensities 
  [0.1 0.2 0.5 1.0 0.5 0.2 0.1 0.15 0.3 0.8 1.5 0.7 0.2])

(kind/hiccup
 [:div
  [:h3 "Test Spectrum"]
  [:p "A simple 13-point spectrum with 2 clear peaks at indices 3 and 10."]])

;; Visualize test data
(-> {:index (range (count test-intensities))
     :intensity test-intensities}
    tc/dataset
    (plotly/base {:=width 600 :=height 300})
    (plotly/layer-point {:=x :index 
                         :=y :intensity
                         :=mark-size 8})
    (plotly/layer-line {:=x :index 
                        :=y :intensity
                        :=mark-opacity 0.3}))

;; ## 1. MAD Noise Estimation
;;
;; Median Absolute Deviation (MAD) is used to estimate noise level.
;; Formula: MAD = median(|x - median(x)|) * 1.4826

;; ### R Implementation
(def r-mad 
  (first (r->clj (r `(mad ~test-intensities)))))

;; ### Clojure Implementation
(def clj-mad 
  (signal/estimate-noise-mad test-intensities {}))

;; ### Comparison
(kind/table
 {:column-names ["Implementation" "MAD Value" "Difference"]
  :row-vectors [["R (MALDIquant)" r-mad "-"]
                ["Clojure" clj-mad (- clj-mad r-mad)]]})

(= r-mad clj-mad)

(kind/test-last [true?])

;; ## 2. Local Maxima Detection
;;
;; The sliding window algorithm identifies local maxima.

;; Create a longer test spectrum for better visualization
(def spectrum-data
  (vec (concat 
        ;; baseline:
        (repeat 10 0.05)
        ;; peak 1:
        [0.1 0.2 0.5 1.2 0.6 0.2 0.1]
        ;; baseline:
        (repeat 5 0.03)
        ;; peak 2:
        [0.08 0.15 0.3 0.8 2.5 1.0 0.4 0.1]
        ;; baseline:
        (repeat 8 0.04))))

;; ### Clojure: Find Local Maxima
(def clj-maxima 
  (signal/find-local-maxima-logical spectrum-data {:half-window-size 3}))

(def clj-maxima-indices 
  (filterv #(aget clj-maxima %) (range (count spectrum-data))))

;; ### R: Create spectrum in R using Clojure data
(r `(do
      (<- spectrum_intensity ~spectrum-data)
      (<- spectrum_mass 
          (bra (seq 2000 (+ 2000 (* (length spectrum_intensity) 2)) :by 2)
               (seq 1 (length spectrum_intensity))))))

;; ## 3. Peak Detection (Full Pipeline)
;;
;; The complete pipeline: local maxima + noise estimation + SNR filtering.

;; ### R Implementation (MALDIquant)
(r `(<- spec (createMassSpectrum :mass spectrum_mass :intensity spectrum_intensity)))

(r `(<- peaks (detectPeaks spec :halfWindowSize 3 :SNR 2 :method "MAD")))

(def r-peak-indices 
  (mapv dec (r->clj (r `(which (%in% spectrum_intensity (intensity peaks)))))))

(def r-peak-intensities 
  (r->clj (r `(intensity peaks))))

;; ### Clojure Implementation
(def clj-peak-indices 
  (signal/detect-peaks spectrum-data {:half-window-size 3 :snr 2}))

(def clj-peak-intensities 
  (mapv #(nth spectrum-data %) clj-peak-indices))

;; ### Comparison
(kind/table
 {:column-names ["Metric" "R (MALDIquant)" "Clojure" "Match?"]
  :row-vectors [["Peak Indices" r-peak-indices clj-peak-indices (= r-peak-indices clj-peak-indices)]
                ["Peak Intensities" r-peak-intensities clj-peak-intensities (= r-peak-intensities clj-peak-intensities)]
                ["Number of Peaks" (count r-peak-indices) (count clj-peak-indices) (= (count r-peak-indices) (count clj-peak-indices))]]})

(= r-peak-indices clj-peak-indices)

(kind/test-last [true?])

(= r-peak-intensities clj-peak-intensities)

(kind/test-last [true?])

;; ### Visualization: Detected Peaks
(-> {:index (range (count spectrum-data))
     :intensity spectrum-data
     :is-peak (mapv #(if (some #{%} clj-peak-indices) "Peak" "Non-peak") 
                    (range (count spectrum-data)))}
    tc/dataset
    (plotly/base {:=width 700 :=height 400})
    (plotly/layer-line {:=x :index 
                        :=y :intensity
                        :=mark-opacity 0.3})
    (plotly/layer-point {:=x :index 
                         :=y :intensity
                         :=color :is-peak
                         :=mark-size 10}))

;; ## 4. Different SNR Thresholds
;;
;; Testing behavior across different SNR values.

(defn compare-snr-thresholds [snr-values]
  (for [snr snr-values]
    (let [peak-var (symbol (str "peaks_snr" snr))
          r-peaks (do
                    (r `(<- ~peak-var 
                            (detectPeaks spec :halfWindowSize 3 :SNR ~snr :method "MAD")))
                    (count (r->clj (r `(intensity ~peak-var)))))
          clj-peaks (count (signal/detect-peaks spectrum-data {:half-window-size 3 :snr snr}))]
      {:snr snr
       :r-peaks r-peaks
       :clj-peaks clj-peaks
       :match? (= r-peaks clj-peaks)})))

(kind/table
 (-> (compare-snr-thresholds [1 2 3 4])
     tc/dataset))

(->> (compare-snr-thresholds [1 2 3 4])
     (every? :match?))

(kind/test-last [true?])

;; ## 5. Square Root Transformation
;;
;; Variance-stabilizing transformation for mass spectrometry data.

(def transform-test-data [1.0 4.0 9.0 16.0 25.0])

;; ### R Implementation
(r `(do
      (<- transform_mass (seq 2000 2008 :by 2))
      (<- transform_spec (createMassSpectrum 
                          :mass transform_mass 
                          :intensity ~transform-test-data))))

(def r-sqrt-transformed 
  (r->clj (r `(intensity (transformIntensity transform_spec :method "sqrt")))))

;; ### Clojure Implementation
(def clj-sqrt-transformed 
  (vec (signal/sqrt-transform transform-test-data)))

;; ### Comparison
(kind/table
 {:column-names ["Metric" "R (MALDIquant)" "Clojure" "Match?"]
  :row-vectors [["Original" transform-test-data transform-test-data true]
                ["Transformed" r-sqrt-transformed clj-sqrt-transformed (= r-sqrt-transformed clj-sqrt-transformed)]]})

(= r-sqrt-transformed clj-sqrt-transformed)

(kind/test-last [true?])

;; ## 6. Savitzky-Golay Smoothing

;; Testing Savitzky-Golay smoothing with negative clamping.
;; MALDIquant clamps negative smoothed values to 0.0.

;; Test data with noise that produces negative values when smoothed
(def sg-test-data [0.1 0.05 0.0 0.0 0.0 0.2 0.15 0.1 0.05 0.0])

;; ### R Implementation (MALDIquant)
(r `(<- sg_spec (createMassSpectrum ~(vec (range 10))
                                    ~sg-test-data)))

(def r-sg-smoothed
  (r->clj (r "intensity(smoothIntensity(sg_spec, method='SavitzkyGolay', halfWindowSize=2))")))

;; ### Clojure Implementation
(def clj-sg-smoothed
  (vec (signal/savitzky-golay-smooth sg-test-data {:window-size 5})))

;; ### Comparison
(kind/table
 {:column-names ["Index" "Original" "R Smoothed" "Clojure Smoothed" "Difference"]
  :row-vectors (map-indexed
                (fn [idx [orig r-val clj-val]]
                  [idx orig r-val clj-val (- clj-val r-val)])
                (map vector sg-test-data r-sg-smoothed clj-sg-smoothed))})

;; Check that negative clamping works (index 3)
(kind/md "**Critical: Index 3 should be 0.0 (clamped), not negative**")

;; Test exact match
(def sg-match? (every? #(< (Math/abs %) 1e-10) 
                       (map - clj-sg-smoothed r-sg-smoothed)))

(kind/table
 {:column-names ["Metric" "Value"]
  :row-vectors [["Match?" sg-match?]
                ["Max Difference" (apply max (map #(Math/abs %) (map - clj-sg-smoothed r-sg-smoothed)))]]})

sg-match?

(kind/test-last [true?])

;; ## 7. SNIP Baseline Removal

;; Testing SNIP with decreasing iteration order (MALDIquant default).
;; Window sizes iterate from large to small (iterations → 1).

;; Test data - simple peak with baseline
(def snip-test-data [0.1 0.15 0.2 0.25 0.5 0.8 0.5 0.25 0.2 0.15 0.1])

;; ### R Implementation (MALDIquant)
(r `(<- snip_spec (createMassSpectrum ~(vec (range 11))
                                      ~snip-test-data)))
(def r-snip-corrected
  (r->clj (r "intensity(removeBaseline(snip_spec, method='SNIP', iterations=10))")))

;; ### Clojure Implementation
(def clj-snip-corrected
  (vec (signal/snip-baseline-removal snip-test-data {:iterations 10})))

;; ### Comparison
(kind/table
 {:column-names ["Index" "Original" "R Corrected" "Clojure Corrected" "Difference"]
  :row-vectors (map-indexed
                (fn [idx [orig r-val clj-val]]
                  [idx orig r-val clj-val (- clj-val r-val)])
                (map vector snip-test-data r-snip-corrected clj-snip-corrected))})

;; Test exact match
(def snip-match? (every? #(< (Math/abs %) 1e-10)
                         (map - clj-snip-corrected r-snip-corrected)))
(kind/table
 {:column-names ["Metric" "Value"]
  :row-vectors [["Match?" snip-match?]
                ["Max Difference" (apply max (map #(Math/abs %) (map - clj-snip-corrected r-snip-corrected)))]]})

snip-match?

(kind/test-last [true?])

;; ## 8. TIC Normalization (Trapezoid Area)

;; Testing TIC normalization using trapezoid rule for area under curve.
;; MALDIquant normalizes AREA to 1.0, not sum of intensities.

;; Test data with non-uniform spacing
(def tic-test-masses [1.0 2.0 3.0 4.0])
(def tic-test-intensities [10.0 20.0 30.0 40.0])

;; ### R Implementation (MALDIquant)
(r `(<- tic_spec (createMassSpectrum ~tic-test-masses
                                     ~tic-test-intensities)))

(def r-tic-normalized
  (r->clj (r "intensity(calibrateIntensity(tic_spec, method='TIC'))")))

;; Verify area calculation
(def r-area (r->clj (r "sum(intensity(tic_spec))")))
(def r-normalized-area (r->clj (r "sum(intensity(calibrateIntensity(tic_spec, method='TIC')))")))
;; ### Clojure Implementation
(def clj-tic-normalized
  (vec (signal/tic-normalize tic-test-masses tic-test-intensities {:target-area 1.0})))

;; ### Comparison
(kind/table
 {:column-names ["Index" "Mass" "Original" "R Normalized" "Clojure Normalized" "Difference"]
  :row-vectors (map-indexed
                (fn [idx [mass orig r-val clj-val]]
                  [idx mass orig r-val clj-val (- clj-val r-val)])
                (map vector tic-test-masses tic-test-intensities r-tic-normalized clj-tic-normalized))})

(kind/md "**Note: Sum of normalized intensities ≠ 1.0 (area = 1.0, not sum)**")

(kind/table
 {:column-names ["Metric" "R" "Clojure"]
  :row-vectors [["Original Sum" (apply + tic-test-intensities) (apply + tic-test-intensities)]
                ["Normalized Sum" (apply + r-tic-normalized) (apply + clj-tic-normalized)]
                ["Original Area" r-area r-area]
                ["Normalized Area" r-normalized-area r-normalized-area]]})
;; Test exact match
(def tic-match? (every? #(< (Math/abs %) 1e-10)
                        (map - clj-tic-normalized r-tic-normalized)))


(kind/table
 {:column-names ["Metric" "Value"]
  :row-vectors [["Match?" tic-match?]
                ["Max Difference" (apply max (map #(Math/abs %) (map - clj-tic-normalized r-tic-normalized)))]]})

tic-match?

(kind/test-last [true?])

;; ## 9. Full Preprocessing Pipeline

;; Testing the complete preprocessing pipeline:
;; sqrt → smooth → baseline → normalize

;; Test data - realistic spectrum fragment
(def pipeline-masses (vec (range 2000.0 2100.0 1.0)))  ; 100 points, 1 Da spacing
(def pipeline-intensities 
  (vec (map (fn [m] 
              (+ 5.0  ; baseline
                 (* 0.5 (Math/sin (/ m 10.0)))  ; slow oscillation
                 (if (< (Math/abs (- m 2050)) 5) 50.0 0.0)  ; peak at 2050
                 (if (< (Math/abs (- m 2080)) 3) 30.0 0.0)  ; peak at 2080
                 0.0))  ; noise
            pipeline-masses)))

;; ### R Implementation (MALDIquant)
(r `(<- pipeline_spec (createMassSpectrum ~pipeline-masses
                                          ~pipeline-intensities)))

(def r-pipeline-result
  (r->clj (r "intensity(calibrateIntensity(
                removeBaseline(
                  smoothIntensity(
                    transformIntensity(pipeline_spec, method='sqrt'),
                    method='SavitzkyGolay', 
                    halfWindowSize=5),
                  method='SNIP', 
                  iterations=25),
                method='TIC'))")))

;; ### Clojure Implementation
(def clj-pipeline-result
  (let [spectrum {:mass pipeline-masses :intensity pipeline-intensities}
        processed (signal/preprocess-spectrum-data 
                   spectrum
                   {:should-sqrt-transform true
                    :smooth-window 11
                    :smooth-polynomial 2
                    :baseline-iterations 25
                    :should-tic-normalize true
                    :tic-target 1.0})]
    (vec (:intensity processed))))

;; ### Comparison
(kind/md "**Comparing first 20 points of preprocessed spectrum:**")

(kind/table
 {:column-names ["Index" "Mass" "R Preprocessed" "Clojure Preprocessed" "Difference"]
  :row-vectors (take 20
                     (map-indexed
                      (fn [idx [mass r-val clj-val]]
                        [idx mass r-val clj-val (- clj-val r-val)])
                      (map vector pipeline-masses r-pipeline-result clj-pipeline-result)))})

;; Test exact match across entire spectrum
(def pipeline-match? (every? #(< (Math/abs %) 1e-5)
                             (map - clj-pipeline-result r-pipeline-result)))

(def pipeline-max-diff (apply max (map #(Math/abs %) (map - clj-pipeline-result r-pipeline-result))))
(def pipeline-mean-diff (/ (apply + (map #(Math/abs %) (map - clj-pipeline-result r-pipeline-result)))
                            (count pipeline-masses)))
(kind/table
 {:column-names ["Metric" "Value"]
  :row-vectors [["Match?" pipeline-match?]
                ["Points Compared" (count pipeline-masses)]
                ["Max Difference" pipeline-max-diff]
                ["Mean Difference" pipeline-mean-diff]]})

pipeline-match?

(kind/test-last [true?])


;; ## Summary of All Test Results

(def all-test-results
  {:section-6-savgol {:match sg-match?
                      :max-diff (apply max (map #(Math/abs %) (map - clj-sg-smoothed r-sg-smoothed)))}
   :section-7-snip {:match snip-match?
                    :max-diff (apply max (map #(Math/abs %) (map - clj-snip-corrected r-snip-corrected)))}
   :section-8-tic {:match tic-match?
                   :max-diff (apply max (map #(Math/abs %) (map - clj-tic-normalized r-tic-normalized)))}
   :section-9-pipeline {:match pipeline-match?
                        :max-diff pipeline-max-diff
                        :mean-diff pipeline-mean-diff
                        :num-points (count pipeline-masses)}})

(kind/table
 {:column-names ["Section" "Match?" "Max Difference" "Notes"]
  :row-vectors [["6. Savitzky-Golay" (:match (:section-6-savgol all-test-results)) 
                 (:max-diff (:section-6-savgol all-test-results)) "Negative clamping"]
                ["7. SNIP Baseline" (:match (:section-7-snip all-test-results))
                 (:max-diff (:section-7-snip all-test-results)) "Decreasing iterations"]
                ["8. TIC Normalize" (:match (:section-8-tic all-test-results))
                 (:max-diff (:section-8-tic all-test-results)) "Trapezoid area"]
                ["9. Full Pipeline" (:match (:section-9-pipeline all-test-results))
                 (:max-diff (:section-9-pipeline all-test-results)) 
                 (str (:num-points (:section-9-pipeline all-test-results)) " points, mean=" 
                      (:mean-diff (:section-9-pipeline all-test-results)))]]})


all-test-results
