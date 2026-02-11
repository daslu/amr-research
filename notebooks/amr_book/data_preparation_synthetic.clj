;; # Data Preparation with Synthetic Spectra
;;
;; This notebook demonstrates the
;; [MALDI-TOF](https://en.wikipedia.org/wiki/Matrix-assisted_laser_desorption/ionization)
;; preprocessing pipeline using a **synthetic spectrum** —
;; no real data files are needed.
;;
;; By constructing a spectrum from known components
;; (peaks, baseline, noise), we can see exactly what
;; each preprocessing step does and verify it works
;; correctly.
;;
;; The preprocessing follows the
;; [Weis et al.](https://doi.org/10.1038/s41591-021-01619-9)
;; protocol, implemented by
;; [Ripple](https://scicloj.github.io/ripple):
;;
;; 1. Square root transform (variance stabilization)
;; 2. [Savitzky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)
;;    smoothing (half-window 10, polynomial order 3)
;; 3. [SNIP](https://doi.org/10.1016/0168-583X(88)90063-8)
;;    baseline removal (20 iterations, applied **twice**)
;; 4. [TIC](https://en.wikipedia.org/wiki/Total_ion_current) normalization
;; 5. Trim to [2000, 20000] Da
;; 6. Bin into 3 Da bins → 6000 features

(ns amr-book.data-preparation-synthetic
  (:require
   ;; Ripple MALDI signal processing (https://scicloj.github.io/ripple):
   [scicloj.ripple.maldi :as ripple]
   ;; Table processing (https://scicloj.github.io/tablecloth/):
   [tablecloth.api :as tc]
   ;; Interactive plotting via Plotly (https://scicloj.github.io/tableplot/):
   [scicloj.tableplot.v1.plotly :as plotly]
   ;; Annotating kinds of visualizations (https://scicloj.github.io/kindly-noted/):
   [scicloj.kindly.v4.kind :as kind]
   ;; High-performance array math (https://github.com/cnuernber/dtype-next):
   [tech.v3.datatype :as dtype]
   [tech.v3.datatype.functional :as dfn]))

;; ## Building a Synthetic Spectrum
;;
;; A real MALDI-TOF spectrum has three components:
;;
;; - **Peaks** — sharp signals from molecules of specific masses
;; - **Baseline** — a broad, slowly varying background from the
;;   chemical matrix used in
;;   [MALDI](https://en.wikipedia.org/wiki/Matrix-assisted_laser_desorption/ionization)
;; - **Noise** — random fluctuations from the detection process
;;
;; We build a synthetic spectrum with all three, so we can
;; watch each preprocessing step remove the artifacts
;; while preserving the true peaks.

;; ### Mass axis
;;
;; Real DRIAMS spectra span roughly 1960–20133 Da with
;; ~18000 points. For a manageable visualization, we use
;; 1800–5000 Da at 1 Da spacing.

(def raw-masses
  (vec (range 1800.0 5000.0 1.0)))

(count raw-masses)

;; ### Gaussian peak helper

(defn gaussian
  "A [Gaussian](https://en.wikipedia.org/wiki/Gaussian_function) peak
  centered at `center` with width `sigma` and `height`."
  [x center sigma height]
  (* height (Math/exp (- (/ (Math/pow (- x center) 2)
                            (* 2 sigma sigma))))))

;; ### Assembling the components
;;
;; We place five peaks of varying width and height on top of a
;; complex baseline: an exponential decay (typical of MALDI matrix
;; effects) plus sinusoidal undulations and broad humps that mimic
;; the irregular backgrounds seen in real spectra. Deterministic
;; sinusoidal "noise" is overlaid for reproducibility.

(def raw-intensities
  (mapv (fn [m]
          (let [;; Complex baseline: exponential decay + sinusoidal undulations + broad bumps
                baseline (+ (* 1500 (Math/exp (- (/ (- m 1800) 400.0))))
                            (* 150 (Math/sin (/ (- m 1800) 300.0)))
                            (* 80 (Math/cos (/ (- m 2000) 150.0)))
                            (gaussian m 2600 100.0 500.0)
                            (gaussian m 3400 80.0 400.0)
                            (gaussian m 4200 120.0 300.0))
                peak1 (gaussian m 2200 8.0 600.0)
                peak2 (gaussian m 2800 15.0 350.0)
                peak3 (gaussian m 3200 5.0 800.0)
                peak4 (gaussian m 3800 20.0 250.0)
                peak5 (gaussian m 4400 10.0 450.0)
                noise (* 18.0 (+ (Math/sin (* m 0.3))
                                 (* 0.7 (Math/sin (* m 1.1)))
                                 (* 0.3 (Math/sin (* m 2.7)))))]
            (max 0.0 (+ baseline peak1 peak2 peak3 peak4 peak5 noise))))
        raw-masses))

;; ### The raw spectrum

(def raw-spectrum
  (tc/dataset {:mass raw-masses :intensity raw-intensities}))

raw-spectrum

(-> raw-spectrum
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=title "Raw synthetic spectrum"
                  :=x-title "m/z (Da)"
                  :=y-title "Intensity (a.u.)"})
    (plotly/layer-line {:=mark-opacity 0.8})
    plotly/plot)

;; The baseline has an exponential decay, broad humps, and wavy
;; undulations — obscuring the true peak heights. Noise adds
;; spurious wiggles. The region below 2000 Da is dominated by
;; matrix signal.

;; ## Step 1 — Square Root Transform
;;
;; MALDI-TOF measures ion counts, which follow
;; [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution)-like
;; statistics: the variance grows with the mean. A
;; [variance-stabilizing](https://en.wikipedia.org/wiki/Variance-stabilizing_transformation)
;; square root compresses the dynamic range so that high
;; and low peaks have comparable noise levels.
;;
;; **y' = √y**

(def sqrt-intensities
  (ripple/sqrt-transform raw-intensities))

(tc/dataset {:mass raw-masses
             :raw raw-intensities
             :sqrt sqrt-intensities})

;; Dynamic range before and after:

{:raw-range [(dfn/reduce-min raw-intensities) (dfn/reduce-max raw-intensities)]
 :sqrt-range [(dfn/reduce-min sqrt-intensities) (dfn/reduce-max sqrt-intensities)]}

(-> (tc/dataset {:mass raw-masses :intensity sqrt-intensities})
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=title "After square root transform"
                  :=x-title "m/z (Da)"
                  :=y-title "√Intensity"})
    (plotly/layer-line {:=mark-opacity 0.8})
    plotly/plot)

;; The tallest peak (raw ~1300) and smallest (raw ~250) are now
;; closer together, giving downstream algorithms equal sensitivity
;; across the spectrum.

;; ## Step 2 — Savitzky-Golay Smoothing
;;
;; The [Savitzky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)
;; filter fits a local polynomial to each sliding window of points
;; and replaces the center with the fitted value. It removes
;; high-frequency noise while preserving peak shapes.
;;
;; The Weis et al. protocol uses:
;; - **Half-window size 10** → full window of **21** points
;; - **Polynomial order 3**
;;   ([MALDIquant](https://strimmerlab.github.io/software/maldiquant/)
;;   default)

(def smoothed-intensities
  (ripple/savitzky-golay-smooth sqrt-intensities
                                {:window-size 21
                                 :polynomial-order 3}))

(tc/dataset {:mass raw-masses
             :sqrt sqrt-intensities
             :smoothed smoothed-intensities})

;; ### Effect of window size
;;
;; Larger windows smooth more aggressively but can broaden peaks.
;; Let's compare three window sizes zoomed to the sharpest peak
;; (σ=5, centered at 3200 Da):

(def sg-narrow
  (ripple/savitzky-golay-smooth sqrt-intensities
                                {:window-size 5 :polynomial-order 3}))

(def sg-default
  smoothed-intensities)

(def sg-wide
  (ripple/savitzky-golay-smooth sqrt-intensities
                                {:window-size 41 :polynomial-order 3}))

(let [idx-start 1370 idx-end 1430
      sub-m (dtype/sub-buffer raw-masses idx-start (- idx-end idx-start))
      sub-sqrt (dtype/sub-buffer sqrt-intensities idx-start (- idx-end idx-start))
      sub-narrow (dtype/sub-buffer sg-narrow idx-start (- idx-end idx-start))
      sub-default (dtype/sub-buffer sg-default idx-start (- idx-end idx-start))
      sub-wide (dtype/sub-buffer sg-wide idx-start (- idx-end idx-start))
      nm (count sub-m)]
  (-> (tc/dataset
       {:mass (vec (concat sub-m sub-m sub-m sub-m))
        :intensity (vec (concat sub-sqrt sub-narrow sub-default sub-wide))
        :series (vec (concat (repeat nm "input (sqrt)")
                             (repeat nm "window=5")
                             (repeat nm "window=21 (Weis et al.)")
                             (repeat nm "window=41")))})
      (plotly/base {:=x :mass
                    :=y :intensity
                    :=color :series
                    :=title "Smoothing comparison — peak at 3200 Da"
                    :=x-title "m/z (Da)"
                    :=y-title "Intensity"})
      (plotly/layer-line {:=mark-opacity 0.8})
      plotly/plot))

;; Window=5 leaves visible noise, window=41 broadens the peak.
;; Window=21 (the paper's choice) balances noise removal and
;; peak preservation.

;; Full spectrum after smoothing:

(-> (tc/dataset {:mass raw-masses :intensity smoothed-intensities})
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=title "After Savitzky-Golay smoothing (window=21, order=3)"
                  :=x-title "m/z (Da)"
                  :=y-title "Intensity"})
    (plotly/layer-line {:=mark-opacity 0.8})
    plotly/plot)

;; ## Step 3 — SNIP Baseline Removal (Twice)
;;
;; The [SNIP algorithm](https://doi.org/10.1016/0168-583X(88)90063-8)
;; (Statistics-sensitive Non-linear Iterative Peak-clipping)
;; estimates the baseline by iteratively "eroding" peaks.
;; At each iteration with window half-width *w*, each point is
;; compared to the average of its neighbors *w* steps away:
;;
;; **baseline[i] = min(baseline[i], (baseline[i-w] + baseline[i+w]) / 2)**
;;
;; Points that are part of a peak get clipped down.
;; After all iterations, only the slow-varying baseline remains.
;;
;; The paper's [actual R code](https://github.com/BorgwardtLab/maldi_amr/blob/master/amr_maldi_ml/DRIAMS_preprocessing/DRIAMS-A_2018_preprocessed.r)
;; applies SNIP **twice**: the second pass catches residual
;; baseline that the first pass left behind.

;; ### First SNIP pass

(def baseline-corrected-once
  (ripple/snip-baseline-removal smoothed-intensities
                                {:iterations 20}))

;; ### Second SNIP pass

(def baseline-corrected
  (ripple/snip-baseline-removal baseline-corrected-once
                                {:iterations 20}))

(tc/dataset {:mass raw-masses
             :smoothed smoothed-intensities
             :after-1st-snip baseline-corrected-once
             :after-2nd-snip baseline-corrected})

;; ### Visualizing the two passes
;;
;; The estimated baselines (input minus corrected) show how
;; the second pass refines the estimate:

(let [n (count raw-masses)]
  (-> (tc/dataset
       {:mass (vec (concat raw-masses raw-masses raw-masses))
        :intensity (vec (concat smoothed-intensities
                                baseline-corrected-once
                                baseline-corrected))
        :step (vec (concat (repeat n "before (smoothed)")
                           (repeat n "after 1st SNIP")
                           (repeat n "after 2nd SNIP")))})
      (plotly/base {:=x :mass
                    :=y :intensity
                    :=color :step
                    :=title "Baseline removal — single vs double SNIP"
                    :=x-title "m/z (Da)"
                    :=y-title "Intensity"})
      (plotly/layer-line {:=mark-opacity 0.7})
      plotly/plot))

;; The first pass removes the bulk of the baseline. The second
;; pass is a subtle refinement — it shaves off residual
;; curvature that the first pass left behind. The effect is
;; small but systematic, bringing inter-peak regions slightly
;; closer to zero.

;; Verify the background region (away from peaks) is near zero:

(let [region (dtype/sub-buffer baseline-corrected 700 100)
      mean-val (/ (dfn/sum region) (count region))]
  {:background-mean mean-val
   :near-zero? (< (Math/abs mean-val) 1.0)})

(kind/test-last
 #(:near-zero? %))

;; ## Step 4 — TIC Normalization
;;
;; [Total Ion Current](https://en.wikipedia.org/wiki/Total_ion_current)
;; normalization scales each spectrum so that its area
;; (by the [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule))
;; equals 1.0. This makes spectra from different measurements
;; comparable, removing differences due to sample amount or
;; instrument sensitivity.
;;
;; Unlike the other functions, `tic-normalize` needs both
;; masses and intensities — the m/z spacings are required
;; to compute the trapezoid area correctly.

(def masses raw-masses)

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
    (plotly/layer-line {:=mark-opacity 0.8})
    plotly/plot)

;; ### Why normalization matters
;;
;; Two measurements of the same sample can differ in overall
;; intensity. TIC normalization removes this difference:

(let [spectrum-a baseline-corrected
      spectrum-b (dfn/* baseline-corrected 4.0)
      norm-a (ripple/tic-normalize masses spectrum-a {:target-area 1.0})
      norm-b (ripple/tic-normalize masses spectrum-b {:target-area 1.0})
      max-diff (dfn/reduce-max (dfn/abs (dfn/- norm-a norm-b)))]
  {:max-difference-after-normalization max-diff
   :identical? (< max-diff 1e-10)})

(kind/test-last
 #(:identical? %))

;; After normalization, the two spectra are identical —
;; the 4× intensity difference has been removed.

;; ## All Steps Combined
;;
;; Ripple's `preprocess-spectrum-data` applies the square root,
;; smoothing, baseline removal, and normalization in one call.
;;
;; **Note:** the current version applies SNIP once (not twice
;; as in the paper's R code). This is a known simplification;
;; a future Ripple version will support double SNIP.

(def preprocessed
  (ripple/preprocess-spectrum-data
   raw-spectrum
   {:should-sqrt-transform true
    :smooth-window 21
    :smooth-polynomial 3
    :baseline-iterations 20
    :should-tic-normalize true}))

preprocessed

(-> preprocessed
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=title "After full preprocessing (single SNIP)"
                  :=x-title "m/z (Da)"
                  :=y-title "Intensity (normalized)"})
    (plotly/layer-line {:=mark-opacity 0.8})
    plotly/plot)

;; ## Trimming and Binning
;;
;; For machine learning, spectra must be converted to
;; fixed-length feature vectors. The DRIAMS protocol:
;;
;; 1. **Trim** to [2000, 20000] Da — removing the low-mass
;;    matrix region
;; 2. **Bin** into 3 Da wide bins — producing 6000 features

;; ### Trimming
;;
;; Our synthetic spectrum extends below 2000 Da. Trimming
;; removes those points:

(def normalized-spectrum
  (tc/dataset {:mass masses :intensity normalized-intensities}))

(def trimmed
  (ripple/trim-spectrum normalized-spectrum {:range [2000 5000]}))

{:points-before (tc/row-count normalized-spectrum)
 :points-after (tc/row-count trimmed)}

(-> trimmed
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=title "Trimmed to [2000, 5000] Da"
                  :=x-title "m/z (Da)"
                  :=y-title "Intensity (normalized)"})
    (plotly/layer-line {:=mark-opacity 0.8})
    plotly/plot)

;; ### Binning
;;
;; The m/z axis is partitioned into 3 Da bins and the
;; intensities in each bin are **summed**. For our range
;; [2000, 5000] this produces 1000 bins. (With the full
;; DRIAMS range [2000, 20000] there would be 6000 bins.)

(def binned
  (ripple/bin-spectrum trimmed {:range [2000 5000] :step 3}))

(count binned)

(kind/test-last
 #(= % 1000))

;; First 10 bin values:

(vec (take 10 binned))

;; ### Comparing continuous vs binned
;;
;; Zooming to the peak at 3200 Da shows how binning
;; approximates the continuous shape:

(let [bin-min 2000
      bin-step 3
      n-bins (count binned)
      bin-centers (mapv #(+ bin-min (* (+ % 0.5) bin-step)) (range n-bins))
      zoom-min 3150 zoom-max 3250
      cont-masses (:mass trimmed)
      cont-intens (:intensity trimmed)
      cont-idx (filterv #(and (>= (cont-masses %) zoom-min)
                              (<= (cont-masses %) zoom-max))
                        (range (count cont-masses)))
      cm (mapv #(cont-masses %) cont-idx)
      ci (mapv #(cont-intens %) cont-idx)
      bin-idx (filterv #(and (>= (nth bin-centers %) zoom-min)
                             (<= (nth bin-centers %) zoom-max))
                       (range n-bins))
      bm (mapv #(nth bin-centers %) bin-idx)
      bi (mapv #(aget binned %) bin-idx)
      nc (count cm) nb (count bm)]
  (-> (tc/dataset
       {:mass (vec (concat cm bm))
        :intensity (vec (concat ci bi))
        :series (vec (concat (repeat nc "continuous (1 Da)")
                             (repeat nb "binned (3 Da)")))})
      (plotly/base {:=x :mass
                    :=y :intensity
                    :=color :series
                    :=title "Continuous vs binned — peak at 3200 Da"
                    :=x-title "m/z (Da)"
                    :=y-title "Intensity"})
      (plotly/layer-line {:=mark-opacity 0.7})
      (plotly/layer-point {:=mark-size 4 :=mark-opacity 0.5})
      plotly/plot))

;; The binned version has one value per 3 Da, preserving
;; the peak shape while producing a fixed-length vector
;; suitable for machine learning.

;; ### The full binned feature vector

(-> (tc/dataset {:bin (range (count binned))
                 :intensity (vec binned)})
    (plotly/base {:=x :bin
                  :=y :intensity
                  :=title "Binned feature vector (1000 features)"
                  :=x-title "Bin index"
                  :=y-title "Summed intensity"})
    (plotly/layer-line {:=mark-opacity 0.8})
    plotly/plot)

;; This is the representation fed to the classifier:
;; a fixed-length vector where each element is the summed
;; intensity in a 3 Da wide bin. In the real DRIAMS pipeline,
;; this would be 6000 features spanning [2000, 20000] Da.

;; ## Summary
;;
;; The preprocessing pipeline transforms a noisy, baseline-
;; contaminated raw spectrum into a clean, normalized,
;; fixed-length feature vector:
;;
;; | Step | Operation | Effect |
;; |-----:|:----------|:-------|
;; | 1 | Square root | Stabilize variance |
;; | 2 | Savitzky-Golay (window 21, order 3) | Remove noise |
;; | 3 | SNIP baseline × 2 (20 iterations) | Remove background |
;; | 4 | TIC normalization | Area → 1.0 |
;; | 5 | Trim [2000, 20000] Da | Focus on analysis range |
;; | 6 | Bin 3 Da → 6000 features | Fixed-length vector |
;;
;; Each step is essential: without variance stabilization,
;; small peaks are invisible; without baseline removal, peak
;; heights are unreliable; without normalization, spectra
;; from different measurements are incomparable; without
;; binning, the ML classifier cannot accept the input.

