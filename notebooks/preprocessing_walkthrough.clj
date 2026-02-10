(ns preprocessing-walkthrough
  "Step-by-step walkthrough of the MALDI-TOF spectrum preprocessing pipeline.
   
   This notebook demonstrates each preprocessing step from the Weis et al. DRIAMS paper:
   1. Square root transformation
   2. Savitzky-Golay smoothing
   3. SNIP baseline removal
   4. TIC normalization
   5. Complete pipeline
   
   Each step is shown with before/after examples to illustrate the transformation."
  (:require [maldi.data.signal :as signal]
            [maldi.data.ingestion :as ingestion]
            [tablecloth.api :as tc]
            [scicloj.kindly.v4.kind :as kind]))

;; ## Load Sample Spectrum
;;
;; We'll use a real MALDI-TOF spectrum from the DRIAMS-A dataset.
;; Raw spectra are stored as gzipped text files with mass and intensity values.

(def test-spectrum-path 
  "/workspace/datasets/DRIAMS/DRIAMS-A/raw/2018/00006690-1411-4a89-87cc-ab84678cc9fb_MALDI1.txt.gz")

(def raw-spectrum (ingestion/load-raw-spectrum test-spectrum-path))

[(tc/row-count raw-spectrum) 
 (tc/column-names raw-spectrum)]

;; Show first few data points:
(tc/head raw-spectrum 10)

;; Extract mass and intensity columns for step-by-step processing:
(def masses (:mass raw-spectrum))
(def intensities (:intensity raw-spectrum))

[(count masses) (count intensities)]

;; ## Step 1: Square Root Transformation
;;
;; **Purpose**: Variance stabilization - reduces the influence of high-intensity peaks
;; and makes the spectrum more suitable for downstream processing.
;;
;; **Formula**: `sqrt(intensity)`

(def step1-sqrt (signal/sqrt-transform intensities))

;; Compare raw vs sqrt-transformed (first 10 values):
(kind/table
  {:column-names ["Index" "Raw Intensity" "Sqrt Intensity"]
   :row-vectors (map-indexed 
                  (fn [i raw] 
                    [i raw (nth step1-sqrt i)])
                  (take 10 intensities))})

;; Notice how the sqrt transformation compresses the range of values.

;; ## Step 2: Savitzky-Golay Smoothing
;;
;; **Purpose**: Noise reduction using polynomial smoothing.
;; **Parameters**: 
;; - Half-window size: 10 (total window = 2*10+1 = 21 points)
;; - Polynomial order: 3 (default)
;;
;; This smooths the spectrum while preserving peak shapes.

(def step2-smooth 
  (signal/savitzky-golay-smooth step1-sqrt {:half-window 10}))

;; Compare sqrt vs smoothed (window around index 1000):
(kind/table
  {:column-names ["Index" "Sqrt" "Smoothed"]
   :row-vectors (map-indexed 
                  (fn [i sqrt-val] 
                    [(+ i 1000) sqrt-val (nth step2-smooth (+ i 1000))])
                  (take 10 (drop 1000 step1-sqrt)))})

;; The smoothed values are more stable - rapid fluctuations are removed.

;; ## Step 3: SNIP Baseline Removal
;;
;; **Purpose**: Remove baseline signal (background chemical noise).
;; **Algorithm**: Statistics-sensitive Non-linear Iterative Peak-clipping
;; **Parameters**: 20 iterations (decreasing window sizes)
;;
;; SNIP estimates and subtracts the baseline, leaving only peak signals.

(def step3-baseline 
  (signal/snip-baseline-removal step2-smooth {:iterations 20}))

;; Compare smoothed vs baseline-removed:
(kind/table
  {:column-names ["Index" "Smoothed" "After Baseline Removal"]
   :row-vectors (map-indexed 
                  (fn [i smooth-val] 
                    [(+ i 1000) smooth-val (nth step3-baseline (+ i 1000))])
                  (take 15 (drop 1000 step2-smooth)))})

;; Notice how baseline removal zeros out low-lying regions and preserves peaks.

;; ## Step 4: TIC Normalization
;;
;; **Purpose**: Normalize total intensity across samples for comparability.
;; **Method**: Scale intensities so trapezoid area under curve = target (1.0)
;;
;; **Formula**: `intensity_new = intensity * (target / current_area)`

(def step4-tic 
  (signal/tic-normalize masses step3-baseline {:target-area 1.0}))

;; Verify normalization - calculate trapezoid area:
(defn trapezoid-area [masses intensities]
  (let [n (count masses)]
    (reduce + (for [i (range (dec n))]
                (* 0.5 
                   (- (nth masses (inc i)) (nth masses i))
                   (+ (nth intensities i) (nth intensities (inc i))))))))

(def area-before (trapezoid-area masses step3-baseline))
(def area-after (trapezoid-area masses step4-tic))

(kind/table
  {:column-names ["Stage" "Trapezoid Area"]
   :row-vectors [["Before TIC" area-before]
                 ["After TIC" area-after]]})

;; The area is now normalized to 1.0 (within floating-point precision).

;; Compare intensities before/after normalization:
(kind/table
  {:column-names ["Index" "Before TIC" "After TIC" "Ratio"]
   :row-vectors (map-indexed 
                  (fn [i before] 
                    (let [after (nth step4-tic i)
                          ratio (if (pos? before) (/ after before) 0.0)]
                      [(+ i 5000) before after ratio]))
                  (take 10 (drop 5000 step3-baseline)))})

;; All intensities are scaled by the same factor.

;; ## Step 5: Complete Pipeline
;;
;; The `preprocess-spectrum-data` function combines all steps and adds trimming.
;; **Additional step**: Trim to [2000, 20000] Da range (DRIAMS paper specification)

(def preprocessed 
  (signal/preprocess-spectrum-data 
    raw-spectrum
    {:trim-range [2000 20000]
     :sqrt-transform? true
     :smooth? true
     :smooth-method :savitzky-golay
     :smooth-half-window 10
     :remove-baseline? true
     :baseline-iterations 20
     :tic-normalize? true
     :tic-target 1.0}))

[(tc/row-count preprocessed) 
 (tc/column-names preprocessed)]

;; Check the mass range after trimming:
(let [mass-col (:mass preprocessed)]
  {:min (apply min mass-col)
   :max (apply max mass-col)
   :count (count mass-col)})

(tc/head preprocessed 10)

;; ## Comparison: Raw vs Preprocessed
;;
;; Let's compare the same region of the spectrum before and after preprocessing.

(defn find-index-near-mass [masses target-mass]
  (first (keep-indexed 
           (fn [i m] (when (>= m target-mass) i))
           masses)))

;; Find indices near mass 5000 Da in both spectra:
(def raw-idx (find-index-near-mass (:mass raw-spectrum) 5000))
(def prep-idx (find-index-near-mass (:mass preprocessed) 5000))

(kind/table
  {:column-names ["Spectrum" "Index" "Mass" "Intensity"]
   :row-vectors 
   (concat
     [["Raw" raw-idx 
       (nth (:mass raw-spectrum) raw-idx)
       (nth (:intensity raw-spectrum) raw-idx)]]
     (for [offset (range 1 6)]
       ["Raw" (+ raw-idx offset)
        (nth (:mass raw-spectrum) (+ raw-idx offset))
        (nth (:intensity raw-spectrum) (+ raw-idx offset))])
     [["---" "---" "---" "---"]]
     [["Preprocessed" prep-idx
       (nth (:mass preprocessed) prep-idx)
       (nth (:intensity preprocessed) prep-idx)]]
     (for [offset (range 1 6)]
       ["Preprocessed" (+ prep-idx offset)
        (nth (:mass preprocessed) (+ prep-idx offset))
        (nth (:intensity preprocessed) (+ prep-idx offset))]))})

;; ## Summary
;;
;; The complete preprocessing pipeline from Weis et al. (2020):
;;
;; | Step | Function | Purpose | Parameters |
;; |------|----------|---------|------------|
;; | 1 | `sqrt-transform` | Variance stabilization | - |
;; | 2 | `savitzky-golay-smooth` | Noise reduction | half-window=10 |
;; | 3 | `snip-baseline-removal` | Remove baseline | iterations=20 |
;; | 4 | `tic-normalize` | Intensity normalization | target-area=1.0 |
;; | 5 | Trim range | Keep relevant m/z range | [2000, 20000] Da |
;;
;; **Output**: A preprocessed spectrum ready for binning and machine learning.
;;
;; **Next step**: The preprocessed spectrum will be binned into fixed-width bins
;; (step=3 Da) to create a fixed-length feature vector for ML models.

;; ## Optional: Disable Individual Steps
;;
;; You can selectively disable preprocessing steps:

(def preprocessed-minimal
  (signal/preprocess-spectrum-data 
    raw-spectrum
    {:trim-range [2000 20000]
     :sqrt-transform? true
     :smooth? false              ; Skip smoothing
     :remove-baseline? false     ; Skip baseline removal
     :tic-normalize? true}))

[(tc/row-count preprocessed-minimal)
 {:with-all-steps (tc/row-count preprocessed)
  :minimal (tc/row-count preprocessed-minimal)}]
