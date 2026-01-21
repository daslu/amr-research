(ns maldi.data.signal
  "Signal processing functions for MALDI spectra.
   Pure JVM implementations of MALDIquant preprocessing algorithms."
  (:require [tech.v3.tensor :as tensor]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.functional :as dfn]
            [clojure.tools.logging :as log]
            [tablecloth.api :as tc]
            [tablecloth.column.api :as tcc]
            [clojure.math :as math])
  (:import [com.github.psambit9791.jdsp.filter Savgol]
           [com.github.psambit9791.jdsp.misc UtilMethods]))

(defn sqrt-transform
  "Args:
   - intensities

   Apply square root transformation to intensities.
   Common preprocessing step to reduce noise and improve peak shape."
  [intensities]
  (tcc/sqrt intensities))

(defn savitzky-golay-smooth
  "Apply Savitzky-Golay smoothing to spectrum intensities.
   
   Args:
   - intensities
   - window-size: smoothing window size (must be odd)
   - polynomial-order: polynomial order
   
   Returns: smoothed intensities"
  [intensities {:keys [window-size polynomial-order]
                :or {window-size 11
                     polynomial-order 2}}]

  ;; Validate parameters
  (when (even? window-size)
    (throw (ex-info "Window size must be odd" {:window-size window-size})))

  (let [n (dtype/ecount intensities)]
    (when (< n window-size)
      (throw (ex-info "Not enough data points for smoothing window"
                      {:data-points n
                       :window-size window-size}))))

  ;; Use JDSP Savgol filter
  (let [savgol (Savgol. window-size polynomial-order)]
    (.filter savgol (double-array intensities))))

(defn snip-baseline-removal
  "Remove baseline using improved SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) algorithm.
   
   This implementation follows the SNIP algorithm more closely:
   1. Creates a working intensities copy
   2. Iteratively clips peaks by comparing each point to linear interpolation of surrounding points
   3. Uses increasing window sizes over iterations
   4. Returns baseline-corrected intensities (original - baseline)
   
   Args:
   - intensities
   - iterations: number of SNIP iterations
   - decreasing: whether to process in decreasing order
   
   Returns: baseline-corrected intensities tensor"
  [intensities {:keys [iterations decreasing]
                :or {iterations 25
                     decreasing false}}]

  (when (<= iterations 0)
    (throw (ex-info "Iterations must be positive" {:iterations iterations})))

  (let [n (dtype/ecount intensities)]

    (when (< n 3)
      (throw (ex-info "Need at least 3 data points for SNIP baseline removal"
                      {:data-points n})))

    ;; Initialize working copy (will become the baseline estimate)
    (let [working (double-array intensities)
          indices (if decreasing
                    (reverse (range n))
                    (range n))]

      ;; SNIP iterations with increasing window size
      (doseq [window-half (range 1 (inc iterations))]
        (doseq [i (-> (range window-half (- n window-half))
                      (cond-> decreasing reverse))]
          (let [;; Linear interpolation between surrounding points
                interpolated (* 0.5 (+ (aget working (- i window-half))
                                       (aget working (+ i window-half))))

                current-val (aget working i)]

            ;; SNIP clipping: keep minimum of current and interpolated
            (aset working i (min current-val interpolated)))))

      ;; Return baseline-corrected intensities (original - baseline)
      (tcc/- intensities working))))

(defn tic-normalize
  "Normalize intensities using Total Ion Current (TIC).
   Scales all intensities so that the total sum equals the target value.
   
   Args:
   - intensities
   - target-sum: desired total sum
   
   Returns: normalized intensities tensor
   
   Note: If all intensities are zero, returns zeros (graceful handling)"
  ([intensities {:keys [target-sum]
                 :or {target-sum 1}}]

   (let [current-sum (tcc/sum intensities)]
     (if (zero? current-sum)
       (do
         (log/warn "Cannot normalize zero signal - all intensities are zero. Returning zeros.")
         (dtype/make-reader :float32
                            (dtype/ecount intensities)
                            0.0))

       ;; else
       (do (when (< current-sum 1e-10)
             (log/warn (format "Very small intensity sum detected: %e. Normalization may be unstable." current-sum)))
           (tcc// intensities
                  (tcc/sum intensities)))))))

(defn median-filter
  "Apply median filter for noise reduction.
   
   Args:
   - intensities: tensor of intensity values  
   - window-size: filter window size (must be odd)
   
   Returns: median-filtered intensities tensor"
  [intensities {:keys [window-size]
                :or {window-size 5}}]

  (when (even? window-size)
    (throw (ex-info "Window size must be odd" {:window-size window-size})))

  (let [n (dtype/ecount intensities)
        half-window (quot window-size 2)
        result (double-array n)]

    (doseq [i (range n)]
      (let [start (max 0 (- i half-window))
            end (min n (+ i half-window 1))
            window-vals (dtype/sub-buffer intensities
                                          start
                                          (- end start))]
        (aset result i (tcc/median window-vals))))

    result))

;; ============================================================================
;; MALDIquant-compatible Peak Detection Functions
;; ============================================================================

(defn- find-max-index-in-window
  "Find the index of maximum value in a window of intensities.
   
   Args:
   - intensities: the intensity buffer
   - start: window start index (inclusive)
   - end: window end index (exclusive)
   
   Returns: absolute index of maximum value within [start, end)"
  [intensities start end]
  (loop [i (inc start)
         max-idx start
         max-val (dtype/get-value intensities start)]
    (if (>= i end)
      max-idx
      (let [val (dtype/get-value intensities i)]
        (if (> val max-val)
          (recur (inc i) i val)
          (recur (inc i) max-idx max-val))))))

(defn find-local-maxima-logical
  "Find local maxima using sliding window approach (MALDIquant algorithm).
   
   For each position i, checks if it is the maximum value within the window
   [i-halfWindowSize, i+halfWindowSize]. Returns a logical vector indicating
   which positions are local maxima.
   
   Args:
   - intensities: intensity values
   - half-window-size: half the window size for local maximum detection
   
   Returns: boolean array where true indicates a local maximum"
  [intensities {:keys [half-window-size]
                :or {half-window-size 20}}]
  (let [n (dtype/ecount intensities)
        result (boolean-array n false)]

    (when (pos? n)
      (doseq [i (range n)]
        (let [window-start (max 0 (- i half-window-size))
              window-end (min n (inc (+ i half-window-size)))
              max-idx (find-max-index-in-window intensities window-start window-end)]

          ;; Mark as local maximum if this position is the window's maximum
          (when (= i max-idx)
            (aset result i true)))))

    result))

(defn estimate-noise-mad
  "Estimate noise level using MAD (Median Absolute Deviation) method.
   
   This follows MALDIquant's MAD noise estimation:
   noise[i] = median(|intensities - median(intensities)|) * 1.4826
   
   The constant 1.4826 makes MAD consistent with standard deviation
   for normally distributed data.
   
   Args:
   - intensities: intensity values
   - half-window-size: half window size for local noise estimation (optional)
   
   Returns: scalar noise estimate or vector of local noise estimates"
  [intensities {:keys [half-window-size]
                :or {half-window-size nil}}]

  (if half-window-size
    ;; Local noise estimation (moving window)
    (let [n (dtype/ecount intensities)
          result (double-array n)]
      (doseq [i (range n)]
        (let [window-start (max 0 (- i half-window-size))
              window-end (min n (inc (+ i half-window-size)))
              window (dtype/sub-buffer intensities window-start (- window-end window-start))
              med (tcc/median window)
              abs-dev (tcc/abs (tcc/- window med))
              mad (tcc/median abs-dev)]
          (aset result i (* mad 1.4826))))
      result)

    ;; Global noise estimation
    (let [med (tcc/median intensities)
          abs-dev (tcc/abs (tcc/- intensities med))
          mad (tcc/median abs-dev)]
      (* mad 1.4826))))

(defn filter-peaks-by-snr
  "Filter peak candidates by Signal-to-Noise Ratio threshold.
   
   Args:
   - intensities: intensity values
   - is-local-maxima: boolean array indicating local maxima positions
   - noise: noise estimate (scalar or array)
   - snr-threshold: minimum SNR for a peak to be accepted
   
   Returns: vector of indices that are both local maxima and above SNR threshold"
  [intensities is-local-maxima noise {:keys [snr-threshold]
                                      :or {snr-threshold 2}}]
  (let [n (dtype/ecount intensities)
        noise-scalar? (number? noise)]

    (filterv
     identity
     (for [i (range n)
           :when (aget is-local-maxima i)]
       (let [intensity (dtype/get-value intensities i)
             noise-val (if noise-scalar? noise (aget noise i))
             snr (if (pos? noise-val) (/ intensity noise-val) 0.0)]
         (when (>= snr snr-threshold)
           i))))))

(defn detect-peaks
  "Detect peaks in spectrum using MALDIquant-compatible algorithm.
   
   This implements the MALDIquant peak detection pipeline:
   1. Find local maxima using sliding window
   2. Estimate noise level using MAD
   3. Filter peaks by SNR threshold
   
   Args:
   - intensities: intensity values
   - options: map with keys:
     - :half-window-size (default 20): window size for local maximum detection
     - :snr (default 2): Signal-to-Noise Ratio threshold
     - :noise-method (default :mad-global): :mad-global or :mad-local
   
   Returns: vector of peak indices"
  [intensities {:keys [half-window-size snr noise-method]
                :or {half-window-size 20
                     snr 2
                     noise-method :mad-global}
                :as options}]

  (when (< (dtype/ecount intensities) 1)
    (throw (ex-info "Cannot detect peaks in empty data" {})))

  ;; Step 1: Find local maxima
  (let [is-local-maxima (find-local-maxima-logical intensities
                                                   {:half-window-size half-window-size})

        ;; Step 2: Estimate noise
        noise (case noise-method
                :mad-global (estimate-noise-mad intensities {})
                :mad-local (estimate-noise-mad intensities
                                               {:half-window-size half-window-size})
                (estimate-noise-mad intensities {}))

        ;; Step 3: Filter by SNR
        peak-indices (filter-peaks-by-snr intensities
                                          is-local-maxima
                                          noise
                                          {:snr-threshold snr})]

    peak-indices))

(defn preprocess-spectrum-data
  "Apply complete preprocessing pipeline to spectrum data.
   
   Args:
   - intensities
   - options: map with preprocessing options
     - :sqrt-transform (boolean)
     - :smooth-window (integer) 
     - :smooth-polynomial (integer)
     - :baseline-iterations (integer)
     - :tic-normalize (boolean)
     - :tic-target (number)
   
   Returns: map with :masses and :intensities keys"
  [intensities {:as options
                :keys [should-sqrt-transform smooth-window smooth-polynomial
                       baseline-iterations should-tic-normalize tic-target]
                :or {should-sqrt-transform true
                     smooth-window 11
                     smooth-polynomial 2
                     baseline-iterations 25
                     should-tic-normalize true
                     tic-target 1.0}}]
  ;; errors/with-error-handling errors/pipeline-error {:operation :preprocess-spectrum-data
  ;;                                                   :options options}

  #_(log/info "Starting spectrum preprocessing pipeline")

  (cond-> intensities
    should-sqrt-transform (sqrt-transform)
    true (savitzky-golay-smooth {:window-size smooth-window
                                 :polynomial-order smooth-polynomial})
    true (snip-baseline-removal {:iterations baseline-iterations})
    should-tic-normalize (tic-normalize {:target-sum tic-target})))
