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
  "Apply Savitzky-Golay smoothing to spectrum intensities - MALDIquant compatible.
   
   MALDIquant clamps negative smoothed values to 0.0 (negative intensities are not physical).
   
   Args:
   - intensities: intensity values
   - window-size: smoothing window size (must be odd)
   - polynomial-order: polynomial order for fitting
   
   Returns: smoothed intensities (with negative values clamped to 0)"
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

  ;; Use JDSP Savgol filter and clamp negative values to 0
  (let [savgol (Savgol. window-size polynomial-order)
        smoothed (.filter savgol (double-array intensities))]
    ;; Clamp negative values to 0.0 (MALDIquant behavior)
    (double-array (map #(max 0.0 %) smoothed))))

(defn snip-baseline-removal
  "Remove baseline using SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) algorithm.
   
   Implements MALDIquant-compatible SNIP algorithm:
   1. Creates a working intensities copy (baseline estimate)
   2. Iteratively clips peaks by comparing each point to linear interpolation of surrounding points
   3. Window size iteration order controlled by decreasing parameter
   4. Returns baseline-corrected intensities (original - baseline)
   
   Args:
   - intensities: spectrum intensities
   - iterations: number of SNIP iterations (controls max window size)
   - decreasing: if true (MALDIquant default), iterate from large to small windows;
                 if false, iterate from small to large windows
   
   Returns: baseline-corrected intensities tensor"
  [intensities {:keys [iterations decreasing]
                :or {iterations 25
                     decreasing true}}]

  (when (<= iterations 0)
    (throw (ex-info "Iterations must be positive" {:iterations iterations})))

  (let [n (dtype/ecount intensities)]

    (when (< n 3)
      (throw (ex-info "Need at least 3 data points for SNIP baseline removal"
                      {:data-points n})))

    ;; Initialize working copy (will become the baseline estimate)
    (let [working (double-array intensities)
          ;; Window iteration order: decreasing=true means large→small windows (MALDIquant default)
          window-sequence (if decreasing
                            (reverse (range 1 (inc iterations)))
                            (range 1 (inc iterations)))]

      ;; SNIP iterations - use temp array to avoid in-place updates
      (doseq [window-half window-sequence]
        (let [temp (double-array working)]
          (doseq [i (range window-half (- n window-half))]
            (let [;; Linear interpolation between surrounding points
                  interpolated (* 0.5 (+ (aget working (- i window-half))
                                         (aget working (+ i window-half))))
                  current-val (aget working i)]
              ;; SNIP clipping: keep minimum of current and interpolated
              (aset temp i (min current-val interpolated))))
          ;; Copy temp back to working
          (System/arraycopy temp 0 working 0 n)))

      ;; Return baseline-corrected intensities (original - baseline)
      (tcc/- intensities working))))

(defn tic-normalize
  "Normalize intensities using Total Ion Current (TIC) - MALDIquant compatible.
   
   MALDIquant normalizes the AREA under the curve (using trapezoid rule) to target-area,
   NOT the sum of intensities. This is important for mass spectrometry data where
   mass values may not be uniformly spaced.
   
   Algorithm:
   1. Calculate area under curve using trapezoid rule: A = Σ[0.5 * (m[i+1] - m[i]) * (int[i] + int[i+1])]
   2. Scale all intensities by (target-area / A)
   
   Args:
   - masses: mass values (required for area calculation)
   - intensities: intensity values
   - target-area: desired area under curve (default 1.0, matching MALDIquant)
   
   Returns: normalized intensities tensor
   
   Note: If area is zero, returns zeros (graceful handling)"
  [masses intensities {:keys [target-area]
                       :or {target-area 1.0}}]

  ;; Calculate area under curve using trapezoid rule
  (let [area (reduce +
                     (map (fn [m1 m2 i1 i2]
                            (* 0.5 (- m2 m1) (+ i1 i2)))
                          masses
                          (rest masses)
                          intensities
                          (rest intensities)))]

    (if (zero? area)
      (do
        (log/warn "Cannot normalize zero signal - area under curve is zero. Returning zeros.")
        (dtype/make-reader :float32
                           (dtype/ecount intensities)
                           0.0))

      ;; Scale intensities by (target-area / area)
      (do (when (< area 1e-10)
            (log/warn (format "Very small area detected: %e. Normalization may be unstable." area)))
          (tcc// (tcc/* intensities target-area)
                 area)))))

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
  "Apply full preprocessing pipeline to spectrum data.
   
   Args:
   - spectrum: map with :mass and :intensity keys
   - options: map with preprocessing options
     - :sqrt-transform (boolean)
     - :smooth-window (integer) 
     - :smooth-polynomial (integer)
     - :baseline-iterations (integer)
     - :tic-normalize (boolean)
     - :tic-target (number)
   
   Returns: map with :mass and :intensity keys (masses unchanged, intensities processed)"
  [spectrum {:as options
             :keys [should-sqrt-transform smooth-window smooth-polynomial
                    baseline-iterations should-tic-normalize tic-target]
             :or {should-sqrt-transform true
                  smooth-window 11
                  smooth-polynomial 2
                  baseline-iterations 25
                  should-tic-normalize true
                  tic-target 1.0}}]

  (let [masses (:mass spectrum)
        intensities (:intensity spectrum)

        ;; Apply preprocessing pipeline to intensities
        processed-intensities
        (cond-> intensities
          should-sqrt-transform (sqrt-transform)
          true (savitzky-golay-smooth {:window-size smooth-window
                                       :polynomial-order smooth-polynomial})
          true (snip-baseline-removal {:iterations baseline-iterations})
          should-tic-normalize (tic-normalize masses {:target-area tic-target}))]

    ;; Return spectrum with processed intensities
    {:mass masses
     :intensity processed-intensities}))
