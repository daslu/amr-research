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


(defn find-peaks
  "Find peaks in spectrum based on local maxima with improved boundary handling.
   
   Args:
   - intensities
   - min-height: minimum peak height (default: 0.0)
   - min-distance: minimum distance between peaks in data points (default: 1)
   
   Returns: a vector of index values for the peaks"
  [intensities
   {:keys [min-height min-distance]
    :or {min-height 0
         min-distance 1}}]

  (let [n (dtype/ecount intensities)]

    (when (< n 1)
      (throw (ex-info "Cannot find peaks in empty data" {})))

    (if (= n 1)
      ;; Single point - treat as peak if meets criteria
      (if (>= (first intensities) min-height)
        [0]
        [])
      
      ;; Multiple points - check for local maxima including boundaries
      (reverse
       (loop [i 0
              peaks '()
              last-peak-idx -1000] ; Initialize with impossible index

         (when (< i n)
           (let [curr (intensities i)
                 relevant (and (>= curr min-height)
                               (>= (- i last-peak-idx) min-distance)
                               ;; Is it a local max?
                               (cond
                                 ;; First point: compare only with next
                                 (= i 0)
                                 (and (< i (dec n))
                                      (> curr (intensities i)))
                                 
                                 ;; Last point: compare only with previous  
                                 (= i (dec n))
                                 (> curr (intensities (dec i)))

                                 ;; Middle points: compare with both neighbors
                                 :else
                                 (and (> curr (intensities (dec i)))
                                      (> curr (intensities (inc i))))))]

             (if relevant
               (recur (inc i)
                      (cons i peaks)
                      i)
               (recur (inc i)
                      peaks
                      last-peak-idx)))))))))

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

