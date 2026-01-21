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
  (first (r->clj (r "mad(c(0.1, 0.2, 0.5, 1.0, 0.5, 0.2, 0.1, 0.15, 0.3, 0.8, 1.5, 0.7, 0.2))"))))

;; ### Clojure Implementation
(def clj-mad 
  (signal/estimate-noise-mad test-intensities {}))

;; ### Comparison
(kind/table
 {:column-names ["Implementation" "MAD Value" "Difference"]
  :row-vectors [["R (MALDIquant)" r-mad "-"]
                ["Clojure" clj-mad (- clj-mad r-mad)]]})

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

;; ### R: Find Local Maxima
;; R doesn't expose the local maxima function directly, so we'll compare
;; via the full detectPeaks pipeline below.
(r "
spectrum_intensity <- c(rep(0.05, 10),
                        0.1, 0.2, 0.5, 1.2, 0.6, 0.2, 0.1,
                        rep(0.03, 5),
                        0.08, 0.15, 0.3, 0.8, 2.5, 1.0, 0.4, 0.1,
                        rep(0.04, 8))
")

;; ## 3. Peak Detection (Full Pipeline)
;;
;; The complete pipeline: local maxima + noise estimation + SNR filtering.

;; ### R Implementation (MALDIquant)
(r "
spectrum_mass <- seq(2000, 2000 + length(spectrum_intensity) * 2, by=2)[1:length(spectrum_intensity)]
spec <- createMassSpectrum(mass=spectrum_mass, intensity=spectrum_intensity)
peaks <- detectPeaks(spec, halfWindowSize=3, SNR=2, method='MAD')
")

(def r-peak-indices 
  (mapv dec (r->clj (r "which(spectrum_intensity %in% intensity(peaks))"))))

(def r-peak-intensities 
  (r->clj (r "intensity(peaks)")))

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
    (let [r-peaks (do
                    (r (str "peaks_snr" snr " <- detectPeaks(spec, halfWindowSize=3, SNR=" snr ", method='MAD')"))
                    (count (r->clj (r (str "intensity(peaks_snr" snr ")")))))
          clj-peaks (count (signal/detect-peaks spectrum-data {:half-window-size 3 :snr snr}))]
      {:snr snr
       :r-peaks r-peaks
       :clj-peaks clj-peaks
       :match? (= r-peaks clj-peaks)})))

(kind/table
 (-> (compare-snr-thresholds [1 2 3 4])
     tc/dataset))

^:kind/test-last
(defn test-ns []
  {:MAD-matches (= r-mad clj-mad)
   :peak-indices-match (= r-peak-indices clj-peak-indices)
   :peak-intensities-match (= r-peak-intensities clj-peak-intensities)})
