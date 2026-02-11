;; # Caching and Scenarios
;;
;; Running the pipeline with direct function calls works, but preprocessing
;; thousands of spectra takes minutes — and we don't want to
;; redo that work every time we change a downstream parameter.
;;
;; [Pocket](https://github.com/scicloj/pocket) solves this by
;; caching each stage's output on disk. When the same function
;; is called with the same arguments, the cached result is
;; returned instantly.
;;
;; This notebook shows the caching pattern and then uses it
;; to evaluate multiple species/antibiotic combinations.

(ns amr-book.caching-and-scenarios
  (:require
   ;; AMR ML pipeline (prepare, train, predict, measure):
   [scicloj.amr.learning :as learning]
   ;; Bacterial species definitions and antibiotic lists:
   [scicloj.amr.data.bacteria :as bacteria]
   ;; Filesystem-based caching (https://github.com/scicloj/pocket):
   [scicloj.pocket :as pocket]
   ;; Table processing (https://scicloj.github.io/tablecloth/):
   [tablecloth.api :as tc]
   ;; Annotating kinds of visualizations (https://scicloj.github.io/kindly-noted/):
   [scicloj.kindly.v4.kind :as kind]))

;; ## The caching pattern
;;
;; `pocket/caching-fn` wraps a var so that its return value
;; is cached on disk (keyed by function identity + arguments).
;; The wrapped function returns a `Cached` object — an `IDeref`
;; that triggers computation (or loads from cache) on `deref`.
;;
;; **Important:** always pass a **var** (`#'fn-name`), not the
;; function value. Vars have stable identity; function objects don't.

;; Here is the full cached pipeline for one scenario:

(defn run-scenario
  "Run the AMR prediction pipeline for one species/antibiotic/site/year.
  Returns the evaluation metrics, or nil if insufficient data."
  [{:keys [site year species antibiotic]}]
  (let [raw-data (-> {:site site :year year
                      :species species :antibiotic antibiotic}
                     ((pocket/caching-fn #'learning/prepare-raw-data)))
        ml-data (-> raw-data
                    ((pocket/caching-fn #'learning/prepare-ml-data)
                     {:preprocessing-params {:smooth-window 21 :smooth-polynomial 3}
                      :binning-params {:range [2000 20000] :step 3}}))
        split-data (-> ml-data
                       ((pocket/caching-fn #'learning/split) {:seed 1}))
        model (-> split-data
                  ((pocket/caching-fn #'learning/train)
                   {:model-type :xgboost/classification
                    :round 50
                    :num-class 2}))
        predictions (-> split-data
                        ((pocket/caching-fn #'learning/predict) model))
        metrics (-> split-data
                    ((pocket/caching-fn #'learning/measure) predictions)
                    deref)]
    metrics))

;; ### Example: single scenario

(def example-metrics
  (run-scenario {:site :A
                 :year 2018
                 :species bacteria/E-coli
                 :antibiotic :Cefepime}))

example-metrics

;; Run it again — this time the cached result is returned instantly:

(def example-metrics-2
  (run-scenario {:site :A
                 :year 2018
                 :species bacteria/E-coli
                 :antibiotic :Cefepime}))

example-metrics-2

;; ## Running multiple scenarios
;;
;; Now we can sweep over antibiotics cheaply, since the expensive
;; preprocessing is cached and shared across scenarios that use
;; the same spectra.

(def e-coli-antibiotics
  [:Cefepime :Ciprofloxacin :Ceftriaxone])

(def scenario-results
  (->> e-coli-antibiotics
       (map (fn [ab]
              (some-> (run-scenario {:site :A
                                     :year 2018
                                     :species bacteria/E-coli
                                     :antibiotic ab})
                      (assoc :antibiotic ab))))
       (remove nil?)
       tc/dataset))

scenario-results

(kind/test-last
 #(= 3 (tc/row-count %)))

;; ### Comparing results
;;
;; A quick summary table showing how well the model discriminates
;; resistance for each antibiotic:

(-> scenario-results
    (tc/select-columns [:antibiotic :n-train :n-test :pri :ROCAUC :PRAUC])
    (tc/order-by [:ROCAUC] :desc))

;; The `:pri` column shows resistance prevalence — antibiotics
;; with very low or very high prevalence are harder to evaluate
;; meaningfully (the model has little signal to learn from).

;; ## What gets cached?
;;
;; Pocket stores results under `$POCKET_BASE_CACHE_DIR/.cache/`.
;; Each pipeline stage gets its own cache entry, keyed by
;; function + arguments. Because `prepare-ml-data` (the expensive
;; step) depends only on the raw data and preprocessing parameters,
;; it is computed once and reused across all downstream variations
;; (different splits, hyperparameters, etc.).

;; ## Next steps
;;
;; - Add more species ([*S. aureus*](https://en.wikipedia.org/wiki/Staphylococcus_aureus), [*K. pneumoniae*](https://en.wikipedia.org/wiki/Klebsiella_pneumoniae)) and their antibiotics
;; - Sweep over sites (A–D) and years (2015–2018)
;; - Compare against the results reported by [Weis et al.](https://doi.org/10.1038/s41591-021-01619-9)
;;
