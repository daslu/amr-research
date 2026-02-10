;; # Caching and Scenarios
;;
;; The [previous notebook](single_prediction.html) ran the pipeline
;; with direct function calls. This works, but preprocessing
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
   [scicloj.amr.learning :as learning]
   [scicloj.amr.data.bacteria :as bacteria]
   [scicloj.pocket :as pocket]
   [tablecloth.api :as tc]
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
  (let [raw-data   (-> {:site site :year year
                         :species species :antibiotic antibiotic}
                        ((pocket/caching-fn #'learning/prepare-raw-data)))
        ml-data    (-> raw-data
                        ((pocket/caching-fn #'learning/prepare-ml-data)
                         {:preprocessing-params {}
                          :binning-params {:range [2000 20000] :step 3}}))
        split-data (-> ml-data
                        ((pocket/caching-fn #'learning/split) {:seed 1}))
        model      (-> split-data
                        ((pocket/caching-fn #'learning/train)
                         {:model-type :xgboost/classification
                          :round 50
                          :num-class 2}))
        predictions (-> split-data
                         ((pocket/caching-fn #'learning/predict) model))
        metrics    (-> split-data
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
;; - Add more species (S. aureus, K. pneumoniae) and their antibiotics
;; - Sweep over sites (A–D) and years (2015–2018)
;; - Compare against the results reported by Weis et al.
;;
;; The [full walkthrough](ml_pipeline_walkthrough.html) covers the
;; pipeline in more detail, showing both cached and uncached variants
;; side by side.
