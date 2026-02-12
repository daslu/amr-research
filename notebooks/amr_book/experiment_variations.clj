;; # Experiment Variations
;;
;; This notebook runs many variations of
;; [AMR](https://en.wikipedia.org/wiki/Antimicrobial_resistance)
;; prediction experiments, combining
;; [Pocket](https://github.com/scicloj/pocket) for caching
;; and [metamorph.ml](https://github.com/scicloj/metamorph.ml)
;; for pipeline composition and evaluation.
;;
;; We grow the scope incrementally:
;;
;; 1. **Within-site holdout** — one scenario, multiple XGBoost configs
;; 2. **Cross-year** — train on one year, test on another
;; 3. **Cross-site** — train on one hospital, test on another
;; 4. **Sweep** — multiple antibiotics across experiment types

(ns amr-book.experiment-variations
  (:require
   ;; AMR ML pipeline (prepare, train, predict, measure):
   [scicloj.amr.learning :as learning]
   ;; Experiment runners (within-site, cross-year, cross-site):
   [scicloj.amr.scenarios :as scenarios]
   ;; Diagnostic plots (ROC, PR, calibration):
   [scicloj.amr.visualization :as viz]
   ;; Bacterial species definitions:
   [scicloj.amr.data.bacteria :as bacteria]
   ;; Pocket filesystem caching (https://github.com/scicloj/pocket):
   [scicloj.pocket :as pocket]
   ;; Metamorph pipeline composition (https://github.com/scicloj/metamorph.ml):
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   ;; Table processing (https://scicloj.github.io/tablecloth/):
   [tablecloth.api :as tc]
   ;; Interactive plotting (https://scicloj.github.io/tableplot/):
   [scicloj.tableplot.v1.plotly :as plotly]
   ;; Annotating kinds of visualizations:
   [scicloj.kindly.v4.kind :as kind]
   ;; EDN reader:
   [clojure.edn :as edn]))

;; ## Shared parameters
;;
;; Preprocessing and binning match the
;; [Weis et al.](https://doi.org/10.1038/s41591-021-01619-9)
;; paper throughout.

;; ## Cache configuration
;;
;; `pocket.edn` configures the cache directory (outside Dropbox)
;; and the in-memory LRU threshold (low, since each ML dataset
;; is ~48 MB).

(let [cfg (-> "pocket.edn" slurp edn/read-string)]
  (pocket/set-base-cache-dir! (:base-cache-dir cfg))
  (pocket/set-mem-cache-options! (:mem-cache cfg)))

;; ---

;; ## Part 1 — Within-site holdout
;;
;; Start with the simplest case: one scenario, random holdout
;; split, and compare XGBoost configurations using
;; `evaluate-pipelines`.

;; ### Preparing the data
;;
;; Pocket caches the expensive preprocessing. The first call
;; takes minutes; subsequent calls return instantly.

(def example-ml-data
  (learning/get-ml-data {:site :A :year 2018
                         :species bacteria/E-coli
                         :antibiotic :Cefepime}))

(when example-ml-data
  {:rows (tc/row-count example-ml-data)
   :features (dec (count (tc/column-names example-ml-data)))})

;; ### Comparing XGBoost configurations
;;
;; Three [XGBoost](https://xgboost.readthedocs.io/)
;; configurations varying the number of boosting rounds:

(def xgboost-round-values [10 50 100])

(def holdout-splits
  (when example-ml-data
    (tc/split->seq example-ml-data :holdout {:seed 1})))

(def eval-results
  (when holdout-splits
    (ml/evaluate-pipelines
     (mapv #(learning/make-pipeline {:rounds %}) xgboost-round-values)
     holdout-splits
     loss/classification-accuracy
     :accuracy
     {:return-best-pipeline-only false
      :return-best-crossvalidation-only false})))

;; ### Results by boosting rounds

(defn rocauc-from-eval-result
  "Extract ROCAUC from an evaluate-pipelines result entry."
  [eval-entry test-actuals]
  (let [prob-dist (get-in eval-entry [:test-transform :probability-distribution])]
    (when prob-dist
      (learning/compute-rocauc test-actuals (get prob-dist 1)))))

(def within-site-metrics
  (when eval-results
    (let [test-actuals (:ri (:test (first holdout-splits)))]
      (->> eval-results
           (map-indexed
            (fn [i pipe-results]
              (let [r (first pipe-results)]
                {:rounds (nth xgboost-round-values i)
                 :accuracy (get-in r [:test-transform :metric])
                 :ROCAUC (rocauc-from-eval-result r test-actuals)})))
           tc/dataset))))

within-site-metrics

(kind/test-last
 #(and (= 3 (tc/row-count %))
       (every? pos? (:accuracy %))))

;; ### Diagnostic curves (within-site, 50 rounds)
;;
;; The curves below visualize the 50-round classifier
;; in more detail than a single ROCAUC number.

(def within-site-plot-data
  (when eval-results
    (let [test-actuals (:ri (:test (first holdout-splits)))
          ;; 50-round pipeline is index 1
          r (first (nth eval-results 1))
          prob-dist (get-in r [:test-transform :probability-distribution])]
      (when prob-dist
        {:actuals (vec test-actuals)
         :probabilities (vec (get prob-dist 1))}))))

(when within-site-plot-data
  (viz/diagnostic-plots (:actuals within-site-plot-data)
                        (:probabilities within-site-plot-data)))

;; Release large Phase 1 intermediates (each ~48 MB)
;; so they can be garbage-collected before the sweeps.
(def example-ml-data nil)
(def holdout-splits nil)
(def eval-results nil)

;; ---

;; ## Part 2 — Cross-year experiment
;;
;; Does a model trained on 2017 data generalize to 2018?
;; Same species, same antibiotic, same hospital — different year.
;;
;; We compose two independently cached datasets as train
;; and test, instead of splitting one dataset randomly.

(def cross-year-result
  (let [train-data (learning/get-ml-data {:site :A :year 2017
                                          :species bacteria/E-coli
                                          :antibiotic :Cefepime})
        test-data (learning/get-ml-data {:site :A :year 2018
                                         :species bacteria/E-coli
                                         :antibiotic :Cefepime})]
    (when (and train-data test-data)
      (let [pipe (learning/make-pipeline {:rounds 50})
            fitted (mm/fit-pipe train-data pipe)
            predicted (mm/transform-pipe test-data pipe fitted)
            pred-ds (:metamorph/data predicted)
            test-actuals (:ri test-data)
            prob-col (get pred-ds 1)]
        {:n-train (tc/row-count train-data)
         :n-test (tc/row-count test-data)
         :ROCAUC (learning/compute-rocauc test-actuals prob-col)
         :actuals (vec test-actuals)
         :probabilities (vec prob-col)}))))

(when cross-year-result
  (select-keys cross-year-result [:n-train :n-test :ROCAUC]))

;; ### Diagnostic curves (cross-year)

(when cross-year-result
  (viz/diagnostic-plots (:actuals cross-year-result)
                        (:probabilities cross-year-result)))

;; ### Within-year vs cross-year

(def year-comparison
  (let [within (some-> within-site-metrics
                       (tc/select-rows #(= 50 (:rounds %)))
                       (tc/rows :as-maps)
                       first)]
    (tc/dataset
     (cond-> []
       within
       (conj {:experiment "within-year holdout (A 2018)"
              :ROCAUC (:ROCAUC within)})
       (:ROCAUC cross-year-result)
       (conj {:experiment "cross-year (A 2017→2018)"
              :ROCAUC (:ROCAUC cross-year-result)})))))

year-comparison

;; ---

;; ## Part 3 — Cross-site experiment
;;
;; Does a model trained at hospital A generalize to hospital C?
;; Same species, same antibiotic, same year — different hospital.
;;
;; Not every site has data for every combination.
;; The runner reports `:status` so we can trace why.

(def cross-site-result
  (let [train-data (learning/get-ml-data {:site :A :year 2018
                                          :species bacteria/E-coli
                                          :antibiotic :Cefepime})
        test-data (learning/get-ml-data {:site :C :year 2018
                                         :species bacteria/E-coli
                                         :antibiotic :Cefepime})]
    (when (and train-data test-data)
      (let [pipe (learning/make-pipeline {:rounds 50})
            fitted (mm/fit-pipe train-data pipe)
            predicted (mm/transform-pipe test-data pipe fitted)
            pred-ds (:metamorph/data predicted)
            test-actuals (:ri test-data)
            prob-col (get pred-ds 1)]
        {:n-train (tc/row-count train-data)
         :n-test (tc/row-count test-data)
         :ROCAUC (learning/compute-rocauc test-actuals prob-col)
         :actuals (vec test-actuals)
         :probabilities (vec prob-col)}))))

(when cross-site-result
  (select-keys cross-site-result [:n-train :n-test :ROCAUC]))

;; ### Diagnostic curves (cross-site)

(when cross-site-result
  (viz/diagnostic-plots (:actuals cross-site-result)
                        (:probabilities cross-site-result)))

;; ### All three experiment types compared

(def split-comparison
  (let [within (some-> within-site-metrics
                       (tc/select-rows #(= 50 (:rounds %)))
                       (tc/rows :as-maps)
                       first)]
    (tc/dataset
     (cond-> []
       within
       (conj {:experiment "within-site (A 2018)"
              :ROCAUC (:ROCAUC within)})
       (:ROCAUC cross-year-result)
       (conj {:experiment "cross-year (A 2017→2018)"
              :ROCAUC (:ROCAUC cross-year-result)})
       (:ROCAUC cross-site-result)
       (conj {:experiment "cross-site (A→C 2018)"
              :ROCAUC (:ROCAUC cross-site-result)})))))

split-comparison

;; The pattern is clear: within-site holdout performs best,
;; cross-year is close behind, and cross-site shows the
;; largest drop — the model relies partly on hospital-specific
;; spectral patterns.

;; ---

;; ## Part 4 — Sweeping across antibiotics
;;
;; Now we scale up to multiple antibiotics for E. coli,
;; comparing all three experiment types side by side.
;; The expensive spectra preprocessing is cached by Pocket,
;; so rerunning this section is fast.
;;
;; Every result carries a `:status` so we can distinguish
;; successful experiments from insufficient data, single-class
;; test sets, or unexpected errors.

;; ### E. coli — within-site (site A, 2018)

(def ecoli-within
  (->> (get bacteria/species->antibiotics bacteria/E-coli)
       (mapv (fn [ab]
               (scenarios/run-within-site {:species bacteria/E-coli
                                           :antibiotic ab
                                           :site :A :year 2018})))
       tc/dataset))

(-> ecoli-within
    (tc/select-columns [:antibiotic :status :n-train :n-test :ROCAUC]))

;; ### E. coli — cross-year (A: 2017→2018)

(def ecoli-cross-year
  (->> (get bacteria/species->antibiotics bacteria/E-coli)
       (mapv (fn [ab]
               (scenarios/run-cross-year {:species bacteria/E-coli
                                          :antibiotic ab
                                          :site :A
                                          :train-year 2017
                                          :test-year 2018})))
       tc/dataset))

(-> ecoli-cross-year
    (tc/select-columns [:antibiotic :status :n-train :n-test :ROCAUC]))

;; ### E. coli — cross-site (A→C, 2018)

(def ecoli-cross-site
  (->> (get bacteria/species->antibiotics bacteria/E-coli)
       (mapv (fn [ab]
               (scenarios/run-cross-site {:species bacteria/E-coli
                                          :antibiotic ab
                                          :train-site :A
                                          :test-site :C
                                          :year 2018})))
       tc/dataset))

(-> ecoli-cross-site
    (tc/select-columns [:antibiotic :status :n-train :n-test :ROCAUC]))

;; ### Diagnosis: why results are missing
;;
;; The `:status` column reveals the reason for every outcome.
;; Let's summarize:

(def status-summary
  (let [all (tc/concat ecoli-within ecoli-cross-year ecoli-cross-site)]
    (-> all
        (tc/group-by [:experiment :status])
        (tc/aggregate {:count tc/row-count})
        (tc/order-by [:experiment :status]))))

status-summary

;; ### Combined ROCAUC comparison
;;
;; Only experiments with `:status :ok` have meaningful ROCAUC:

(def ecoli-combined
  (let [tag (fn [ds label]
              (-> ds
                  (tc/select-rows #(= :ok (:status %)))
                  (tc/select-columns [:antibiotic :ROCAUC])
                  (tc/add-column :experiment label)))]
    (tc/concat (tag ecoli-within "within-site")
               (tag ecoli-cross-year "cross-year")
               (tag ecoli-cross-site "cross-site"))))

ecoli-combined

;; ### ROCAUC comparison plot

(when (pos? (tc/row-count ecoli-combined))
  (-> ecoli-combined
      (plotly/base {:=x :antibiotic
                    :=y :ROCAUC
                    :=color :experiment
                    :=title "E. coli — ROCAUC by experiment type"})
      plotly/layer-point
      plotly/plot
      (assoc-in [:layout :xaxis :tickangle] -45)))
