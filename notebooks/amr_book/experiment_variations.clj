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
   [tablecloth.column.api :as tcc]
   ;; Dataset modelling and column filters:
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.column-filters :as cf]
   ;; Interactive plotting (https://scicloj.github.io/tableplot/):
   [scicloj.tableplot.v1.plotly :as plotly]
   ;; Annotating kinds of visualizations:
   [scicloj.kindly.v4.kind :as kind]
   ;; EDN reader:
   [clojure.edn :as edn])
  (:import
   (org.tribuo.classification.evaluation LabelEvaluationUtil)))

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

(def ml-params
  {:preprocessing-params {:smooth-window 21 :smooth-polynomial 3}
   :binning-params {:range [2000 20000] :step 3}})

;; ## Species and antibiotics
;;
;; The 23 species–antibiotic combinations from the paper:

(def species->antibiotics
  {bacteria/E-coli
   [:Meropenem :Ertapenem :Ceftriaxone :Cefepime
    :Piperacillin-Tazobactam :Nitrofurantoin
    :Ciprofloxacin :Cotrimoxazole]

   bacteria/S-aureus
   [:Cotrimoxazole :Clindamycin :Vancomycin :Linezolid
    (keyword "Amoxicillin-Clavulanic acid")
    (keyword "Ampicillin-Amoxicillin")
    :Oxacillin]

   bacteria/P-aeruginosa
   [:Piperacillin-Tazobactam :Cefepime :Ceftazidime
    :Meropenem :Amikacin :Ciprofloxacin
    :Colistin :Tobramycin]})

;; ## Helpers
;;
;; `get-ml-data` loads and caches one (species, antibiotic,
;; site, year) scenario. The `:ri` column is converted from
;; boolean to integer 0/1 for compatibility with
;; `evaluate-pipelines`.

(defn get-ml-data
  "Get cached, preprocessed ML data for one scenario.
  Returns nil if the data is unavailable."
  [{:keys [site year species antibiotic]}]
  (try
    (some-> {:site site :year year
             :species species :antibiotic antibiotic}
            learning/prepare-raw-data-cached
            (learning/prepare-ml-data-cached ml-params)
            deref
            (tc/map-columns :ri [:ri] #(if % 1 0))
            (ds-mod/set-inference-target :ri))
    (catch Exception _e nil)))

;; ## The `pocket-model` step
;;
;; A metamorph.ml pipeline step that wraps `ml/train` with
;; `pocket/cached` — a drop-in replacement for `ml/model`
;; that persists trained models to disk.
;;
;; Adapted from the
;; [Pocket ML pipelines tutorial](https://scicloj.github.io/pocket/pocket_book.ml_pipelines.html).

(defn pocket-model
  "Like `ml/model`, but caches `ml/train` through Pocket.
  In :fit mode, wraps ml/train with pocket/cached.
  In :transform mode, calls ml/predict directly."
  [options]
  (fn [{:metamorph/keys [id data mode] :as ctx}]
    (case mode
      :fit
      (let [model (deref (pocket/cached #'ml/train data options))]
        (assoc ctx id (assoc model ::ml/unsupervised? false)))
      :transform
      (-> ctx
          (update id assoc
                  ::ml/feature-ds (cf/feature data)
                  ::ml/target-ds (cf/target data))
          (assoc :metamorph/data (ml/predict data (get ctx id)))))))

(defn make-pipeline
  "Build an XGBoost classification pipeline with pocket-cached training."
  [{:keys [rounds] :or {rounds 50}}]
  (mm/pipeline
   {:metamorph/id :model}
   (pocket-model {:model-type :xgboost/classification
                  :round rounds
                  :num-class 2})))

;; ## Metrics

(defn compute-rocauc
  "Compute ROCAUC from actual labels (int 0/1) and
  probability predictions. Returns nil when ROCAUC
  is undefined (e.g. single-class test data)."
  [actuals prob-col]
  (try
    (let [v (LabelEvaluationUtil/binaryAUCROC
             (boolean-array (mapv pos? actuals))
             (double-array prob-col))]
      (when-not (Double/isNaN v) v))
    (catch Exception _e nil)))

(defn evaluate-cross-split
  "Train a pipeline on `train-data`, predict on `test-data`,
  and return metrics. Returns nil on failure."
  [pipeline train-data test-data]
  (try
    (let [fitted-ctx (mm/fit-pipe train-data pipeline)
          predicted-ctx (mm/transform-pipe test-data pipeline fitted-ctx)
          pred-ds (:metamorph/data predicted-ctx)]
      {:n-train (tc/row-count train-data)
       :n-test (tc/row-count test-data)
       :ROCAUC (compute-rocauc (:ri test-data) (get pred-ds 1))})
    (catch Exception _e nil)))

;; ## Diagnostic plots
;;
;; Per-scenario
;; [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic),
;; [precision–recall](https://en.wikipedia.org/wiki/Precision_and_recall),
;; and [calibration](https://en.wikipedia.org/wiki/Calibration_(statistics))
;; curves.

(defn plot-roc-curve
  "Plot the ROC curve (TPR vs FPR)."
  [actuals probabilities]
  (let [curve (LabelEvaluationUtil/generateROCCurve
               (boolean-array (mapv pos? actuals))
               (double-array probabilities))]
    (-> {:fpr (vec (.fpr curve))
         :tpr (vec (.tpr curve))}
        tc/dataset
        (plotly/base {:=title "ROC curve"
                      :=x :fpr :=y :tpr})
        plotly/layer-line
        plotly/plot)))

(defn plot-pr-curve
  "Plot the precision–recall curve."
  [actuals probabilities]
  (let [curve (LabelEvaluationUtil/generatePRCurve
               (boolean-array (mapv pos? actuals))
               (double-array probabilities))]
    (-> {:recall (vec (.recall curve))
         :precision (vec (.precision curve))}
        tc/dataset
        (plotly/base {:=title "Precision–Recall curve"
                      :=x :recall :=y :precision})
        plotly/layer-line
        plotly/plot)))

(defn plot-calibration
  "Plot predicted probability vs observed resistance rate.
  Samples are sorted by predicted probability and grouped
  into bins of 30."
  [actuals probabilities]
  (-> {:probability probabilities :actual actuals}
      tc/dataset
      (tc/order-by :probability)
      (tc/add-column :i (range))
      (tc/map-columns :bin [:i] #(quot % 30))
      (tc/group-by [:bin])
      (tc/aggregate {:predicted-probability #(tcc/mean (:probability %))
                     :observed-rate #(tcc/mean (:actual %))})
      (plotly/base {:=title "Calibration curve"
                    :=x :predicted-probability
                    :=y :observed-rate})
      plotly/layer-line
      plotly/layer-point
      plotly/plot))

(defn diagnostic-plots
  "Show ROC, precision–recall, and calibration curves
  for a binary classifier."
  [actuals probabilities]
  (kind/fragment
   [(plot-roc-curve actuals probabilities)
    (plot-pr-curve actuals probabilities)
    (plot-calibration actuals probabilities)]))

;; ## Experiment runners
;;
;; Each function loads data, runs the experiment, and returns
;; a small metrics map. The large datasets are not retained
;; in top-level defs.
;;
;; Every result includes a `:status` key:
;; - `:ok` — experiment ran and produced a ROCAUC
;; - `:no-data` — one or both datasets missing
;; - `:too-few` — not enough samples
;; - `:single-class` — test set has only one class (ROCAUC undefined)
;; - `:error` — unexpected failure (message included)

(defn run-within-site
  "Within-site holdout experiment."
  [{:keys [species antibiotic site year rounds]
    :or {rounds 50}}]
  (let [base {:species species :antibiotic antibiotic
              :site site :year year
              :experiment :within-site}]
    (try
      (let [ml-data (get-ml-data {:site site :year year
                                  :species species
                                  :antibiotic antibiotic})]
        (cond
          (nil? ml-data)
          (assoc base :status :no-data)

          (< (tc/row-count ml-data) 100)
          (assoc base :status :too-few
                 :n-total (tc/row-count ml-data))

          :else
          (let [split (first (tc/split->seq ml-data :holdout {:seed 1}))
                pipe (make-pipeline {:rounds rounds})
                fitted (mm/fit-pipe (:train split) pipe)
                predicted (mm/transform-pipe (:test split) pipe fitted)
                pred-ds (:metamorph/data predicted)
                rocauc (compute-rocauc (:ri (:test split))
                                       (get pred-ds 1))]
            (merge base
                   {:n-train (tc/row-count (:train split))
                    :n-test (tc/row-count (:test split))
                    :ROCAUC rocauc
                    :status (if rocauc :ok :single-class)}))))
      (catch Exception e
        (assoc base :status :error
               :message (.getMessage e))))))

(defn run-cross-year
  "Train on train-year, test on test-year."
  [{:keys [species antibiotic site train-year test-year rounds]
    :or {rounds 50}}]
  (let [base {:species species :antibiotic antibiotic
              :site site
              :train-year train-year :test-year test-year
              :experiment :cross-year}]
    (try
      (let [train-data (get-ml-data {:site site :year train-year
                                     :species species
                                     :antibiotic antibiotic})
            test-data (get-ml-data {:site site :year test-year
                                    :species species
                                    :antibiotic antibiotic})]
        (cond
          (nil? train-data)
          (assoc base :status :no-data :detail :no-train)

          (nil? test-data)
          (assoc base :status :no-data :detail :no-test)

          (< (tc/row-count train-data) 50)
          (assoc base :status :too-few
                 :n-train (tc/row-count train-data))

          (< (tc/row-count test-data) 50)
          (assoc base :status :too-few
                 :n-test (tc/row-count test-data))

          :else
          (let [result (evaluate-cross-split (make-pipeline {:rounds rounds})
                                             train-data test-data)]
            (merge base result
                   {:status (if (:ROCAUC result) :ok :single-class)}))))
      (catch Exception e
        (assoc base :status :error
               :message (.getMessage e))))))

(defn run-cross-site
  "Train on train-site, test on test-site."
  [{:keys [species antibiotic train-site test-site year rounds]
    :or {rounds 50}}]
  (let [base {:species species :antibiotic antibiotic
              :train-site train-site :test-site test-site
              :year year
              :experiment :cross-site}]
    (try
      (let [train-data (get-ml-data {:site train-site :year year
                                     :species species
                                     :antibiotic antibiotic})
            test-data (get-ml-data {:site test-site :year year
                                    :species species
                                    :antibiotic antibiotic})]
        (cond
          (nil? train-data)
          (assoc base :status :no-data :detail :no-train)

          (nil? test-data)
          (assoc base :status :no-data :detail :no-test)

          (< (tc/row-count train-data) 50)
          (assoc base :status :too-few
                 :n-train (tc/row-count train-data))

          (< (tc/row-count test-data) 50)
          (assoc base :status :too-few
                 :n-test (tc/row-count test-data))

          :else
          (let [result (evaluate-cross-split (make-pipeline {:rounds rounds})
                                             train-data test-data)]
            (merge base result
                   {:status (if (:ROCAUC result) :ok :single-class)}))))
      (catch Exception e
        (assoc base :status :error
               :message (.getMessage e))))))

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
  (get-ml-data {:site :A :year 2018
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
     (mapv #(make-pipeline {:rounds %}) xgboost-round-values)
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
      (compute-rocauc test-actuals (get prob-dist 1)))))

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
  (diagnostic-plots (:actuals within-site-plot-data)
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
  (let [train-data (get-ml-data {:site :A :year 2017
                                 :species bacteria/E-coli
                                 :antibiotic :Cefepime})
        test-data (get-ml-data {:site :A :year 2018
                                :species bacteria/E-coli
                                :antibiotic :Cefepime})]
    (when (and train-data test-data)
      (let [pipe (make-pipeline {:rounds 50})
            fitted (mm/fit-pipe train-data pipe)
            predicted (mm/transform-pipe test-data pipe fitted)
            pred-ds (:metamorph/data predicted)
            test-actuals (:ri test-data)
            prob-col (get pred-ds 1)]
        {:n-train (tc/row-count train-data)
         :n-test (tc/row-count test-data)
         :ROCAUC (compute-rocauc test-actuals prob-col)
         :actuals (vec test-actuals)
         :probabilities (vec prob-col)}))))

(when cross-year-result
  (select-keys cross-year-result [:n-train :n-test :ROCAUC]))

;; ### Diagnostic curves (cross-year)

(when cross-year-result
  (diagnostic-plots (:actuals cross-year-result)
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
  (let [train-data (get-ml-data {:site :A :year 2018
                                 :species bacteria/E-coli
                                 :antibiotic :Cefepime})
        test-data (get-ml-data {:site :C :year 2018
                                :species bacteria/E-coli
                                :antibiotic :Cefepime})]
    (when (and train-data test-data)
      (let [pipe (make-pipeline {:rounds 50})
            fitted (mm/fit-pipe train-data pipe)
            predicted (mm/transform-pipe test-data pipe fitted)
            pred-ds (:metamorph/data predicted)
            test-actuals (:ri test-data)
            prob-col (get pred-ds 1)]
        {:n-train (tc/row-count train-data)
         :n-test (tc/row-count test-data)
         :ROCAUC (compute-rocauc test-actuals prob-col)
         :actuals (vec test-actuals)
         :probabilities (vec prob-col)}))))

(when cross-site-result
  (select-keys cross-site-result [:n-train :n-test :ROCAUC]))

;; ### Diagnostic curves (cross-site)

(when cross-site-result
  (diagnostic-plots (:actuals cross-site-result)
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
  (->> (get species->antibiotics bacteria/E-coli)
       (mapv (fn [ab]
               (run-within-site {:species bacteria/E-coli
                                 :antibiotic ab
                                 :site :A :year 2018})))
       tc/dataset))

(-> ecoli-within
    (tc/select-columns [:antibiotic :status :n-train :n-test :ROCAUC]))

;; ### E. coli — cross-year (A: 2017→2018)

(def ecoli-cross-year
  (->> (get species->antibiotics bacteria/E-coli)
       (mapv (fn [ab]
               (run-cross-year {:species bacteria/E-coli
                                :antibiotic ab
                                :site :A
                                :train-year 2017
                                :test-year 2018})))
       tc/dataset))

(-> ecoli-cross-year
    (tc/select-columns [:antibiotic :status :n-train :n-test :ROCAUC]))

;; ### E. coli — cross-site (A→C, 2018)

(def ecoli-cross-site
  (->> (get species->antibiotics bacteria/E-coli)
       (mapv (fn [ab]
               (run-cross-site {:species bacteria/E-coli
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

