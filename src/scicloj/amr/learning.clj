(ns scicloj.amr.learning
  (:require [scicloj.amr.data.ingestion :as ingestion]
            [scicloj.ripple.maldi :as ripple]
            [tablecloth.api :as tc]
            [tech.v3.parallel.for :as pfor]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as tensor]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.column-filters :as cf]
            [scicloj.amr.data.bacteria :as bacteria]
            [scicloj.ml.xgboost]
            [scicloj.pocket :as pocket]
            [clojure.tools.logging :as log]
            [tablecloth.column.api :as tcc])
  (:import (org.tribuo.classification.evaluation LabelEvaluationUtil)))

(defn prepare-raw-data
  [{:keys [site year species antibiotic]}]
  (let [cases (ingestion/available-cases)
        metadata (or (ingestion/load-metadata {:site site :year year})
                     (throw (ex-info "Metadata not found"
                                     {:site site :year year})))
        filtered-cases-1 (-> cases
                             (tc/select-rows #(and (= (:site %) site)
                                                   (= (:year %) year))))
        filtered-cases-2 (when (and (-> metadata :$error not)
                                    (-> filtered-cases-1 tc/row-count pos?))
                           (-> filtered-cases-1
                               (tc/left-join metadata [:code])
                               (tc/select-rows #(and (= (:species %) species)
                                                     (some? (get % antibiotic))))))]
    (when (some-> filtered-cases-2
                  tc/row-count
                  pos?)
      (-> filtered-cases-2
          (tc/map-columns :ri antibiotic (complement #{"S"}))
          (tc/set-dataset-name
           (format "raw-data [%s / %s / %s / %d]"
                   species (name antibiotic) (name site) year))))))

(comment
  (-> {:site :A
       :year 2018
       :species bacteria/E-coli
       :antibiotic :Cefepime}
      (prepare-raw-data-cached)
      deref
      time))

(defn prepare-ml-data
  "Prepare complete training dataset from cases"
  [raw-data {:keys [preprocessing-params binning-params]}]
  (try (-> raw-data
           pocket/maybe-deref
           (tc/add-column :features
                          (fn [ds]
                            (pfor/pmap
                             (fn [acase]
                               (-> acase
                                   :path
                                   ingestion/load-raw-spectrum
                                   (ripple/preprocess-spectrum-data preprocessing-params)
                                   (ripple/bin-spectrum binning-params)))
                             (tc/rows ds :as-maps))))
           (tc/select-rows :features)
           (as-> ds
                 (tc/add-columns
                  ds
                  (-> ds
                      :features
                      tensor/->tensor
                      (as-> t
                            (zipmap (->> t
                                         dtype/shape
                                         second
                                         range
                                         (map (comp keyword (partial str "x"))))
                                    (tensor/transpose t [1 0])))
                      tc/dataset)))
           (as-> ds
                 (tc/select-columns
                  ds
                  (->> ds
                       keys
                       (filter #(re-matches #"x[0-9]*" (name %)))
                       sort
                       (cons :ri))))
           (ds-mod/set-inference-target :ri)
           (tc/set-dataset-name "ml-data"))
       (catch Exception e
         (log/warn e "prepare-ml-data failed")
         nil)))

(comment
  (-> (prepare-raw-data-cached {:site :A
                                :year 2018
                                :species bacteria/E-coli
                                :antibiotic :Cefepime})
      (prepare-ml-data-cached {:preprocessing-params {:smooth-window 21 :smooth-polynomial 3}
                               :binning-params {:range [2000 20000]
                                                :step 3}})
      deref
      time))

(defn split [ml-data {:keys [seed]}]
  (let [ds (pocket/maybe-deref ml-data)]
    (when (-> ds tc/row-count (>= 100))
      (-> ds
          (tc/split->seq :holdout {:seed seed})
          first))))

(comment
  (-> (prepare-raw-data-cached {:site :A
                                :year 2018
                                :species bacteria/E-coli
                                :antibiotic :Cefepime})
      (prepare-ml-data-cached {:preprocessing-params {:smooth-window 21 :smooth-polynomial 3}
                               :binning-params {:range [2000 20000]
                                                :step 3}})
      (split-cached {:seed 1})
      deref
      time))

(defn train [split-data hyper]
  (some-> split-data
          pocket/maybe-deref
          :train
          (ml/train hyper)))

(comment
  (-> (prepare-raw-data-cached {:site :A
                                :year 2018
                                :species bacteria/E-coli
                                :antibiotic :Cefepime})
      (prepare-ml-data-cached {:preprocessing-params {:smooth-window 21 :smooth-polynomial 3}
                               :binning-params {:range [2000 20000]
                                                :step 3}})
      (split-cached {:seed 1})
      (train-cached {:model-type :xgboost/classification
                     :round 50
                     :num-class 2})
      deref
      time))

(defn predict
  [split-data model]
  (when-let [m (pocket/maybe-deref model)]
    (some-> split-data
            pocket/maybe-deref
            :test
            (ml/predict m)
            (tc/set-dataset-name "predictions"))))

(comment
  (let [split-data (-> (prepare-raw-data-cached {:site :A
                                                 :year 2018
                                                 :species bacteria/E-coli
                                                 :antibiotic :Cefepime})
                       (prepare-ml-data-cached {:preprocessing-params {:smooth-window 21 :smooth-polynomial 3}
                                                :binning-params {:range [2000 20000]
                                                                 :step 3}})
                       (split-cached {:seed 1}))
        model (-> split-data
                  (train-cached {:model-type :xgboost/classification
                                 :round 50
                                 :num-class 2}))]
    (->> model
         (predict-cached split-data)
         deref
         time)))

(defn measure
  [split-data predictions]
  (let [{:keys [train test]} (pocket/maybe-deref split-data)]
    (when-let [to-measure (some-> predictions
                                  pocket/maybe-deref
                                  (tc/add-column :ri (:ri test)))]
      {:n-train (tc/row-count train)
       :n-test (tc/row-count test)
       :pri (-> to-measure :ri tcc/mean)
       :PRAUC (LabelEvaluationUtil/averagedPrecision
               (boolean-array (to-measure :ri))
               (double-array (to-measure 1)))
       :ROCAUC (LabelEvaluationUtil/binaryAUCROC
                (boolean-array (to-measure :ri))
                (double-array (to-measure 1)))})))

;; ## Named caching functions
;;
;; Pre-built cached wrappers for each pipeline stage.
;; Use these instead of calling `(pocket/caching-fn #'...)` inline.

(def prepare-raw-data-cached (pocket/caching-fn #'prepare-raw-data))
(def prepare-ml-data-cached (pocket/caching-fn #'prepare-ml-data))
(def split-cached (pocket/caching-fn #'split))
(def train-cached (pocket/caching-fn #'train))
(def predict-cached (pocket/caching-fn #'predict))
(def measure-cached (pocket/caching-fn #'measure))

(comment
  (let [split-data (-> (prepare-raw-data-cached {:site :A
                                                 :year 2018
                                                 :species bacteria/E-coli
                                                 :antibiotic :Cefepime})
                       (prepare-ml-data-cached {:preprocessing-params {:smooth-window 21 :smooth-polynomial 3}
                                                :binning-params {:range [2000 20000]
                                                                 :step 3}})
                       (split-cached {:seed 1}))
        model (-> split-data
                  (train-cached {:model-type :xgboost/classification
                                 :round 50
                                 :num-class 2}))]
    (->> model
         (predict-cached split-data)
         (measure-cached split-data)
         deref
         time)))

;; ## Default parameters
;;
;; Preprocessing and binning parameters matching the Weis et al. paper.

(def default-ml-params
  {:preprocessing-params {:smooth-window 21 :smooth-polynomial 3}
   :binning-params {:range [2000 20000] :step 3}})

;; ## Higher-level helpers

(defn get-ml-data
  "Get cached, preprocessed ML data for one scenario.
  Returns nil if the data is unavailable.
  The :ri column is converted from boolean to integer 0/1
  for compatibility with evaluate-pipelines."
  ([params] (get-ml-data params default-ml-params))
  ([{:keys [site year species antibiotic]} ml-params]
   (try
     (some-> {:site site :year year
              :species species :antibiotic antibiotic}
             prepare-raw-data-cached
             (prepare-ml-data-cached ml-params)
             deref
             (tc/map-columns :ri [:ri] #(if % 1 0))
             (ds-mod/set-inference-target :ri))
     (catch Exception _e nil))))

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
