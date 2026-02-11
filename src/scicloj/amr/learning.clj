(ns scicloj.amr.learning
  (:require [scicloj.amr.data.ingestion :as ingestion]
            [scicloj.ripple.maldi :as ripple]
            [tablecloth.api :as tc]
            [tech.v3.parallel.for :as pfor]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as tensor]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset.modelling :as ds-mod]
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
