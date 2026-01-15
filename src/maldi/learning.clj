(ns maldi.learning
  (:require [maldi.data.ingestion :as ingestion]
            [maldi.data.signal :as signal]
            [maldi.data.binning :as binning]
            [tablecloth.api :as tc]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tech.v3.parallel.for :as pfor]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as tensor]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.print :as ds-print]
            [maldi.data.bacteria :as bacteria]
            [scicloj.ml.xgboost]
            [maldi.cache :as cache]
            [clojure.tools.logging :as log]
            [tablecloth.column.api :as tcc]
            [scicloj.kindly.v4.kind :as kind])
  (:import (org.tribuo.classification.evaluation LabelEvaluationUtil)))

(defn prepare-raw-data
  [{:keys [site year species antibiotic]}]
  (let [cases (ingestion/available-cases)
        metadata (ingestion/load-metadata {:site site
                                           :year year})
        filtered-cases-1 (-> cases
                             (tc/select-rows #(and (= (:site %) site)
                                                   (= (:year %) year))))
        filtered-cases-2 (when (and (-> metadata :$error not)
                                    (-> filtered-cases-1 tc/row-count pos?))
                           (-> filtered-cases-1
                               (tc/left-join metadata [:code])
                               (tc/select-rows #(and (= (:species %) species)
                                                     (contains? % antibiotic)))))]
    (when (some-> filtered-cases-2
                  tc/row-count
                  pos?)
      (-> filtered-cases-2
          (tc/map-columns :ri antibiotic (complement #{"S"}))))))

(comment
  (-> {:site :A
       :year 2018
       :species bacteria/E-coli
       :antibiotic :Cefepime}
      ((cache/cached-fn #'prepare-raw-data))
      deref
      time))


(defn prepare-ml-data
  "Prepare complete training dataset from cases"
  [raw-data {:keys [preprocessing-params binning-params]}]
  (try (-> raw-data
           cache/maybe-deref
           (tc/add-column :features
                          (fn [ds]
                            (pfor/pmap
                             (fn [acase]
                               (-> acase
                                   :path
                                   ingestion/load-raw-spectrum
                                   (update :intensity #(signal/preprocess-spectrum-data
                                                        %
                                                        preprocessing-params))
                                   (binning/bin-spectrum binning-params)))
                             (tc/rows ds :as-maps))))
           (tc/select-rows :features)
           (as->  ds
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
           (as->  ds
               (tc/select-columns
                ds
                (->> ds
                     keys
                     (filter #(re-matches #"x[0-9]*" (name %)))
                     sort
                     (cons :ri))))
           (ds-mod/set-inference-target :ri))
       (catch Exception e nil)))

(comment
  (-> ((cache/cached-fn #'prepare-raw-data) {:site :A
                                             :year 2018
                                             :species bacteria/E-coli
                                             :antibiotic :Cefepime})
      ((cache/cached-fn #'prepare-ml-data) {:preprocessing-params {}
                                            :binning-params {:range [2000 20000]
                                                             :step 3}})
      deref
      time))

(defn split [ml-data {:keys [seed]}]
  (let [ds (cache/maybe-deref ml-data)]
    (when (-> ds tc/row-count (>= 100))
      (-> ds
          (tc/split->seq :holdout {:seed seed})
          first))))

(comment
  (-> ((cache/cached-fn #'prepare-raw-data) {:site :A
                                             :year 2018
                                             :species bacteria/E-coli
                                             :antibiotic :Cefepime})
      ((cache/cached-fn #'prepare-ml-data) {:preprocessing-params {}
                                            :binning-params {:range [2000 20000]
                                                             :step 3}})
      ((cache/cached-fn #'split) {:seed 1})
      deref
      time))

(defn train [split-data hyper]
  (some-> split-data
          cache/maybe-deref
          :train
          (ml/train hyper)))

(comment
  (-> ((cache/cached-fn #'prepare-raw-data) {:site :A
                                             :year 2018
                                             :species bacteria/E-coli
                                             :antibiotic :Cefepime})
      ((cache/cached-fn #'prepare-ml-data) {:preprocessing-params {}
                                            :binning-params {:range [2000 20000]
                                                             :step 3}})
      ((cache/cached-fn #'split) {:seed 1})
      ((cache/cached-fn #'train) {:model-type :xgboost/classification
                                  :round 10
                                  :num-class 2})
      deref
      time))


(defn predict
  [split-data model]
  (some-> split-data
          cache/maybe-deref
          :test
          (ml/predict (cache/maybe-deref model))))


(defn measure
  [split-data predictions]
  (let [{:keys [train test]} (cache/maybe-deref split-data)]
    (when-let [to-measure (some-> predictions
                                  cache/maybe-deref
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

