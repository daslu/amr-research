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
            [maldi.data.bacteria :as bacteria]
            [scicloj.ml.xgboost]
            [maldi.cache :as cache])
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
       :antibiotic :Ciprofloxacin}
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
                                             :antibiotic :Ciprofloxacin})
      ((cache/cached-fn #'prepare-ml-data) {:preprocessing-params {}
                                            :binning-params {:range [2000 20000]
                                                             :step 3}})
      deref
      time))

(defn split [ml-data {:keys [seed]}]
  (-> ml-data
      cache/maybe-deref
      (tc/split->seq :holdout {:seed seed})
      first))

(comment
  (-> ((cache/cached-fn #'prepare-raw-data) {:site :A
                                             :year 2018
                                             :species bacteria/E-coli
                                             :antibiotic :Ciprofloxacin})
      ((cache/cached-fn #'prepare-ml-data) {:preprocessing-params {}
                                            :binning-params {:range [2000 20000]
                                                             :step 3}})
      ((cache/cached-fn #'split) {:seed 1})
      deref
      time))

(defn train [split-data hyper]
  (-> split-data
      cache/maybe-deref
      :train
      (ml/train hyper)))

(comment
  (-> ((cache/cached-fn #'prepare-raw-data) {:site :A
                                             :year 2018
                                             :species bacteria/E-coli
                                             :antibiotic :Ciprofloxacin})
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
  (ml/predict
   (-> split-data
       cache/maybe-deref
       :test)
   (-> model
       cache/maybe-deref)))


(defn measure
  [split-data predictions]
  (-> predictions
      cache/maybe-deref
      (tc/add-column :ri (-> split-data
                             cache/maybe-deref
                             :test
                             :ri))))


(defn eval-case [{:as case
                  :keys [site year antibiotic species]}]
  (let [ml-data (-> ((cache/cached-fn #'prepare-raw-data) case)
                    ((cache/cached-fn #'prepare-ml-data) {:preprocessing-params {}
                                                          :binning-params {:range [2000 20000]
                                                                           :step 3}}))]
    (when @ml-data
      (let [split-data (-> ml-data
                           ((cache/cached-fn #'split) {:seed 1}))
            model (cache/cached #'train split-data {:model-type :xgboost/classification
                                                    :round 50
                                                    :num-class 2})
            predictions @(cache/cached #'predict split-data model)
            m (measure split-data
                       predictions)]
        #_(-> m
              (plotly/layer-histogram {:=x 1
                                       :=color :ri
                                       :=mark-opacity 0.5}))
        (merge case
               {:n (tc/row-count m)
                :PRAUC (LabelEvaluationUtil/averagedPrecision
                        (boolean-array (m :ri))
                        (double-array (m 1)))
                :ROCAUC (LabelEvaluationUtil/binaryAUCROC
                         (boolean-array (m :ri))
                         (double-array (m 1)))})))))


(comment
  (-> (for [site [:A ;; :B :C :D
                  ]
            year [;; 2015 2016 2017
                  2018]
            antibiotic (ingestion/all-antibiotics)
            species [bacteria/E-coli] #_(bacteria/important-bacteria)]
        (let [acase {:site site
                     :year year
                     :antibiotic antibiotic
                     :species species}]
          (prn [:case acase])
          (eval-case acase)))
      (->> (remove nil?))
      tc/dataset
      time))

