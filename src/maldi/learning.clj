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

(defn prepare-ml-data
  "Prepare complete training dataset from cases"
  [{:keys [site year species antibiotic
           preprocessing-params binning-params]}]
  (let [cases (ingestion/available-cases {})
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
      (try (-> filtered-cases-2
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
               (tc/map-columns :ri antibiotic (complement #{"S"}))
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
           (catch Exception e nil)))))

(comment
  (-> {:site :A
       :year 2018
       :species bacteria/E-coli
       :antibiotic :Ciprofloxacin
       :preprocessing-params {}
       :binning-params {:range [2000 20000]
                        :step 3}}
      ((cache/cached-fn #'prepare-ml-data))
      deref
      time))

(defn split [ml-data {:keys [seed]}]
  (-> ml-data
      cache/maybe-deref
      (tc/split->seq :holdout {:seed seed})
      first))

(comment
  (-> {:site :A
       :year 2018
       :species bacteria/E-coli
       :antibiotic :Ciprofloxacin
       :preprocessing-params {}
       :binning-params {:range [2000 20000]
                        :step 3}}
      ((cache/cached-fn #'prepare-ml-data))
      ((cache/cached-fn #'split) {:seed 1})
      deref
      time))

(defn train [split-data hyper]
  (-> split-data
      cache/maybe-deref
      :train
      (ml/train hyper)))

(comment
  (-> {:site :A
       :year 2018
       :species bacteria/E-coli
       :antibiotic :Ciprofloxacin
       :preprocessing-params {}
       :binning-params {:range [2000 20000]
                        :step 3}}
      ((cache/cached-fn #'prepare-ml-data))
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


(defn eval-case [{:keys [site year antibiotic]}]
  (let [ml-data (-> {:site site
                     :year year
                     :species bacteria/E-coli
                     :antibiotic antibiotic
                     :preprocessing-params {}
                     :binning-params {:range [2000 20000]
                                      :step 3}}
                    ((cache/cached-fn #'prepare-ml-data)))]
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
        {:site site
         :year year
         :n (tc/row-count m)
         :PRAUC (LabelEvaluationUtil/averagedPrecision
                 (boolean-array (m :ri))
                 (double-array (m 1)))
         :ROCAUC (LabelEvaluationUtil/binaryAUCROC
                  (boolean-array (m :ri))
                  (double-array (m 1)))}))))



(-> (for [site [:A :B :C :D]
          year [2015 2016 2017 2018]
          antibiotic (ingestion/all-antibiotics {})]
      (let [acase {:site site
                   :year year
                   :antibiotic antibiotic}]
        (prn [:case acase])
        (eval-case acase)))
    (->> (remove nil?))
    tc/dataset
    time)

