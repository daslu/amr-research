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
            [maldi.cache :as cache]))


(defn prepare-ml-data
  "Prepare complete training dataset from cases"
  [{:keys [site year species antibiotic
           preprocessing-params binning-params]}]
  (let [cases (ingestion/available-cases {})
        metadata (ingestion/load-metadata {:site site
                                           :year year})
        filtered-cases (-> cases
                           (tc/select-rows #(and (= (:site %) site)
                                                 (= (:year %) year)))
                           (tc/left-join metadata [:code])
                           (tc/select-rows #(and (= (:species %) species)
                                                 (contains? % antibiotic))))]
    (-> filtered-cases
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
        (ds-mod/set-inference-target :ri))))

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














