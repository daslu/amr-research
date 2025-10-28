(ns maldi.learning
  (:require [maldi.data.ingestion :as ingestion]
            [maldi.data.signal :as signal]
            [maldi.data.binning :as binning]
            [tablecloth.api :as tc]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tech.v3.parallel.for :as pfor]
            [scicloj.metamorph.ml :as ml]))


(defn prepare-learning-data
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
        (tc/select-rows :features))))


(comment
  (prepare-learning-data
   {:site :A
    :year 2018
    :species "Escherichia coli"
    :antibiotic :Ciprofloxacin
    :preprocessing-params {}
    :binning-params {:range [2000 20000]
                     :step 3}}))



