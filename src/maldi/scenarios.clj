(ns maldi.scenarios
  (:require [maldi.cache :as cache]
            [tablecloth.api :as tc]
            [scicloj.kindly.v4.kind :as kind]
            [maldi.learning :as learning]
            [clojure.tools.logging :as log]
            [maldi.data.bacteria :as bacteria]
            [maldi.data.ingestion :as ingestion]
            [tech.v3.dataset.print :as ds-print]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tablecloth.column.api :as tcc])
  (:import (org.tribuo.classification.evaluation LabelEvaluationUtil)))



(def eval-scenario
  (memoize
   (fn [{:as scenario
         :keys [binning-step
                xgboost-rounds]}]
     (let [ml-data (-> ((cache/cached-fn #'learning/prepare-raw-data) (:case scenario))
                       ((cache/cached-fn #'learning/prepare-ml-data) {:preprocessing-params {}
                                                                      :binning-params {:range [2000 20000]
                                                                                       :step binning-step}}))]
       (when @ml-data
         (log/info [:learning scenario])
         (let [split-data (-> ml-data
                              ((cache/cached-fn #'learning/split) {:seed 1})
                              cache/maybe-deref)
               {:keys [train test]} split-data
               model (cache/cached #'learning/train split-data {:model-type :xgboost/classification
                                                                :round xgboost-rounds
                                                                :num-class 2})
               predictions @(cache/cached #'learning/predict split-data model)
               to-measure (some-> predictions
                                  cache/maybe-deref
                                  (tc/add-column :ri (:ri test)))
               result (merge (:case scenario)
                             (dissoc scenario :case)
                             (learning/measure split-data
                                               predictions)
                             {:to-measure to-measure})]
           (log/info [:done scenario])
           result))))))

(def summary
  (delay
    (-> (for [xgboost-rounds [50]
              binning-step [3]
              site [:A :B :C :D]
              year [2015 2016 2017 2018]
              antibiotic (ingestion/all-antibiotics)
              species [bacteria/E-coli
                       bacteria/S-aureus
                       bacteria/P-aeruginosai]]
          (let [scenario {:case {:site site
                                 :year year
                                 :antibiotic antibiotic
                                 :species species}
                          :binning-step binning-step
                          :xgboost-rounds xgboost-rounds}]
            (log/info [:scenario scenario])
            (some-> scenario
                    eval-scenario
                    (dissoc :to-measure))))
        (->> (remove nil?))
        tc/dataset
        (tc/select-rows #(some-> % :n-test pos?))
        (tc/order-by [:n-test])
        (ds-print/print-range :all))))



(delay
  (-> @summary
      (tc/select-columns [:species :antibiotic
                          :site :year
                          :pri
                          :n-train :n-test
                          :PRAUC :ROCAUC])
      (tc/rename-columns {:n-train "train cases"
                          :n-test "test cases"
                          :pri "probability of R/I"})
      (tc/order-by [:species :antibiotic :site :year])
      (tc/write-csv! "scenarios-draft-20260116.csv")
      time))




(defn vis [{:as evaluated-scenario
            :keys [train-case test-case]}]
  (-> evaluated-scenario
      :to-measure
      ((fn [tm]
         [(kind/md "## scenario")
          (kind/table
           [(merge {"." "training"} train-case)
            (merge {"." "testing"} test-case)])
          (kind/md "## statistics")
          (kind/md (format "**actual resistance probability**: %.02f%%" (-> tm :ri tcc/mean (* 100))))
          (kind/md (format "**predictor AUC**: %.02f%%" (* 100 (LabelEvaluationUtil/binaryAUCROC
                                                                (boolean-array (tm :ri))
                                                                (double-array (tm 1))))))
          (kind/md "## ROC curve")
          #_(let [curve (LabelEvaluationUtil/generatePRCurve
                         (boolean-array (tm :ri))
                         (double-array (tm 1)))]
              (-> {:precision (.precision curve)
                   :recall (.recall curve)}
                  tc/dataset
                  (ds-print/print-range :all)
                  (tc/order-by [:precision])
                  (plotly/layer-line {:=x :precision
                                      :=y :recall})))
          (let [curve (LabelEvaluationUtil/generateROCCurve
                       (boolean-array (tm :ri))
                       (double-array (tm 1)))]
            (-> {:fpr (.fpr curve)
                 :tpr (.tpr curve)}
                tc/dataset
                (ds-print/print-range :all)
                (tc/order-by [:precision])
                (plotly/layer-line {:=x :fpr
                                    :=y :tpr})))
          (kind/md "## calibration curve")
          (-> tm
              (tc/order-by 1)
              (tc/add-column :i (range))
              (tc/map-columns :g :i #(quot % 30))
              (tc/group-by [:g])
              (tc/aggregate {:signal #(-> 1
                                          %
                                          ((juxt tcc/reduce-min
                                                 tcc/reduce-max))
                                          tcc/mean)
                             :resistance-probability #(-> :ri
                                                          %
                                                          tcc/mean)
                             :n #(tc/row-count %)})
              (plotly/base {:=x :signal
                            :=y :resistance-probability})
              plotly/layer-line
              plotly/layer-point)]))
      kind/fragment))






