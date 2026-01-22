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


(def species->antibiotics
  (-> {bacteria/E-coli ["Meropenem"
                        "Ertapenem"
                        "Ceftriaxone"
                        "Cefepime"
                        "Piperacillin–Tazobactam"
                        "Nitrofurantoin"
                        "Ciprofloxacin"
                        "Cotrimoxazole"]
       bacteria/S-aureus ["Cotrimoxazole"
                          "Clindamycin"
                          "Vancomycin"
                          "Linezolid"
                          "Amoxicillin–Clavulanic acid"
                          "Ampicillin–Amoxicillin"
                          "Oxacillin"]
       bacteria/P-aeruginosa ["Piperacillin-Tazobactam"
                              "Cefepime"
                              "Ceftazidime"
                              "Meropenem"
                              "Amikacin"
                              "Ciprofloxacin"
                              "Colistin"
                              "Tobramycin"]}
      (update-vals (partial map keyword))))



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
                             scenario
                             (learning/measure split-data
                                               predictions)
                             {:to-measure to-measure})]
           (log/info [:done scenario])
           result))))))

(def summary
  (delay
    (-> (for [[species antibiotics] species->antibiotics
              antibiotic antibiotics
              site [:A :B :C :D]
              year [2015 2016 2017 2018]
              xgboost-rounds [50]
              binning-step [3]]
          (let [scenario {:case {:site site
                                 :year year
                                 :antibiotic antibiotic
                                 :species species}
                          :binning-step binning-step
                          :xgboost-rounds xgboost-rounds}]
            (log/info [:scenario scenario])
            (some-> scenario
                    eval-scenario
                    (dissoc :to-measure
                            :case))))
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
      (tc/write-csv! "scenarios-draft-20260122.csv")
      time))




(defn vis [{:as evaluated-scenario
            :keys [case]}]
  (-> evaluated-scenario
      :to-measure
      ((fn [tm]
         [(kind/code (pr-str case))
          (kind/md (format "**actual resistance probability**: %.02f%%" (-> tm :ri tcc/mean (* 100))))
          (kind/md (format "**predictor AUC**: %.02f%%" (* 100 (LabelEvaluationUtil/binaryAUCROC
                                                                (boolean-array (tm :ri))
                                                                (double-array (tm 1))))))
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
                (plotly/base {:=title "ROC curve"
                              :=height 300 :=width 400})
                (plotly/layer-line {:=x :fpr
                                    :=y :tpr})))
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
                            :=y :resistance-probability
                            :=title "calibration curve"
                            :=height 300 :=width 400})
              plotly/layer-line
              plotly/layer-point)
          (kind/hiccup
           [:div {:style "page-break-after: always;"}])]))
      
      kind/fragment))


(->> (for [[species antibiotics] species->antibiotics
           antibiotic antibiotics
           site [:A :B :C :D]
           year [2015 2016 2017 2018]
           xgboost-rounds [50]
           binning-step [3]]
       (let [scenario {:case {:site site
                              :year year
                              :antibiotic antibiotic
                              :species species}
                       :binning-step binning-step
                       :xgboost-rounds xgboost-rounds}]
         (log/info [:scenario scenario])
         (try (some-> scenario
                      eval-scenario
                      vis)
              (catch Exception e nil))))
     (remove nil?)
     kind/fragment)






