^{:clay {:hide-code true
         :hide-info-line true
         :hide-ui-header true}}
(ns scicloj.amr.scenarios
  (:require [scicloj.pocket :as pocket]
            [tablecloth.api :as tc]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.amr.learning :as learning]
            [clojure.tools.logging :as log]
            [scicloj.amr.data.bacteria :as bacteria]
            [scicloj.amr.data.ingestion :as ingestion]
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

(defonce eval-scenario
  (memoize
   (fn [{:as scenario
         :keys [binning-step
                xgboost-rounds]}]
     (let [ml-data (-> (learning/prepare-raw-data-cached (:case scenario))
                       (learning/prepare-ml-data-cached {:preprocessing-params {:smooth-window 21 :smooth-polynomial 3}
                                                         :binning-params {:range [2000 20000]
                                                                          :step binning-step}}))]
       (when @ml-data
         (let [split-data (-> ml-data
                              (learning/split-cached {:seed 1})
                              pocket/maybe-deref)
               {:keys [train test]} split-data
               model (pocket/cached #'learning/train split-data {:model-type :xgboost/classification
                                                                 :round xgboost-rounds
                                                                 :num-class 2})
               predictions @(pocket/cached #'learning/predict split-data model)
               to-measure (some-> predictions
                                  pocket/maybe-deref
                                  (tc/add-column :ri (:ri test)))
               result (merge (:case scenario)
                             scenario
                             (learning/measure split-data
                                               predictions)
                             {:to-measure to-measure})]
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
            (some-> scenario
                    eval-scenario
                    (dissoc :to-measure
                            :case))))
        (->> (remove nil?))
        tc/dataset
        (tc/select-rows #(some-> % :n-test pos?))
        (tc/order-by [:n-test])
        (ds-print/print-range :all))))

(comment
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

(def break
  (kind/hiccup
   [:div {:style "page-break-after: always;"}]))

(defn adjust-layout [template]
  (-> template
      plotly/plot
      (update :layout merge {:height 300
                             :width 300
                             :showlegend false})
      (assoc-in [:layout :margin]
                {:l 50 :r 20 :b 50 :t 50})))

(defn vis [{:as evaluated-scenario
            :keys [case]}]
  (-> evaluated-scenario
      :to-measure
      ((fn [tm]
         [(kind/table (update-vals case vector))
          (kind/table
           {"actual resistance probability"
            [(format "%.02f%%" (-> tm :ri tcc/mean (* 100)))]
            "predictor AUC"
            [(format "%.02f%%" (* 100 (LabelEvaluationUtil/binaryAUCROC
                                       (boolean-array (tm :ri))
                                       (double-array (tm 1)))))]})
          (kind/table
           [[(let [curve (LabelEvaluationUtil/generatePRCurve
                          (boolean-array (tm :ri))
                          (double-array (tm 1)))]
               (-> {:precision (.precision curve)
                    :recall (.recall curve)}
                   tc/dataset
                   (ds-print/print-range :all)
                   (tc/order-by [:precision])
                   (plotly/base {:=title "precision-recall curve"})
                   (plotly/layer-line {:=x :precision
                                       :=y :recall})
                   adjust-layout))
             (let [curve (LabelEvaluationUtil/generateROCCurve
                          (boolean-array (tm :ri))
                          (double-array (tm 1)))]
               (-> {:fpr (.fpr curve)
                    :tpr (.tpr curve)}
                   tc/dataset
                   (ds-print/print-range :all)
                   (tc/order-by [:precision])
                   (plotly/base {:=title "ROC curve"})
                   (plotly/layer-line {:=x :fpr
                                       :=y :tpr})
                   adjust-layout))]])
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
                            :=title "calibration curve"})
              plotly/layer-line
              plotly/layer-point
              adjust-layout)
          break]))

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
         (try (some-> scenario
                      eval-scenario
                      vis)
              (catch Exception e nil))))
     (remove nil?)
     kind/fragment)
