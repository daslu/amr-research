(ns scicloj.amr.draft)

(comment
  
  
  (-> @summary
      (tc/select-rows #(and (some-> % :n-test (> 400))
                            (some-> % :species (= bacteria/E-coli))
                            (some-> % :pri (< 0.9))))
      (ds-print/print-range :all))



  
  (-> {:case {:species bacteria/E-coli
              :antibiotic :Ceftazidime ;;:Fosfomycin-Trometamol
              :year 2018
              :site :A}
       :binning-step 3
       :xgboost-rounds 50}
      eval-scenario
      :to-measure
      ((fn [tm]
         [(kind/md (format "probability of R/I: %02f" (-> tm :ri tcc/mean)))
          (kind/md "## ROC curve")
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
      kind/fragment)

  
  (ingestion/all-antibiotics)



  (-> {:case {:species bacteria/S-aureus
              :antibiotic :Oxacillin
              :year 2018
              :site :A}
       :binning-step 3
       :xgboost-rounds 50}
      eval-scenario
      vis)
  ,)




(def eval-scenario-2
  (memoize
   (fn [{:as scenario
         :keys [train-case
                test-case
                binning-step
                xgboost-rounds]}]
     (let [prep (fn [acase]
                  (-> acase
                      ((pocket/caching-fn #'learning/prepare-raw-data))
                      ((pocket/caching-fn #'learning/prepare-ml-data) {:preprocessing-params {}
                                                                       :binning-params {:range [2000 20000]
                                                                                        :step binning-step}})
                      ((pocket/caching-fn #'learning/split) {:seed 1})
                      deref))
           train-data (-> train-case prep :train)
           test-data (-> test-case prep :test)]
       (when (and train-data test-data)
         (log/info [:learning scenario])
         (let [split-data {:train train-data
                           :test test-data}
               {:keys [test train]} split-data
               model (pocket/cached #'learning/train split-data {:model-type :xgboost/classification
                                                                 :round xgboost-rounds
                                                                 :num-class 2})
               predictions @(pocket/cached #'learning/predict split-data model)
               to-measure (some-> predictions
                                  pocket/maybe-deref
                                  (tc/add-column :ri (:ri test)))]
           (merge (:case scenario)
                  (dissoc scenario :case)
                  (learning/measure split-data
                                    predictions)
                  {:to-measure to-measure})))))))

(comment
  (-> {:train-case {:species bacteria/S-aureus
                    :antibiotic :Oxacillin
                    :year 2018
                    :site :A}
       :test-case {:species bacteria/S-aureus
                   :antibiotic :Oxacillin
                   :year 2018
                   :site :A}
       :binning-step 3
       :xgboost-rounds 50}
      eval-scenario-2
      vis)
  

  (kind/fragment
   (for [test-site [:A :C]]
     (-> {:train-case {:species bacteria/E-coli
                       :antibiotic :Cefepime
                       :year 2018
                       :site :A}
          :test-case {:species bacteria/E-coli
                      :antibiotic :Cefepime
                      :year 2018
                      :site test-site}
          :binning-step 3
          :xgboost-rounds 50}
         eval-scenario-2
         vis)))
  

  (kind/fragment
   (for [train-year [2017 2018]]
     (-> {:train-case {:species bacteria/E-coli
                       :antibiotic :Ceftriaxone
                       :year train-year
                       :site :A}
          :test-case {:species bacteria/E-coli
                      :antibiotic :Ceftriaxone
                      :year 2018
                      :site :A}
          :binning-step 3
          :xgboost-rounds 50}
         eval-scenario-2
         vis))))


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
