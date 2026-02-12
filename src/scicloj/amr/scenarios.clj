(ns scicloj.amr.scenarios
  (:require [scicloj.amr.learning :as learning]
            [scicloj.amr.data.bacteria :as bacteria]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [tablecloth.api :as tc]))

(defn evaluate-cross-split
  "Train a pipeline on `train-data`, predict on `test-data`,
  and return metrics. Returns nil on failure."
  [pipeline train-data test-data]
  (try
    (let [fitted-ctx (mm/fit-pipe train-data pipeline)
          predicted-ctx (mm/transform-pipe test-data pipeline fitted-ctx)
          pred-ds (:metamorph/data predicted-ctx)]
      {:n-train (tc/row-count train-data)
       :n-test (tc/row-count test-data)
       :ROCAUC (learning/compute-rocauc (:ri test-data) (get pred-ds 1))})
    (catch Exception _e nil)))

(defn run-within-site
  "Within-site holdout experiment.
  Returns a metrics map with :status (:ok, :no-data, :too-few,
  :single-class, or :error)."
  [{:keys [species antibiotic site year rounds]
    :or {rounds 50}}]
  (let [base {:species species :antibiotic antibiotic
              :site site :year year
              :experiment :within-site}]
    (try
      (let [ml-data (learning/get-ml-data {:site site :year year
                                           :species species
                                           :antibiotic antibiotic})]
        (cond
          (nil? ml-data)
          (assoc base :status :no-data)

          (< (tc/row-count ml-data) 100)
          (assoc base :status :too-few
                 :n-total (tc/row-count ml-data))

          :else
          (let [split (first (tc/split->seq ml-data :holdout {:seed 1}))
                pipe (learning/make-pipeline {:rounds rounds})
                fitted (mm/fit-pipe (:train split) pipe)
                predicted (mm/transform-pipe (:test split) pipe fitted)
                pred-ds (:metamorph/data predicted)
                rocauc (learning/compute-rocauc (:ri (:test split))
                                                (get pred-ds 1))]
            (merge base
                   {:n-train (tc/row-count (:train split))
                    :n-test (tc/row-count (:test split))
                    :ROCAUC rocauc
                    :status (if rocauc :ok :single-class)}))))
      (catch Exception e
        (assoc base :status :error
               :message (.getMessage e))))))

(defn run-cross-year
  "Train on train-year, test on test-year.
  Returns a metrics map with :status."
  [{:keys [species antibiotic site train-year test-year rounds]
    :or {rounds 50}}]
  (let [base {:species species :antibiotic antibiotic
              :site site
              :train-year train-year :test-year test-year
              :experiment :cross-year}]
    (try
      (let [train-data (learning/get-ml-data {:site site :year train-year
                                              :species species
                                              :antibiotic antibiotic})
            test-data (learning/get-ml-data {:site site :year test-year
                                             :species species
                                             :antibiotic antibiotic})]
        (cond
          (nil? train-data)
          (assoc base :status :no-data :detail :no-train)

          (nil? test-data)
          (assoc base :status :no-data :detail :no-test)

          (< (tc/row-count train-data) 50)
          (assoc base :status :too-few
                 :n-train (tc/row-count train-data))

          (< (tc/row-count test-data) 50)
          (assoc base :status :too-few
                 :n-test (tc/row-count test-data))

          :else
          (let [result (evaluate-cross-split (learning/make-pipeline {:rounds rounds})
                                             train-data test-data)]
            (merge base result
                   {:status (if (:ROCAUC result) :ok :single-class)}))))
      (catch Exception e
        (assoc base :status :error
               :message (.getMessage e))))))

(defn run-cross-site
  "Train on train-site, test on test-site.
  Returns a metrics map with :status."
  [{:keys [species antibiotic train-site test-site year rounds]
    :or {rounds 50}}]
  (let [base {:species species :antibiotic antibiotic
              :train-site train-site :test-site test-site
              :year year
              :experiment :cross-site}]
    (try
      (let [train-data (learning/get-ml-data {:site train-site :year year
                                              :species species
                                              :antibiotic antibiotic})
            test-data (learning/get-ml-data {:site test-site :year year
                                             :species species
                                             :antibiotic antibiotic})]
        (cond
          (nil? train-data)
          (assoc base :status :no-data :detail :no-train)

          (nil? test-data)
          (assoc base :status :no-data :detail :no-test)

          (< (tc/row-count train-data) 50)
          (assoc base :status :too-few
                 :n-train (tc/row-count train-data))

          (< (tc/row-count test-data) 50)
          (assoc base :status :too-few
                 :n-test (tc/row-count test-data))

          :else
          (let [result (evaluate-cross-split (learning/make-pipeline {:rounds rounds})
                                             train-data test-data)]
            (merge base result
                   {:status (if (:ROCAUC result) :ok :single-class)}))))
      (catch Exception e
        (assoc base :status :error
               :message (.getMessage e))))))
