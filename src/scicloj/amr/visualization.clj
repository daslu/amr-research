(ns scicloj.amr.visualization
  (:require [tablecloth.api :as tc]
            [tablecloth.column.api :as tcc]
            [scicloj.tableplot.v1.plotly :as plotly]
            [scicloj.kindly.v4.kind :as kind])
  (:import (org.tribuo.classification.evaluation LabelEvaluationUtil)))

(defn plot-roc-curve
  "Plot the ROC curve (TPR vs FPR)."
  [actuals probabilities]
  (let [curve (LabelEvaluationUtil/generateROCCurve
               (boolean-array (mapv pos? actuals))
               (double-array probabilities))]
    (-> {:fpr (vec (.fpr curve))
         :tpr (vec (.tpr curve))}
        tc/dataset
        (plotly/base {:=title "ROC curve"
                      :=x :fpr :=y :tpr})
        plotly/layer-line
        plotly/plot)))

(defn plot-pr-curve
  "Plot the precision–recall curve."
  [actuals probabilities]
  (let [curve (LabelEvaluationUtil/generatePRCurve
               (boolean-array (mapv pos? actuals))
               (double-array probabilities))]
    (-> {:recall (vec (.recall curve))
         :precision (vec (.precision curve))}
        tc/dataset
        (plotly/base {:=title "Precision–Recall curve"
                      :=x :recall :=y :precision})
        plotly/layer-line
        plotly/plot)))

(defn plot-calibration
  "Plot predicted probability vs observed resistance rate.
  Samples are sorted by predicted probability and grouped
  into bins of 30."
  [actuals probabilities]
  (-> {:probability probabilities :actual actuals}
      tc/dataset
      (tc/order-by :probability)
      (tc/add-column :i (range))
      (tc/map-columns :bin [:i] #(quot % 30))
      (tc/group-by [:bin])
      (tc/aggregate {:predicted-probability #(tcc/mean (:probability %))
                     :observed-rate #(tcc/mean (:actual %))})
      (plotly/base {:=title "Calibration curve"
                    :=x :predicted-probability
                    :=y :observed-rate})
      plotly/layer-line
      plotly/layer-point
      plotly/plot))

(defn diagnostic-plots
  "Show ROC, precision–recall, and calibration curves
  for a binary classifier."
  [actuals probabilities]
  (kind/fragment
   [(plot-roc-curve actuals probabilities)
    (plot-pr-curve actuals probabilities)
    (plot-calibration actuals probabilities)]))
