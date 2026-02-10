(ns scratch
  (:require [scicloj.amr.data.ingestion :as ingestion]
            [scicloj.ripple.maldi :as ripple]
            [tablecloth.api :as tc]
            [scicloj.tableplot.v1.plotly :as plotly]))

(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (plotly/base {:=width 700})
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))

(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity ripple/sqrt-transform)
    (plotly/base {:=width 700})
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))


(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity #(-> %
                            ripple/sqrt-transform
                            (ripple/savitzky-golay-smooth
                             {:window-size 11
                              :polynomial-order 2})))
    (plotly/base {:=width 700})
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))

(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity #(-> %
                            ripple/sqrt-transform
                            (ripple/savitzky-golay-smooth
                             {:window-size 11
                              :polynomial-order 2})
                            (ripple/snip-baseline-removal
                             {:iterations 25})))
    (plotly/base {:=width 700})
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))


(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity #(-> %
                            ripple/sqrt-transform
                            (ripple/savitzky-golay-smooth
                             {:window-size 11
                              :polynomial-order 2})
                            (ripple/snip-baseline-removal
                             {:iterations 25})
                            (ripple/tic-normalize
                             {:target-sum 1})))
    (plotly/base {:=width 700})
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))


(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (ripple/preprocess-spectrum-data {})
    (plotly/base {:=width 700})
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))


(let [binning-params {:range [2000 20000]
                      :step 3}
      n-bins (ripple/calculate-n-bins binning-params)]
  (-> (ingestion/example-path)
      ingestion/load-raw-spectrum
      (ripple/preprocess-spectrum-data {})
      (ripple/bin-spectrum binning-params)
      ((fn [features]
         (-> {:feature (range)
              :value features}
             tc/dataset
             (plotly/base {:=width 700})
             (plotly/layer-line {:=x :feature
                                 :=y :value}))))))
