(ns scratch
  (:require [maldi.data.ingestion :as ingestion]
            [maldi.data.signal :as signal]
            [tablecloth.api :as tc]
            [scicloj.tableplot.v1.plotly :as plotly]))

(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))

(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity signal/sqrt-transform)
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))


(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity #(-> %
                            signal/sqrt-transform
                            (signal/savitzky-golay-smooth
                             {:window-size 11
                              :polynomial-order 2})))
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))

(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity #(-> %
                            signal/sqrt-transform
                            (signal/savitzky-golay-smooth
                             {:window-size 11
                              :polynomial-order 2})
                            (signal/snip-baseline-removal
                             {:iterations 25})))
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))


(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity #(-> %
                            signal/sqrt-transform
                            (signal/savitzky-golay-smooth
                             {:window-size 11
                              :polynomial-order 2})
                            (signal/snip-baseline-removal
                             {:iterations 25})
                            (signal/tic-normalize
                             {:target-sum 1})))
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))
