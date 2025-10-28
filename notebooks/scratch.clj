(ns scratch
  (:require [maldi.data.ingestion :as ingestion]
            [maldi.data.signal :as signal]
            [maldi.data.binning :as binning]
            [tablecloth.api :as tc]
            [scicloj.tableplot.v1.plotly :as plotly]))

(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (plotly/base {:=width 700})
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))

(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity signal/sqrt-transform)
    (plotly/base {:=width 700})
    (plotly/layer-line {:=x :mass
                        :=y :intensity}))


(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity #(-> %
                            signal/sqrt-transform
                            (signal/savitzky-golay-smooth
                             {:window-size 11
                              :polynomial-order 2})))
    (plotly/base {:=width 700})
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
    (plotly/base {:=width 700})
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
    (plotly/base {:=width 700})
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
    (binning/create-bin-column {:range [2000 20000]
                                :step 3})
    (plotly/base {:=width 700})
    (plotly/layer-line {:=x :bin
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
    (binning/create-bin-column {:range [2000 20000]
                                :step 3})
    binning/aggregate-by-bins
    (tc/order-by :bin)
    (plotly/base {:=width 700})
    (plotly/layer-line {:=x :bin
                        :=y :intensity}))


(let [binning-params {:range [2000 20000]
                      :step 3}
      n-bins (binning/calculate-n-bins binning-params)]
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
      (binning/create-bin-column binning-params)
      binning/aggregate-by-bins
      (binning/bins->array n-bins)
      ((fn [features]
         (-> {:feature (range)
              :value features}
             tc/dataset
             (plotly/base {:=width 700})
             (plotly/layer-line {:=x :feature
                                 :=y :value}))))))


(-> (ingestion/example-path)
    ingestion/load-raw-spectrum
    (update :intensity #(signal/preprocess-spectrum-data % {})))
