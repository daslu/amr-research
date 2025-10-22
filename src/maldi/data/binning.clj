(ns maldi.data.binning
  (:require [tablecloth.api :as tc]
            [tablecloth.column.api :as tcc]
            [tech.v3.datatype :as dtype]
            [tech.v3.dataset.reductions :as ds-reduce]
            [ham-fisted.api :as hamf]
            [maldi.errors :as errors]
            [clojure.tools.logging :as log]))

(defn create-bin-column
  "Add bin column to dataset, filtering out values outside range"
  [dataset {:keys [range step] :as params}]
  (let [[min-val max-val] range]
    (-> dataset
        ;; Filter to only include values within range [min-val, max-val]
        (tc/select-rows #(and (>= (:mass %) min-val)
                              (<= (:mass %) max-val)))
        ;; Add bin column for remaining values
        (tc/add-column :bin
                       #(-> %
                            :mass
                            (tcc/- min-val)
                            (tcc// step)
                            tcc/round
                            (tcc/+ 0.0))))))

(defn aggregate-by-bins
  "Aggregate intensities by bin"
  [dataset]
  (if (zero? (tc/row-count dataset))
    ;; Return empty dataset with correct structure for empty input
    (tc/dataset {:bin [] :intensity []})
    (ds-reduce/group-by-column-agg
     :bin
     {:intensity (ds-reduce/sum :intensity)}
     dataset)))

(defn bins->array
  "Convert binned data to fixed-size array"
  [binned-data n-bins]
  (let [arr (double-array n-bins)]
    ;; Handle empty or special case datasets
    (when (and binned-data
               (satisfies? tech.v3.dataset.protocols/PDataset binned-data)
               (pos? (tc/row-count binned-data)))
      (doseq [row (tc/rows binned-data)]
        (let [bin-idx (int (first row))
              intensity (double (second row))]
          (when (and (>= bin-idx 0) (< bin-idx n-bins))
            (aset arr bin-idx intensity)))))
    arr))

(defn calculate-n-bins
  "Calculate number of bins for given range and step with safety limits"
  [{:keys [range step]}]
  (let [[min-val max-val] range
        n-bins (inc (int (/ (- max-val min-val) step)))]

    ;; Safety check for memory usage
    (when (> n-bins 10000000) ; 10M bins = ~80MB for double array
      (throw (ex-info "Binning would create too many bins (memory safety limit)"
                      {:calculated-bins n-bins
                       :max-allowed 10000000
                       :estimated-memory-mb (/ (* n-bins 8) 1024 1024)
                       :range range
                       :step step})))

    ;; Warn about large allocations
    (when (> n-bins 1000000) ; 1M bins = ~8MB
      (log/warn (format "Large memory allocation: %d bins (~%.1f MB)"
                        n-bins (/ (* n-bins 8) 1024.0 1024.0))))

    n-bins))

(defn bin-spectrum
  "Bin a preprocessed spectrum into fixed-width bins"
  [spectrum params]
  (let [n-bins (calculate-n-bins params)]
    (errors/with-error-handling errors/pipeline-error {:stage :binning
                                                       :params params}
      (-> spectrum
          (create-bin-column params)
          aggregate-by-bins
          (bins->array n-bins)))))

