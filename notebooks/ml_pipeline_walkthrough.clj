(ns ml-pipeline-walkthrough
  "Progressive walkthrough of the complete ML pipeline.
   
   This notebook demonstrates each stage of the MALDI-TOF AMR prediction pipeline,
   building progressively from data ingestion to final evaluation.
   
   Each stage is shown twice:
   - Without caching (direct function calls)
   - With caching (using cache/cached-fn)"
  (:require [maldi.learning :as learning]
            [maldi.cache :as cache]
            [maldi.data.bacteria :as bacteria]
            [tablecloth.api :as tc]
            [scicloj.kindly.v4.kind :as kind]))

;; ## Example Configuration
;;
;; We'll use a single example throughout:
;; - **Species**: E. coli
;; - **Antibiotic**: Cefepime
;; - **Site**: A
;; - **Year**: 2018

(def example-params
  {:site :A
   :year 2018
   :species bacteria/E-coli
   :antibiotic :Cefepime})

example-params

;; ## Stage 1: Prepare Raw Data
;;
;; Loads metadata and raw spectra, filters by species/antibiotic/site/year.
;; Returns a dataset with columns including :code, :Cefepime (resistance), :path

;; Without caching:
(def raw-data-no-cache
  (learning/prepare-raw-data example-params))

[(tc/row-count raw-data-no-cache)
 (take 10 (tc/column-names raw-data-no-cache))]

;; With caching:
(def raw-data-cached
  (-> example-params
      ((cache/cached-fn #'learning/prepare-raw-data))
      deref))

[(tc/row-count raw-data-cached)
 (= (tc/row-count raw-data-no-cache) (tc/row-count raw-data-cached))]

;; ## Stage 2: Prepare ML Data
;;
;; Preprocesses spectra (sqrt, smooth, baseline, normalize) and bins them.
;; Transforms the dataset to have :ri (resistance indicator) and feature columns :x0, :x1, ... :x6000

(def ml-params
  {:preprocessing-params {}
   :binning-params {:range [2000 20000]
                    :step 3}})

;; Without caching (slow - preprocessing + binning 1400 spectra):
(def ml-data-no-cache
  (learning/prepare-ml-data raw-data-no-cache ml-params))

[(tc/row-count ml-data-no-cache)
 (take 5 (tc/column-names ml-data-no-cache))]

;; With caching:
(def ml-data-cached
  (-> example-params
      ((cache/cached-fn #'learning/prepare-raw-data))
      ((cache/cached-fn #'learning/prepare-ml-data) ml-params)
      deref))

[(tc/row-count ml-data-cached)
 (= (tc/row-count ml-data-no-cache) (tc/row-count ml-data-cached))]

;; ## Stage 3: Split Data
;;
;; Splits into train/test sets (default ~66/33 split).
;; Requires ≥100 samples, returns nil otherwise.
;; Returns map with :train and :test datasets.

(def split-params
  {:seed 1})

;; Without caching:
(def split-data-no-cache
  (learning/split ml-data-no-cache split-params))

[(keys split-data-no-cache)
 (tc/row-count (:train split-data-no-cache))
 (tc/row-count (:test split-data-no-cache))]

;; With caching:
(def split-data-cached
  (-> example-params
      ((cache/cached-fn #'learning/prepare-raw-data))
      ((cache/cached-fn #'learning/prepare-ml-data) ml-params)
      ((cache/cached-fn #'learning/split) split-params)
      deref))

[(keys split-data-cached)
 (tc/row-count (:train split-data-cached))
 (tc/row-count (:test split-data-cached))]

;; ## Stage 4: Train Model
;;
;; Trains XGBoost classifier on training set.
;; Returns trained model map with keys like :model-data, :options, :feature-columns

(def train-params
  {:model-type :xgboost/classification
   :round 10
   :num-class 2})

;; Without caching:
(def model-no-cache
  (learning/train split-data-no-cache train-params))

(keys model-no-cache)

;; With caching:
(def model-cached
  (-> example-params
      ((cache/cached-fn #'learning/prepare-raw-data))
      ((cache/cached-fn #'learning/prepare-ml-data) ml-params)
      ((cache/cached-fn #'learning/split) split-params)
      ((cache/cached-fn #'learning/train) train-params)
      deref))

(keys model-cached)

;; ## Stage 5: Predict
;;
;; Generates predictions on test set.
;; Returns dataset with prediction columns (0, 1, and :ri for actual labels).

;; Without caching:
(def predictions-no-cache
  (learning/predict split-data-no-cache model-no-cache))

[(tc/row-count predictions-no-cache)
 (tc/column-names predictions-no-cache)]

(tc/head predictions-no-cache 5)

;; With caching:
(def predictions-cached
  (let [split-data (-> example-params
                       ((cache/cached-fn #'learning/prepare-raw-data))
                       ((cache/cached-fn #'learning/prepare-ml-data) ml-params)
                       ((cache/cached-fn #'learning/split) split-params))
        model (-> split-data
                  ((cache/cached-fn #'learning/train) train-params))]
    (-> split-data
        ((cache/cached-fn #'learning/predict) model)
        deref)))

[(tc/row-count predictions-cached)
 (tc/column-names predictions-cached)]

;; ## Stage 6: Measure Performance
;;
;; Calculates ROCAUC and PRAUC metrics.
;; Returns map with :n-train, :n-test, :pri (prevalence), :PRAUC, :ROCAUC

;; Without caching:
(def metrics-no-cache
  (learning/measure split-data-no-cache predictions-no-cache))

metrics-no-cache

;; With caching:
(def metrics-cached
  (let [split-data (-> example-params
                       ((cache/cached-fn #'learning/prepare-raw-data))
                       ((cache/cached-fn #'learning/prepare-ml-data) ml-params)
                       ((cache/cached-fn #'learning/split) split-params))
        model (-> split-data
                  ((cache/cached-fn #'learning/train) train-params))
        predictions (-> split-data
                        ((cache/cached-fn #'learning/predict) model))]
    (-> split-data
        ((cache/cached-fn #'learning/measure) predictions)
        deref)))

metrics-cached

;; ## Summary
;;
;; The complete pipeline has 6 stages:
;;
;; 1. **prepare-raw-data**: Load metadata and filter by species/antibiotic/site/year → 1400 samples
;; 2. **prepare-ml-data**: Preprocess (sqrt, smooth, baseline, normalize) and bin spectra → 6001 features per sample
;; 3. **split**: Create train/test split → 933 train, 467 test
;; 4. **train**: Train XGBoost classifier (10 rounds) → model map
;; 5. **predict**: Generate predictions on test set → dataset with class probabilities
;; 6. **measure**: Calculate ROCAUC (0.78) and PRAUC (0.60) metrics
;;
;; ### Caching Benefits
;;
;; The cached version enables:
;; - **Reusing expensive computations**: Preprocessing 1400 spectra is slow; cached results are instant
;; - **Reproducible experiments**: Cache keys are deterministic based on inputs
;; - **Incremental development**: Add pipeline stages without re-running earlier ones
;; - **Parallel execution**: Different scenarios can share cached preprocessing results
;;
;; ### Pipeline Composition
;;
;; Notice the threading pattern in the cached version:
;; ```clojure
;; (-> params
;;     ((cache/cached-fn #'fn1))
;;     ((cache/cached-fn #'fn2) params2)
;;     deref)
;; ```
;;
;; Each function returns a delay/future that gets passed to the next stage,
;; and `deref` forces evaluation at the end.
