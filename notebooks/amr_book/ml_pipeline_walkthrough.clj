;; # ML Pipeline Walkthrough
;;
;; This notebook demonstrates each stage of the [MALDI-TOF](https://en.wikipedia.org/wiki/Matrix-assisted_laser_desorption/ionization) [AMR](https://en.wikipedia.org/wiki/Antimicrobial_resistance)
;; prediction pipeline, showing both direct (uncached) and cached
;; variants side by side.
;;
;; Here we focus on the caching pattern and how [Pocket](https://github.com/scicloj/pocket) threads stages together.

(ns amr-book.ml-pipeline-walkthrough
  (:require
   ;; AMR ML pipeline (prepare, train, predict, measure):
   [scicloj.amr.learning :as learning]
   ;; AMR data loading utilities:
   [scicloj.amr.data.ingestion :as ingestion]
   ;; Bacterial species definitions and antibiotic lists:
   [scicloj.amr.data.bacteria :as bacteria]
   ;; Ripple MALDI signal processing (https://scicloj.github.io/ripple):
   [scicloj.ripple.maldi :as ripple]
   ;; Table processing (https://scicloj.github.io/tablecloth/):
   [tablecloth.api :as tc]
   ;; Annotating kinds of visualizations (https://scicloj.github.io/kindly-noted/):
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
      (learning/prepare-raw-data-cached)
      deref))

[(tc/row-count raw-data-cached)
 (= (tc/row-count raw-data-no-cache) (tc/row-count raw-data-cached))]

;; ## Stage 2: Prepare ML Data
;;
;; Preprocesses spectra (sqrt, smooth, baseline, normalize) and bins them.
;; Transforms the dataset to have :ri (resistance indicator) and feature columns :x0, :x1, ... :x5999

(def ml-params
  {:preprocessing-params {:smooth-window 21 :smooth-polynomial 3}
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
      (learning/prepare-raw-data-cached)
      (learning/prepare-ml-data-cached ml-params)
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
      (learning/prepare-raw-data-cached)
      (learning/prepare-ml-data-cached ml-params)
      (learning/split-cached split-params)
      deref))

[(keys split-data-cached)
 (tc/row-count (:train split-data-cached))
 (tc/row-count (:test split-data-cached))]

;; ## Stage 4: Train Model
;;
;; Trains [XGBoost](https://xgboost.readthedocs.io/) classifier on training set.
;; Returns trained model map with keys like :model-data, :options, :feature-columns

(def train-params
  {:model-type :xgboost/classification
   :round 50
   :num-class 2})

;; Without caching:
(def model-no-cache
  (learning/train split-data-no-cache train-params))

(keys model-no-cache)

;; With caching:
(def model-cached
  (-> example-params
      (learning/prepare-raw-data-cached)
      (learning/prepare-ml-data-cached ml-params)
      (learning/split-cached split-params)
      (learning/train-cached train-params)
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
                       (learning/prepare-raw-data-cached)
                       (learning/prepare-ml-data-cached ml-params)
                       (learning/split-cached split-params))
        model (-> split-data
                  (learning/train-cached train-params))]
    (-> split-data
        (learning/predict-cached model)
        deref)))

[(tc/row-count predictions-cached)
 (tc/column-names predictions-cached)]

;; ## Stage 6: Measure Performance
;;
;; Calculates [ROCAUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and [PRAUC](https://en.wikipedia.org/wiki/Precision_and_recall) metrics.
;; Returns map with :n-train, :n-test, :pri (prevalence), :PRAUC, :ROCAUC

;; Without caching:
(def metrics-no-cache
  (learning/measure split-data-no-cache predictions-no-cache))

metrics-no-cache

;; With caching:
(def metrics-cached
  (let [split-data (-> example-params
                       (learning/prepare-raw-data-cached)
                       (learning/prepare-ml-data-cached ml-params)
                       (learning/split-cached split-params))
        model (-> split-data
                  (learning/train-cached train-params))
        predictions (-> split-data
                        (learning/predict-cached model))]
    (-> split-data
        (learning/measure-cached predictions)
        deref)))

metrics-cached

;; ## Summary
;;
;; The complete pipeline has 6 stages:
;;
;; 1. **prepare-raw-data**: Load metadata and filter by species/antibiotic/site/year
;; 2. **prepare-ml-data**: Preprocess (sqrt, smooth, baseline, normalize) and bin spectra
;; 3. **split**: Create train/test split
;; 4. **train**: Train XGBoost classifier
;; 5. **predict**: Generate predictions on test set
;; 6. **measure**: Calculate ROCAUC and PRAUC metrics
;;
;; ### Caching Benefits
;;
;; The cached version enables:
;; - **Reusing expensive computations**: Preprocessing spectra is slow; cached results are instant
;; - **Reproducible experiments**: Cache keys are deterministic based on inputs
;; - **Incremental development**: Add pipeline stages without re-running earlier ones
;; - **Parallel execution**: Different scenarios can share cached preprocessing results
;;
;; ### Pipeline Composition
;;
;; Notice the threading pattern in the cached version:
;; ```clojure
;; (-> params
;;     (learning/prepare-raw-data-cached)
;;     (learning/prepare-ml-data-cached ml-params)
;;     deref)
;; ```
;;
;; Each cached function returns a `Cached` (an `IDeref`) that gets
;; passed to the next stage. `deref` forces evaluation at the end.
