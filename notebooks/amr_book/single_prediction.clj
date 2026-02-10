;; # A Single AMR Prediction
;;
;; This notebook walks through the simplest case: predicting
;; [antimicrobial resistance](https://en.wikipedia.org/wiki/Antimicrobial_resistance) for **one species and one antibiotic**
;; using data from a single hospital site and year.
;;
;; We call the pipeline functions directly (no caching) so each
;; step is transparent.

(ns amr-book.single-prediction
  (:require
   ;; AMR ML pipeline (prepare, train, predict, measure):
   [scicloj.amr.learning :as learning]
   ;; Bacterial species definitions and antibiotic lists:
   [scicloj.amr.data.bacteria :as bacteria]
   ;; Table processing (https://scicloj.github.io/tablecloth/):
   [tablecloth.api :as tc]
   ;; Annotating kinds of visualizations (https://scicloj.github.io/kindly-noted/):
   [scicloj.kindly.v4.kind :as kind]))

;; ## Choosing a scenario
;;
;; We'll predict **Cefepime** resistance in **E. coli**
;; from DRIAMS-A 2018 — one of the cases reported by
;; [Weis et al.](https://doi.org/10.1038/s41591-021-01619-9):

(def params
  {:site :A
   :year 2018
   :species bacteria/E-coli
   :antibiotic :Cefepime})

;; ## Stage 1 — Prepare raw data
;;
;; Load DRIAMS metadata, join with available spectra,
;; and filter to the chosen species/antibiotic.
;; The result has a `:ri` column (true = resistant or intermediate)
;; and a `:path` column pointing to each raw spectrum file.

(def raw-data
  (learning/prepare-raw-data params))

(tc/row-count raw-data)

(tc/head (tc/select-columns raw-data [:code :species :ri :path]) 5)

;; ## Stage 2 — Prepare ML data
;;
;; This is the expensive step. For each spectrum:
;; 1. Load the raw `.txt.gz` file
;; 2. Preprocess (sqrt, smooth, baseline, normalize) via [Ripple](https://scicloj.github.io/ripple)
;; 3. Bin to 6,000 features (3 Da bins over [2000, 20000] Da)
;;
;; The result is a dataset with columns `:ri`, `:x0`, `:x1`, ... `:x5999`.

(def ml-params
  {:preprocessing-params {}
   :binning-params {:range [2000 20000] :step 3}})

(def ml-data
  (learning/prepare-ml-data raw-data ml-params))

{:rows (tc/row-count ml-data)
 :features (dec (count (tc/column-names ml-data)))}

;; ## Stage 3 — Train/test split
;;
;; A holdout split with a fixed random seed for reproducibility.
;; Requires at least 100 samples (returns nil otherwise).

(def split-data
  (learning/split ml-data {:seed 1}))

{:train (tc/row-count (:train split-data))
 :test (tc/row-count (:test split-data))}

;; ## Stage 4 — Train
;;
;; Train an [XGBoost](https://xgboost.readthedocs.io/) binary classifier
;; on the training set:

(def model
  (learning/train split-data
                  {:model-type :xgboost/classification
                   :round 50
                   :num-class 2}))

;; ## Stage 5 — Predict
;;
;; Generate probability predictions on the test set.
;; The result has columns `0` (probability of susceptible)
;; and `1` (probability of resistant/intermediate):

(def predictions
  (learning/predict split-data model))

(tc/head predictions 5)

;; ## Stage 6 — Measure
;;
;; Evaluate with [ROCAUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
;; and [PRAUC](https://en.wikipedia.org/wiki/Precision_and_recall) (area under the
;; precision-recall curve):

(def metrics
  (learning/measure split-data predictions))

metrics

(kind/test-last
 #(< 0.5 (:ROCAUC %)))

;; ### Reading the results
;;
;; - **ROCAUC** close to 1.0 means the model separates resistant
;;   from susceptible cases well.
;; - **PRAUC** is especially informative when resistance is rare
;;   (low `:pri`), since it accounts for class imbalance.
;; - **`:pri`** is the prevalence of resistance in the test set.

;; ## Summary
;;
;; The six stages form a linear pipeline:
;;
;; ```
;; params → prepare-raw-data → prepare-ml-data → split → train → predict → measure
;; ```
;;
;; Each function takes the output of the previous stage plus
;; its own configuration. Wrapping each stage with
;; [Pocket](https://github.com/scicloj/pocket)'s caching avoids recomputing expensive steps.
