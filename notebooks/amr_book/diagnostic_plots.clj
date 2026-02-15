;; # Diagnostic Plots — Within-site Predictions
;;
;; This notebook shows
;; [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic),
;; [precision–recall](https://en.wikipedia.org/wiki/Precision_and_recall),
;; and calibration curves for every species–antibiotic combination
;; in the [Weis et al.](https://doi.org/10.1038/s41591-021-01619-9)
;; study where within-site prediction (site A, year 2018)
;; produces a valid classifier.
;;
;; Each case uses a random holdout split (seed 1) and
;; [XGBoost](https://xgboost.readthedocs.io/) with 50
;; boosting rounds.  All data and models are
;; [Pocket](https://github.com/scicloj/pocket)-cached.

(ns amr-book.diagnostic-plots
  (:require
   ;; AMR ML pipeline:
   [scicloj.amr.learning :as learning]
   ;; Diagnostic plots (ROC, PR, calibration):
   [scicloj.amr.visualization :as viz]
   ;; Bacterial species definitions:
   [scicloj.amr.data.bacteria :as bacteria]
   ;; Metamorph pipeline composition:
   [scicloj.metamorph.core :as mm]
   ;; Table processing:
   [tablecloth.api :as tc]
   ;; Annotating kinds of visualizations:
   [scicloj.kindly.v4.kind :as kind]))

;; ---

;; ## Helpers

(defn- within-site-diagnostics
  "Run within-site holdout for one species–antibiotic and
  return {:ROCAUC ... :actuals ... :probabilities ...},
  or nil if data is insufficient or single-class."
  [species antibiotic]
  (let [ml-data (learning/get-ml-data {:site :A :year 2018
                                       :species species
                                       :antibiotic antibiotic})]
    (when (and ml-data (>= (tc/row-count ml-data) 100))
      (let [split (first (tc/split->seq ml-data :holdout {:seed 1}))
            pipe (learning/make-pipeline {:rounds 50})
            fitted (mm/fit-pipe (:train split) pipe)
            predicted (mm/transform-pipe (:test split) pipe fitted)
            pred-ds (:metamorph/data predicted)
            test-actuals (:ri (:test split))
            prob-col (get pred-ds 1)
            rocauc (learning/compute-rocauc test-actuals prob-col)]
        (when rocauc
          {:species species
           :antibiotic antibiotic
           :n-train (tc/row-count (:train split))
           :n-test (tc/row-count (:test split))
           :ROCAUC rocauc
           :actuals (vec test-actuals)
           :probabilities (vec prob-col)})))))

(defn- diagnostics-for-species
  "Build a fragment with diagnostic plots for all
  antibiotics of one species."
  [species]
  (let [antibiotics (get bacteria/species->antibiotics species)]
    (->> antibiotics
         (keep (fn [ab]
                 (let [result (within-site-diagnostics species ab)]
                   (when result
                     (kind/fragment
                      [(kind/md (str "### " (name ab)
                                     " — ROCAUC: "
                                     (format "%.3f" (:ROCAUC result))
                                     " (n-train=" (:n-train result)
                                     ", n-test=" (:n-test result) ")"))
                       (viz/diagnostic-plots
                        (:actuals result)
                        (:probabilities result))])))))
         vec
         kind/fragment)))

;; ---

;; ## Escherichia coli

(diagnostics-for-species bacteria/E-coli)

;; ---

;; ## Staphylococcus aureus

(diagnostics-for-species bacteria/S-aureus)

;; ---

;; ## Pseudomonas aeruginosa

(diagnostics-for-species bacteria/P-aeruginosa)
