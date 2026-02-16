
;; # Diagnostic Plots
;;
;; This notebook shows
;; [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic),
;; [precision–recall](https://en.wikipedia.org/wiki/Precision_and_recall),
;; and calibration curves for every species–antibiotic combination
;; in the [Weis et al.](https://doi.org/10.1038/s41591-021-01619-9) study,
;; across five experiment settings:
;;
;; 1. **Within-site** (site A, 2018) — holdout split, seed 1
;; 2. **Within-site** (site C, 2018) — holdout split, seed 1
;; 3. **Cross-year** (train 2017, test 2018, site A) — no holdout needed
;; 4. **Cross-site A → C** (2018) — train on site A, test on site C
;; 5. **Cross-site C → A** (2018) — train on site C, test on site A
;;
;; All classifiers use [XGBoost](https://xgboost.readthedocs.io/)
;; with 50 boosting rounds.  Data and models are
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
  [species antibiotic site year]
  (let [ml-data (learning/get-ml-data {:site site :year year
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
  antibiotics of one species (within-site holdout)."
  [species site year]
  (let [antibiotics (get bacteria/species->antibiotics species)]
    (->> antibiotics
         (keep (fn [ab]
                 (let [result (within-site-diagnostics species ab site year)]
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

(defn- cross-diagnostics
  "Train on train-data, predict on test-data, return
  {:ROCAUC ... :actuals ... :probabilities ...} or nil."
  [train-data test-data species antibiotic]
  (when (and train-data test-data
             (>= (tc/row-count train-data) 50)
             (>= (tc/row-count test-data) 50))
    (let [pipe (learning/make-pipeline {:rounds 50})
          fitted (mm/fit-pipe train-data pipe)
          predicted (mm/transform-pipe test-data pipe fitted)
          pred-ds (:metamorph/data predicted)
          test-actuals (:ri test-data)
          prob-col (get pred-ds 1)
          rocauc (learning/compute-rocauc test-actuals prob-col)]
      (when rocauc
        {:species species
         :antibiotic antibiotic
         :n-train (tc/row-count train-data)
         :n-test (tc/row-count test-data)
         :ROCAUC rocauc
         :actuals (vec test-actuals)
         :probabilities (vec prob-col)}))))

(defn- cross-year-diagnostics-for-species
  "Diagnostic plots for all antibiotics of one species,
  training on train-year and testing on test-year at the given site."
  [species site train-year test-year]
  (let [antibiotics (get bacteria/species->antibiotics species)]
    (->> antibiotics
         (keep (fn [ab]
                 (let [train (learning/get-ml-data {:site site :year train-year
                                                    :species species :antibiotic ab})
                       test (learning/get-ml-data {:site site :year test-year
                                                   :species species :antibiotic ab})
                       result (cross-diagnostics train test species ab)]
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

(defn- cross-site-diagnostics-for-species
  "Diagnostic plots for all antibiotics of one species,
  training on train-site and testing on test-site."
  [species train-site test-site year]
  (let [antibiotics (get bacteria/species->antibiotics species)]
    (->> antibiotics
         (keep (fn [ab]
                 (let [train (learning/get-ml-data {:site train-site :year year
                                                    :species species :antibiotic ab})
                       test (learning/get-ml-data {:site test-site :year year
                                                   :species species :antibiotic ab})
                       result (cross-diagnostics train test species ab)]
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

(defn- all-species-diagnostics
  "Generate diagnostic plots for all species in
  `bacteria/important-bacteria` using the given plot function.
  Each species gets a ## heading."
  [plot-fn]
  (->> bacteria/important-bacteria
       (mapv (fn [species]
               (kind/fragment
                [(kind/md (str "## " species))
                 (plot-fn species)])))
       kind/fragment))

;; ---

;; # Within-site Predictions (Site A, 2018)
;;
;; Each case uses a random holdout split (seed 1).

(all-species-diagnostics
 (fn [species] (diagnostics-for-species species :A 2018)))

;; ---

;; # Within-site Predictions (Site C, 2018)
;;
;; Site C is the second-largest DRIAMS site with AMR labels.
;; The same holdout split and XGBoost configuration are used.

(all-species-diagnostics
 (fn [species] (diagnostics-for-species species :C 2018)))

;; ---

;; # Cross-year Predictions (2017 → 2018, Site A)
;;
;; The model is trained on the full 2017 dataset at site A
;; and evaluated on the full 2018 dataset — no holdout split
;; is needed since train and test are from different years.

(all-species-diagnostics
 (fn [species] (cross-year-diagnostics-for-species species :A 2017 2018)))

;; ---

;; # Cross-site Predictions (A → C, 2018)
;;
;; The model is trained on the full site A dataset (2018)
;; and evaluated on the full site C dataset (2018).
;; Sites B and D lack AMR labels, so cross-site is limited to A ↔ C.

(all-species-diagnostics
 (fn [species] (cross-site-diagnostics-for-species species :A :C 2018)))

;; ---

;; # Cross-site Predictions (C → A, 2018)
;;
;; The reverse direction: trained on site C, tested on site A.
;; Site C has fewer samples, so these models are expected to
;; perform worse than the A → C direction.

(all-species-diagnostics
 (fn [species] (cross-site-diagnostics-for-species species :C :A 2018)))
