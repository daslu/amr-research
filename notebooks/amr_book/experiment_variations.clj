
;; # Experiment Variations
;;
;; This notebook runs many variations of
;; [AMR](https://en.wikipedia.org/wiki/Antimicrobial_resistance)
;; prediction experiments, combining
;; [Pocket](https://github.com/scicloj/pocket) for caching
;; and [metamorph.ml](https://github.com/scicloj/metamorph.ml)
;; for pipeline composition and evaluation.
;;
;; We grow the scope incrementally:
;;
;; 1. **Within-site holdout** — one scenario, multiple XGBoost configs
;; 2. **Cross-year** — train on one year, test on another
;; 3. **Cross-site** — train on one hospital, test on another
;; 4. **Multi-species sweep** — all species and antibiotics
;; 5. **Bidirectional cross-site** — A→C vs C→A transfer
;; 6. **K-fold cross-validation** — robust estimates with standard deviations
;; 7. **Within-site at site C** — how does a smaller hospital compare?
;; 8. **Comprehensive comparison** — all experiment types side by side

(ns amr-book.experiment-variations
  (:require
   ;; AMR ML pipeline (prepare, train, predict, measure):
   [scicloj.amr.learning :as learning]
   ;; Experiment runners (within-site, cross-year, cross-site):
   [scicloj.amr.scenarios :as scenarios]
   ;; Diagnostic plots (ROC, PR, calibration):
   [scicloj.amr.visualization :as viz]
   ;; Bacterial species definitions:
   [scicloj.amr.data.bacteria :as bacteria]
   ;; Pocket filesystem caching (https://github.com/scicloj/pocket):
   [scicloj.pocket :as pocket]
   ;; Metamorph pipeline composition (https://github.com/scicloj/metamorph.ml):
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   ;; Table processing (https://scicloj.github.io/tablecloth/):
   [tablecloth.api :as tc]
   ;; Column-level operations:
   [tablecloth.column.api :as tcc]
   ;; Interactive plotting (https://scicloj.github.io/tableplot/):
   [scicloj.tableplot.v1.plotly :as plotly]
   ;; Annotating kinds of visualizations:
   [scicloj.kindly.v4.kind :as kind]))

;; ## Shared parameters
;;
;; Preprocessing and binning match the
;; [Weis et al.](https://doi.org/10.1038/s41591-021-01619-9)
;; paper throughout.

;; ## Configuration
;;
;; Pocket reads `pocket.edn` at project root on startup.
;; The cache directory lives outside Dropbox (to avoid
;; syncing ~15 GB), and the in-memory LRU threshold is
;; low since each ML dataset is ~48 MB.

(pocket/config)

;; ---

;; ## Part 1 — Within-site holdout
;;
;; Start with the simplest case: one scenario, random holdout
;; split, and compare XGBoost configurations using
;; `evaluate-pipelines`.

;; ### Preparing the data
;;
;; Pocket caches the expensive preprocessing. The first call
;; takes minutes; subsequent calls return instantly.

(def example-ml-data
  (learning/get-ml-data {:site :A :year 2018
                         :species bacteria/E-coli
                         :antibiotic :Cefepime}))

(when example-ml-data
  {:rows (tc/row-count example-ml-data)
   :features (dec (count (tc/column-names example-ml-data)))})

;; ### Comparing XGBoost configurations
;;
;; Three [XGBoost](https://xgboost.readthedocs.io/)
;; configurations varying the number of boosting rounds:

(def xgboost-round-values [10 50 100])

(def holdout-splits
  (when example-ml-data
    (tc/split->seq example-ml-data :holdout {:seed 1})))

(def eval-results
  (when holdout-splits
    (ml/evaluate-pipelines
     (mapv #(learning/make-pipeline {:rounds %}) xgboost-round-values)
     holdout-splits
     loss/classification-accuracy
     :accuracy
     {:return-best-pipeline-only false
      :return-best-crossvalidation-only false})))

;; ### Results by boosting rounds

(defn rocauc-from-eval-result
  "Extract ROCAUC from an evaluate-pipelines result entry."
  [eval-entry test-actuals]
  (let [prob-dist (get-in eval-entry [:test-transform :probability-distribution])]
    (when prob-dist
      (learning/compute-rocauc test-actuals (get prob-dist 1)))))

(def within-site-metrics
  (when eval-results
    (let [test-actuals (:ri (:test (first holdout-splits)))]
      (->> eval-results
           (map-indexed
            (fn [i pipe-results]
              (let [r (first pipe-results)]
                {:rounds (nth xgboost-round-values i)
                 :accuracy (get-in r [:test-transform :metric])
                 :ROCAUC (rocauc-from-eval-result r test-actuals)})))
           tc/dataset))))

within-site-metrics

(kind/test-last
 #(and (= 3 (tc/row-count %))
       (every? pos? (:accuracy %))))

;; ### Diagnostic curves (within-site, 50 rounds)
;;
;; The curves below visualize the 50-round classifier
;; in more detail than a single ROCAUC number.

(def within-site-plot-data
  (when eval-results
    (let [test-actuals (:ri (:test (first holdout-splits)))
          ;; 50-round pipeline is index 1
          r (first (nth eval-results 1))
          prob-dist (get-in r [:test-transform :probability-distribution])]
      (when prob-dist
        {:actuals (vec test-actuals)
         :probabilities (vec (get prob-dist 1))}))))

(when within-site-plot-data
  (viz/diagnostic-plots (:actuals within-site-plot-data)
                        (:probabilities within-site-plot-data)))

;; Release large Phase 1 intermediates (each ~48 MB)
;; so they can be garbage-collected before the sweeps.
(def example-ml-data nil)
(def holdout-splits nil)
(def eval-results nil)

;; ---

;; ## Part 2 — Cross-year experiment
;;
;; Does a model trained on 2017 data generalize to 2018?
;; Same species, same antibiotic, same hospital — different year.
;;
;; We compose two independently cached datasets as train
;; and test, instead of splitting one dataset randomly.

(def cross-year-result
  (let [train-data (learning/get-ml-data {:site :A :year 2017
                                          :species bacteria/E-coli
                                          :antibiotic :Cefepime})
        test-data (learning/get-ml-data {:site :A :year 2018
                                         :species bacteria/E-coli
                                         :antibiotic :Cefepime})]
    (when (and train-data test-data)
      (let [pipe (learning/make-pipeline {:rounds 50})
            fitted (mm/fit-pipe train-data pipe)
            predicted (mm/transform-pipe test-data pipe fitted)
            pred-ds (:metamorph/data predicted)
            test-actuals (:ri test-data)
            prob-col (get pred-ds 1)]
        {:n-train (tc/row-count train-data)
         :n-test (tc/row-count test-data)
         :ROCAUC (learning/compute-rocauc test-actuals prob-col)
         :actuals (vec test-actuals)
         :probabilities (vec prob-col)}))))

(when cross-year-result
  (select-keys cross-year-result [:n-train :n-test :ROCAUC]))

;; ### Diagnostic curves (cross-year)

(when cross-year-result
  (viz/diagnostic-plots (:actuals cross-year-result)
                        (:probabilities cross-year-result)))

;; ### Within-year vs cross-year

(let [within (some-> within-site-metrics
                     (tc/select-rows #(= 50 (:rounds %)))
                     (tc/rows :as-maps)
                     first)]
  (tc/dataset
   (cond-> []
     within
     (conj {:experiment "within-year holdout (A 2018)"
            :ROCAUC (:ROCAUC within)})
     (:ROCAUC cross-year-result)
     (conj {:experiment "cross-year (A 2017→2018)"
            :ROCAUC (:ROCAUC cross-year-result)}))))

;; ---

;; ## Part 3 — Cross-site experiment
;;
;; Does a model trained at hospital A generalize to hospital C?
;; Same species, same antibiotic, same year — different hospital.
;;
;; Not every site has data for every combination.
;; The runner reports `:status` so we can trace why.

(def cross-site-result
  (let [train-data (learning/get-ml-data {:site :A :year 2018
                                          :species bacteria/E-coli
                                          :antibiotic :Cefepime})
        test-data (learning/get-ml-data {:site :C :year 2018
                                         :species bacteria/E-coli
                                         :antibiotic :Cefepime})]
    (when (and train-data test-data)
      (let [pipe (learning/make-pipeline {:rounds 50})
            fitted (mm/fit-pipe train-data pipe)
            predicted (mm/transform-pipe test-data pipe fitted)
            pred-ds (:metamorph/data predicted)
            test-actuals (:ri test-data)
            prob-col (get pred-ds 1)]
        {:n-train (tc/row-count train-data)
         :n-test (tc/row-count test-data)
         :ROCAUC (learning/compute-rocauc test-actuals prob-col)
         :actuals (vec test-actuals)
         :probabilities (vec prob-col)}))))

(when cross-site-result
  (select-keys cross-site-result [:n-train :n-test :ROCAUC]))

;; ### Diagnostic curves (cross-site)

(when cross-site-result
  (viz/diagnostic-plots (:actuals cross-site-result)
                        (:probabilities cross-site-result)))

;; ### All three experiment types compared

(let [within (some-> within-site-metrics
                     (tc/select-rows #(= 50 (:rounds %)))
                     (tc/rows :as-maps)
                     first)]
  (tc/dataset
   (cond-> []
     within
     (conj {:experiment "within-site (A 2018)"
            :ROCAUC (:ROCAUC within)})
     (:ROCAUC cross-year-result)
     (conj {:experiment "cross-year (A 2017→2018)"
            :ROCAUC (:ROCAUC cross-year-result)})
     (:ROCAUC cross-site-result)
     (conj {:experiment "cross-site (A→C 2018)"
            :ROCAUC (:ROCAUC cross-site-result)}))))

;; Note: the within-site experiment uses a holdout split,
;; while cross-year and cross-site use the full source
;; dataset for training.

;; ---

;; ## Part 4 — Sweeping across species and antibiotics
;;
;; We scale up to all three species in
;; [Weis et al.](https://doi.org/10.1038/s41591-021-01619-9),
;; running three experiment types for each species–antibiotic
;; combination. Every result carries a `:status` so we can
;; distinguish successful experiments from insufficient data.

;; ### Helper: sweep one species

(defn sweep-species
  "Run within-site, cross-year, and cross-site experiments
  for all antibiotics of a given species. Returns a dataset."
  [species]
  (let [antibiotics (get bacteria/species->antibiotics species)]
    (->> (concat
          (mapv (fn [ab]
                  (scenarios/run-within-site {:species species
                                              :antibiotic ab
                                              :site :A :year 2018}))
                antibiotics)
          (mapv (fn [ab]
                  (scenarios/run-cross-year {:species species
                                             :antibiotic ab
                                             :site :A
                                             :train-year 2017
                                             :test-year 2018}))
                antibiotics)
          (mapv (fn [ab]
                  (scenarios/run-cross-site {:species species
                                             :antibiotic ab
                                             :train-site :A
                                             :test-site :C
                                             :year 2018}))
                antibiotics))
         tc/dataset)))

;; ### Running the sweep

(def sweep-species-list
  bacteria/important-bacteria)

(def all-sweep-results
  (->> sweep-species-list
       (mapv sweep-species)
       (apply tc/concat)))

{:rows (tc/row-count all-sweep-results)
 :columns (count (tc/column-names all-sweep-results))}

;; ### Status summary

(-> all-sweep-results
    (tc/group-by [:species :experiment :status])
    (tc/aggregate {:count tc/row-count})
    (tc/order-by [:species :experiment :status]))

;; ### Per-species results
;;
;; The `:ok` rows contain valid ROCAUC values. Other
;; statuses indicate data limitations (too few samples,
;; no data at the test site, or single-class test sets).

(defn- plots-per-species
  "Given a dataset with :species, :antibiotic, and a y-col,
  produce a kind/fragment of separate plots per species
  with consistent y-axis [0,1] and a shared x-axis
  category order within each species."
  ([ds color-col title-prefix]
   (plots-per-species ds color-col title-prefix :ROCAUC))
  ([ds color-col title-prefix y-col]
   (->> (distinct (:species ds))
        (mapv (fn [species]
                (let [sub (tc/select-rows ds #(= species (:species %)))
                      antibiotics (vec (distinct (:antibiotic sub)))]
                  (-> sub
                      (plotly/base {:=x :antibiotic
                                    :=y y-col
                                    :=color color-col
                                    :=title (str title-prefix " — " species)})
                      (plotly/layer-point {:=mark-size 10
                                           :=mark-opacity 0.7})
                      plotly/plot
                      (assoc-in [:layout :yaxis :range] [0 1])
                      (assoc-in [:layout :xaxis :tickangle] -45)
                      (assoc-in [:layout :xaxis :categoryorder] "array")
                      (assoc-in [:layout :xaxis :categoryarray] antibiotics)))))
        kind/fragment)))

(def ok-results
  (-> all-sweep-results
      (tc/select-rows #(= :ok (:status %)))
      (tc/select-columns [:species :antibiotic :experiment :ROCAUC])
      (tc/map-columns :experiment [:experiment] name)))

ok-results

;; ### Mean ROCAUC by species and experiment type

(def species-experiment-summary
  (-> ok-results
      (tc/group-by [:species :experiment])
      (tc/aggregate {:mean-ROCAUC #(-> % :ROCAUC tcc/mean)
                     :count tc/row-count})
      (tc/order-by [:species :experiment])))

species-experiment-summary

(kind/test-last
 #(pos? (tc/row-count %)))

;; ### Summary bar chart

(-> species-experiment-summary
    (plotly/base {:=x :species
                  :=y :mean-ROCAUC
                  :=color :experiment
                  :=title "Mean ROCAUC by species and experiment type"})
    (plotly/layer-bar {:=mark-opacity 0.8})
    plotly/plot)

;; ### All individual results

(plots-per-species ok-results :experiment "ROCAUC")

;; ---

;; ## Part 5 — Bidirectional cross-site transfer
;;
;; Part 4 tested A→C transfer. Here we add the
;; reverse direction (C→A) to see whether transfer
;; performance is symmetric.
;;
;; Sites B and D have no AMR labels for the species
;; in this study, so cross-site experiments are limited
;; to the A↔C pair. Similarly, cross-year experiments
;; are limited to 2017→2018 at site A (2015 and 2016
;; have metadata that doesn't join with spectra files).

;; ### Helper: reverse cross-site sweep

(defn sweep-cross-site-reverse
  "Run C→A cross-site experiments for all antibiotics
  of a given species. Returns a dataset."
  [species]
  (let [antibiotics (get bacteria/species->antibiotics species)]
    (->> antibiotics
         (mapv (fn [ab]
                 (scenarios/run-cross-site {:species species
                                            :antibiotic ab
                                            :train-site :C
                                            :test-site :A
                                            :year 2018})))
         tc/dataset)))

;; ### Running the reverse sweep

(def all-reverse-results
  (->> sweep-species-list
       (mapv sweep-cross-site-reverse)
       (apply tc/concat)))

(-> all-reverse-results
    (tc/select-columns [:species :antibiotic :status :n-train :n-test :ROCAUC]))

;; ### Bidirectional comparison
;;
;; Combining A→C (from Part 4) and C→A results:

(defn- tag-direction [ds direction-label]
  (-> ds
      (tc/select-rows #(= :ok (:status %)))
      (tc/select-columns [:species :antibiotic :ROCAUC])
      (tc/add-column :direction direction-label)))

(def bidirectional-results
  (let [forward (-> all-sweep-results
                    (tc/select-rows #(= :cross-site (:experiment %))))]
    (tc/concat
     (tag-direction forward "A → C")
     (tag-direction all-reverse-results "C → A"))))

bidirectional-results

;; ### Mean ROCAUC by direction and species

(-> bidirectional-results
    (tc/group-by [:species :direction])
    (tc/aggregate {:mean-ROCAUC #(-> % :ROCAUC tcc/mean)
                   :count tc/row-count})
    (tc/order-by [:species :direction]))

;; ### Bidirectional comparison plot

(plots-per-species bidirectional-results :direction "Cross-site ROCAUC")

;; ---

;; ## Part 6 — K-fold cross-validation
;;
;; Parts 1–4 use single holdout splits. Here we use
;; [5-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
;; to get more robust ROCAUC estimates with standard
;; deviations.

;; ### Helper: one CV fold

(defn- cv-fold-result
  "Run one fold: fit on train, predict on test, compute ROCAUC."
  [pipe train-ds test-ds]
  (try
    (let [fitted (mm/fit-pipe train-ds pipe)
          predicted (mm/transform-pipe test-ds pipe fitted)
          pred-ds (:metamorph/data predicted)
          test-actuals (:ri test-ds)
          prob-col (get pred-ds 1)]
      {:ROCAUC (learning/compute-rocauc test-actuals prob-col)
       :n-test (tc/row-count test-ds)})
    (catch Exception _e nil)))

;; ### Helper: k-fold CV for one scenario

(defn run-kfold-cv
  "Run k-fold cross-validation for one species–antibiotic
  combination. Returns a map with :mean-ROCAUC and :std-ROCAUC,
  or nil if data is insufficient."
  [{:keys [species antibiotic site year k rounds]
    :or {k 5 rounds 50}}]
  (let [ml-data (learning/get-ml-data {:site site :year year
                                       :species species
                                       :antibiotic antibiotic})]
    (when (and ml-data (>= (tc/row-count ml-data) 100))
      (let [splits (tc/split->seq ml-data :kfold {:k k :seed 1})
            pipe (learning/make-pipeline {:rounds rounds})
            rocaucs (->> splits
                         (keep (fn [split]
                                 (:ROCAUC (cv-fold-result pipe
                                                          (:train split)
                                                          (:test split)))))
                         vec)]
        (when (> (count rocaucs) 1)
          {:species species
           :antibiotic antibiotic
           :n-folds (count rocaucs)
           :mean-ROCAUC (tcc/mean rocaucs)
           :std-ROCAUC (tcc/standard-deviation rocaucs)})))))

;; ### Running the CV sweep

(def cv-results
  (->> sweep-species-list
       (mapcat (fn [species]
                 (->> (get bacteria/species->antibiotics species)
                      (keep (fn [ab]
                              (run-kfold-cv {:species species
                                             :antibiotic ab
                                             :site :A :year 2018}))))))
       tc/dataset))

(-> cv-results
    (tc/select-columns [:species :antibiotic :n-folds :mean-ROCAUC :std-ROCAUC]))

;; ### CV summary by species

(-> cv-results
    (tc/group-by [:species])
    (tc/aggregate {:mean-ROCAUC #(-> % :mean-ROCAUC tcc/mean)
                   :mean-std #(-> % :std-ROCAUC tcc/mean)
                   :count tc/row-count})
    (tc/order-by [:species]))

;; ### CV results plot

(plots-per-species cv-results :species "5-fold CV ROCAUC" :mean-ROCAUC)

;; ---

;; ## Part 7 — Within-site at site C
;;
;; Site C is the second-largest
;; [DRIAMS](https://doi.org/10.1038/s41591-021-01619-9)
;; site with AMR labels. Running within-site holdout
;; experiments here lets us compare classifier performance
;; across hospitals using the same methodology.

;; ### Helper: within-site sweep at site C

(defn sweep-within-site-C
  "Run within-site holdout experiments at site C (2018)
  for all antibiotics of a given species. Returns a dataset."
  [species]
  (let [antibiotics (get bacteria/species->antibiotics species)]
    (->> antibiotics
         (mapv (fn [ab]
                 (scenarios/run-within-site {:species species
                                             :antibiotic ab
                                             :site :C :year 2018})))
         tc/dataset)))

;; ### Running the site C sweep

(def site-C-results
  (->> sweep-species-list
       (mapv sweep-within-site-C)
       (apply tc/concat)))

(-> site-C-results
    (tc/select-columns [:species :antibiotic :status :n-train :n-test :ROCAUC]))

;; ### Site A vs site C within-site comparison
;;
;; Combining the within-site results from Part 4 (site A)
;; and site C to see whether both hospitals yield
;; comparable classifier performance.

(defn- tag-site [ds site-label]
  (-> ds
      (tc/select-rows #(= :ok (:status %)))
      (tc/select-columns [:species :antibiotic :ROCAUC])
      (tc/add-column :site site-label)))

(def site-comparison
  (let [site-A-within (-> all-sweep-results
                          (tc/select-rows #(= :within-site (:experiment %))))]
    (tc/concat
     (tag-site site-A-within "A")
     (tag-site site-C-results "C"))))

site-comparison

;; ### Mean ROCAUC by site and species

(-> site-comparison
    (tc/group-by [:species :site])
    (tc/aggregate {:mean-ROCAUC #(-> % :ROCAUC tcc/mean)
                   :count tc/row-count})
    (tc/order-by [:species :site]))

;; ### Site comparison plot

(plots-per-species site-comparison :site "Within-site ROCAUC")
;; ---

;; ## Part 8 — Comprehensive comparison
;;
;; Combining all experiment types into one view:
;; within-site at both sites, cross-year, and
;; bidirectional cross-site.

(def comprehensive-results
  (tc/concat
   (-> all-sweep-results
       (tc/select-rows #(= :ok (:status %)))
       (tc/select-columns [:species :antibiotic :experiment :ROCAUC])
       (tc/map-columns :label [:experiment]
                       #(case %
                          :within-site "within-site A"
                          :cross-year "cross-year 2017→2018"
                          :cross-site "cross-site A→C"
                          (name %))))
   (-> site-C-results
       (tc/select-rows #(= :ok (:status %)))
       (tc/select-columns [:species :antibiotic :ROCAUC])
       (tc/add-column :experiment :within-site-C)
       (tc/add-column :label "within-site C"))
   (-> all-reverse-results
       (tc/select-rows #(= :ok (:status %)))
       (tc/select-columns [:species :antibiotic :ROCAUC])
       (tc/add-column :experiment :cross-site-reverse)
       (tc/add-column :label "cross-site C→A"))))

{:rows (tc/row-count comprehensive-results)
 :experiment-types (distinct (:label comprehensive-results))}

;; ### Mean ROCAUC by experiment type

(def comprehensive-summary
  (-> comprehensive-results
      (tc/group-by [:label])
      (tc/aggregate {:mean-ROCAUC #(-> % :ROCAUC tcc/mean)
                     :count tc/row-count})
      (tc/order-by [:mean-ROCAUC] :desc)))

comprehensive-summary

;; ### Mean ROCAUC by experiment type and species

(-> comprehensive-results
    (tc/group-by [:species :label])
    (tc/aggregate {:mean-ROCAUC #(-> % :ROCAUC tcc/mean)
                   :count tc/row-count})
    (tc/order-by [:species :mean-ROCAUC] [:asc :desc]))

;; ### Comprehensive bar chart

(-> comprehensive-results
    (tc/group-by [:species :label])
    (tc/aggregate {:mean-ROCAUC #(-> % :ROCAUC tcc/mean)})
    (plotly/base {:=x :species
                  :=y :mean-ROCAUC
                  :=color :label
                  :=title "Mean ROCAUC across all experiment types"})
    (plotly/layer-bar {:=mark-opacity 0.8})
    plotly/plot)

;; ### All individual results

(plots-per-species comprehensive-results :label "ROCAUC")
