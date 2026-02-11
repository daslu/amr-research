;; # Exploring the DRIAMS Dataset
;;
;; The [DRIAMS dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q)
;; is a large collection of [MALDI-TOF](https://en.wikipedia.org/wiki/Matrix-assisted_laser_desorption/ionization)
;; mass spectra linked to [antimicrobial resistance](https://en.wikipedia.org/wiki/Antimicrobial_resistance) profiles,
;; published by [Weis et al. (2022)](https://doi.org/10.1038/s41591-021-01619-9).
;;
;; It contains spectra from four Swiss hospital sites:
;;
;; - **DRIAMS-A** — University Hospital of Basel (2015–2018, ~80K spectra)
;; - **DRIAMS-B** — Canton Hospital Basel-Land (2018, ~6K spectra)
;; - **DRIAMS-C** — Canton Hospital Aarau (2018, ~22K spectra)
;; - **DRIAMS-D** — Viollier AG laboratory (2018, ~76K spectra)
;;
;; This notebook walks through the data using AMR's ingestion
;; utilities, then shows how Ripple preprocesses a spectrum
;; for machine learning.

(ns amr-book.exploring-driams
  (:require
   ;; AMR data loading utilities:
   [scicloj.amr.data.ingestion :as ingestion]
   ;; Table processing (https://scicloj.github.io/tablecloth/):
   [tablecloth.api :as tc]
   ;; Interactive plotting via Plotly (https://scicloj.github.io/tableplot/):
   [scicloj.tableplot.v1.plotly :as plotly]))

;; ## Where is the data?
;;
;; The `ingestion/base-dir` function resolves the DRIAMS path
;; from the `DRIAMS_BASE_DIR` environment variable (or `amr.edn`):

(ingestion/base-dir)

;; ## Loading a raw spectrum
;;
;; Each spectrum is a gzipped text file with space-separated
;; mass/intensity pairs — typically ~18,000 points spanning
;; roughly 2,000–20,000 Da.
;;
;; Let's pick the first file returned by the ingestion utilities:

(def spectrum-path
  (-> (ingestion/find-data-files "txt.gz")
      first))

spectrum-path

;; Load it as a tablecloth dataset:

(def raw-spectrum
  (ingestion/load-raw-spectrum spectrum-path))

raw-spectrum

;; Basic statistics:

{:rows (tc/row-count raw-spectrum)
 :mass-min (-> raw-spectrum :mass first)
 :mass-max (-> raw-spectrum :mass last)}

;; ### Visualizing the raw spectrum

(-> raw-spectrum
    (plotly/base {:=x :mass
                  :=y :intensity
                  :=title "Raw MALDI-TOF spectrum"
                  :=x-title "m/z (Da)"
                  :=y-title "Intensity (a.u.)"})
    (plotly/layer-line)
    plotly/plot)

;; ## Metadata
;;
;; The `id/` directory contains species identification and
;; antimicrobial resistance labels (R/S/I) for each spectrum.

(def metadata
  (ingestion/load-metadata {:site :A :year 2018}))

{:spectra (tc/row-count metadata)
 :columns (count (tc/column-names metadata))}

;; A few rows (showing selected antibiotic columns):

(-> metadata
    (tc/select-columns [:code :species :Cefepime :Ciprofloxacin :Ceftriaxone]))

;; ### Species distribution
;;
;; How many distinct species are there, and which are most common?

(def species-counts
  (-> metadata
      (tc/group-by [:species])
      (tc/aggregate {:count tc/row-count})
      (tc/order-by [:count] :desc)))

(tc/select-rows species-counts (range 10))

;; The top species as a bar chart:

(-> species-counts
    (tc/select-rows (range 15))
    (plotly/base {:=x :count
                  :=y :species
                  :=title "Top 15 species — DRIAMS-A 2018"
                  :=x-title "Number of spectra"
                  :=y-title ""})
    (plotly/layer-bar)
    plotly/plot
    (assoc-in [:data 0 :orientation] :h)
    (assoc-in [:layout :margin :l] 200))

;; ### Antibiotics
;;
;; The metadata has one column per antibiotic, with values
;; "R" (resistant), "S" (susceptible), or "I" (intermediate).
;; Which antibiotics are tested?

(def antibiotic-columns
  (->> (tc/column-names metadata)
       (remove #{:code :species :laboratory_species
                 :combined_code :column-0 (keyword "Unnamed: 0")})))

(count antibiotic-columns)

;; ## References
;;
;; - Weis, C., et al. (2022).
;;   [Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning](https://doi.org/10.1038/s41591-021-01619-9).
;;   *Nature Medicine*, 28, 164–174.
