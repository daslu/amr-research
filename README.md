# scicloj.amr

[Antimicrobial resistance](https://en.wikipedia.org/wiki/Antimicrobial_resistance) (AMR) prediction from [MALDI-TOF](https://en.wikipedia.org/wiki/Matrix-assisted_laser_desorption/ionization) mass spectra, reproducing the approach of [Weis et al. (2022)](https://doi.org/10.1038/s41591-021-01619-9) using the [DRIAMS dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q).

This project provides well-explained notebooks that walk through the full pipeline — from raw spectra to [XGBoost](https://xgboost.readthedocs.io/) classification — making it convenient for researchers to explore, reproduce, and extend this line of work in [Clojure](https://clojure.org/).

## Setup

### 1. Get the DRIAMS dataset

Download from the [Dryad repository](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q) and extract so site folders are under a single directory. You may remove the `preprocessing` subdirectories:

```
/path/to/DRIAMS/
├── DRIAMS-A/
│   ├── raw/2015/  raw/2016/  raw/2017/  raw/2018/
│   └── id/2015/   id/2016/   id/2017/   id/2018/
├── DRIAMS-B/ ...
├── DRIAMS-C/ ...
└── DRIAMS-D/ ...
```

We recommend gzipping the raw `.txt` files to save space (~60 GB → ~15 GB). [tablecloth](https://scicloj.github.io/tablecloth/) reads `.txt.gz` transparently.

### 2. Configure the data path

Either set the environment variable:

```bash
export DRIAMS_BASE_DIR=/path/to/DRIAMS/
```

Or edit `amr.edn` in the project root:

```edn
{:base-dir "/path/to/DRIAMS/"}
```

### 3. Render the notebooks

From the REPL:

```clojure
(require '[dev])
(dev/make-book!)
```

## Key Libraries

This project builds on the [Scicloj](https://scicloj.github.io/) ecosystem:

- **[tablecloth](https://scicloj.github.io/tablecloth/)** — dataframe library for tabular data manipulation (built on [tech.ml.dataset](https://github.com/techascent/tech.ml.dataset) and [dtype-next](https://github.com/cnuernber/dtype-next))
- **[metamorph.ml](https://github.com/scicloj/metamorph.ml)** — machine learning pipelines ([XGBoost](https://xgboost.readthedocs.io/) classification in this project)
- **[tableplot](https://scicloj.github.io/tableplot/)** — interactive plotting via [Plotly](https://plotly.com/javascript/)
- **[Ripple](https://clojurecivitas.github.io/ripple/)** (`scicloj.ripple.maldi`) — [MALDIquant](https://strimmerlab.github.io/software/maldiquant/)-compatible signal preprocessing, binning, and peak detection
- **[Pocket](https://github.com/scicloj/pocket)** (`scicloj.pocket`) — filesystem-based caching for expensive computations
- **[fastmath](https://github.com/generateme/fastmath)** — numerical and statistical functions
- **[Kindly](https://scicloj.github.io/kindly-noted/)** — annotation system for notebook visualizations (used with [Clay](https://scicloj.github.io/clay/) for rendering)

## References

- Weis, C., et al. (2022). [Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning](https://doi.org/10.1038/s41591-021-01619-9). *Nature Medicine*, 28, 164–174.

---

Part of the [scicloj](https://scicloj.github.io/) ecosystem for scientific computing in [Clojure](https://clojure.org/).
