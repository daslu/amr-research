# scicloj.amr

Antimicrobial resistance (AMR) prediction from MALDI-TOF mass spectra, reproducing the approach of [Weis et al. (2022)](https://doi.org/10.1038/s41591-021-01619-9) using the [DRIAMS dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q).

This project provides well-explained notebooks that walk through the full pipeline — from raw spectra to XGBoost classification — making it convenient for researchers to explore, reproduce, and extend this line of work in Clojure.

## Companion Libraries

Signal processing and caching live in standalone libraries (included as local symlinks):

- **[Ripple](https://github.com/scicloj/ripple)** (`scicloj.ripple.maldi`) — MALDIquant-compatible preprocessing, binning, and peak detection
- **[Pocket](https://github.com/scicloj/pocket)** (`scicloj.pocket`) — Filesystem-based caching for expensive computations

## Setup

### 1. Get the DRIAMS dataset

Download from the [Dryad repository](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q) and extract so site folders are under a single directory:

```
/path/to/DRIAMS/
├── DRIAMS-A/
│   ├── raw/2015/  raw/2016/  raw/2017/  raw/2018/
│   └── id/2015/   id/2016/   id/2017/   id/2018/
├── DRIAMS-B/ ...
├── DRIAMS-C/ ...
└── DRIAMS-D/ ...
```

We recommend gzipping the raw `.txt` files to save space (~60 GB → ~15 GB). tablecloth reads `.txt.gz` transparently.

### 2. Configure the data path

Either set the environment variable:

```bash
export DRIAMS_BASE_DIR=/path/to/DRIAMS/
```

Or edit `amr.edn` in the project root:

```edn
{:base-dir "/path/to/DRIAMS/"}
```

### 3. Start the REPL

```bash
clojure -M:nrepl-for-mcp
```

This starts an nREPL server on port 7100.

### 4. Render the notebooks

From the REPL:

```clojure
(require '[dev])
(dev/make-book!)
```

## References

- Weis, C., et al. (2022). [Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning](https://doi.org/10.1038/s41591-021-01619-9). *Nature Medicine*, 28, 164–174.

---

Part of the [scicloj](https://scicloj.github.io/) ecosystem for scientific computing in Clojure.
