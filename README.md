# Monitoring Forest Recovery in Northeast China: Adapting an Attention U-Net Beyond the Amazon

COMP0173 Coursework 2 (AI for Sustainable Development). This repo replicates the original
paper below, then adapts it to a new context — monitoring post-logging-ban forest recovery in
Heilongjiang, China — while keeping the same core methodology.

**Original paper**: David John & Ce Zhang (2022), "An attention-based U-Net for detecting
deforestation within satellite sensor imagery," *International Journal of Applied Earth
Observation and Geoinformation* 107.
DOI: [10.1016/j.jag.2022.102685](https://doi.org/10.1016/j.jag.2022.102685)
(open access: https://nora.nerc.ac.uk/id/eprint/532301/1/N532301JA.pdf)

The repo is split into two top-level folders:

- **`replication/`** — reproduces the original paper on its own Amazon/Atlantic Forest data.
- **`experiment/`** — adapts the same architecture to Heilongjiang, China, plus every
  ablation/sub-experiment run on top of it.

---

## `replication/`

| File | What it does |
|---|---|
| `code_replicate.ipynb` | Full pipeline: ingestion, augmentation, Attention U-Net definition, training, evaluation against `metrics/metrics_3d.csv`. **Reproduced the original paper's results within the required ±5% tolerance on every metric.** |
| `Experimentation.ipynb` | Original data processing / augmentation / training / testing notebook (3-band and 4-band Amazon + Atlantic Forest variants, plus the 4 baseline models compared against Attention U-Net). |
| `Figures.ipynb` | Generates the figures in `figures/` (band comparisons, loss/accuracy curves, the Amazon/Atlantic Forest location map). |
| `predictor.py` | Takes an RGB or 4-band image and outputs an Attention U-Net-predicted deforestation mask. |
| `preprocess-4band-amazon-data.py` / `preprocess-4band-atlantic-forest-data.py` / `preprocess-rgb-data.py` | Preprocess the respective GeoTIFF datasets into numpy pickles. |
| `metrics/` | Accuracy/precision/recall/F1 for every replicated model/variant. |
| `models/` | Trained `.hdf5` weights — load with `keras.models.load_model([path])`. |
| `figures/` | Reference figures and the two shapefiles actually used (Amazon, Atlantic Forest / Mata Atlântica) for the location map. |

**Datasets**: Amazon 1 (regular 3-band) — https://zenodo.org/record/3233081 · Amazon 2 (larger
4-band Amazon + Atlantic Forest) — https://zenodo.org/record/4498086#.YMh3GfKSmCU

---

## `experiment/`

### Main pipeline (Heilongjiang, June–Aug 2025)

| File | What it does |
|---|---|
| `preprocess-heilongjiang-sentinel2-data.py` | Reads B04/B03/B02/B08 from 14 raw Sentinel-2 L2A scenes, applies the ESA reflectance correction, computes NDVI, tiles to 512×512, and filters out tiles with >10% cloud/shadow/snow/nodata (SCL band). 4,721 of 6,174 candidate tiles kept (76%). |
| `generate-heilongjiang-ground-truth.py` | Labels forest via NDVI persistence across all available dates for a tile position (not a single-date snapshot), with a morphological filter that strips out thin farmland shelterbelt strands so they aren't misread as forest. |
| `code_phase2_heilongjiang_unfiltered.ipynb` / `code_phase2_heilongjiang_filtered.ipynb` | Train on the unfiltered vs. shelterbelt-filtered ground truth. **Filtered is the headline/adopted model** (accuracy 0.909 vs. 0.889 unfiltered, p=5.4×10⁻¹⁰, paired Wilcoxon). |
| `code_phase2_heilongjiang_comparison.ipynb` | Loads both trained models, evaluates each on its own validation set, runs the paired significance test above. |
| `code_phase2_heilongjiang.ipynb` | Earlier single-run version of the phase-2 pipeline, kept for reference. |

### Sub-experiments / ablations

All of these are disclosed tests, not all adopted into the headline model — see each row for the
actual result.

| File | What it tests | Result |
|---|---|---|
| `code_phase2_heilongjiang_filtered_archfix.ipynb` | Fixes a genuine bug (the finest-resolution attention gate was silently discarded by an overwrite one line later) present in every notebook in this repo, including the Amazon replication. | Small but statistically significant **decrease** (p=7.1×10⁻⁴) — a disclosed negative result, **not adopted**. |
| `generate-heilongjiang-multitemporal-ground-truth.py` + `code_phase2_heilongjiang_multitemporal.ipynb` | Stacks two acquisition dates' bands as an 8-channel input (4ch→8ch), testing whether temporal context helps — something the original paper's single-date Amazon data never had. | Accuracy 0.988, but the input shares two bands (R, NIR) with the NDVI-persistence label itself — **likely label circularity, not a genuine improvement. Not adopted.** |
| `generate-heilongjiang-sepnov-ground-truth.py` + `preprocess-heilongjiang-sepnov-sentinel2-data.py` + `code_phase2_heilongjiang_sepnov.ipynb` + `code_phase2_heilongjiang_comparison_sepnov.ipynb` | Repeats the pipeline on a Sept–Nov (autumn) acquisition window, testing whether crop senescence makes the forest/cropland NDVI distinction easier. | Accuracy 0.980, significantly higher than the June–Aug filtered model (p=1.2×10⁻³¹, unpaired Mann-Whitney) — attributed to an easier seasonal signal, not a better model. |
| `code_phase2_heilongjiang_filterbase_sweep.ipynb` | Reruns the filtered-dataset pipeline at `filter_base` = 16/24/32, since the original choice (24) was reasoned, not systematically searched. | See notebook output — this was a disclosed hyperparameter-tuning gap being closed. |
| `download-crossregion-sentinel2-data.py` + `package-crossregion-scene.py` + `code_phase2_heilongjiang_crossregion.ipynb` | Applies the trained Heilongjiang model, with no retraining, to two genuinely unseen regions (Shangri-La, Yunnan; Horqin Sandy Land, Liaoning). | Visual plausibility check only — no ground truth exists for these tiles. |
| `download-heilongjiang-sepnov-sentinel2-data.py` + `.ipynb` | Downloads the Sept–Nov Sentinel-2 scenes used by the sepnov experiment above. | — |
| `package-heilongjiang-unlabeled-scene.py` | Packages tile T52TDT (excluded from every ground-truth script — only 1 acquisition date, can't support the persistence check) for a purely visual "apply the model to unlabeled data" sanity check. | — |

**Fairness check** (run inside `code_phase2_heilongjiang_comparison.ipynb` /
`code_phase2_heilongjiang_comparison_sepnov.ipynb`): a Kruskal-Wallis test on per-image accuracy
across tile locations found statistically significant differences (p<10⁻¹²) — farmland-dominated
tiles score lower than forest-dominated ones, consistent with the shelterbelt/cropland confusion
finding above. Disclosed, not resolved by filtering.

**Data provenance**: Sentinel-2 L2A imagery via the Copernicus Data Space Ecosystem — free,
open-access, no cost, no personal/identifiable data (10m-resolution land imagery only).

---

## Setup

```
pip install -r requirements.txt
```

### Obtaining Attention U-Net deforestation masks (replication models)
- Download `unet-attention-3d.hdf5`, `unet-attention-4d.hdf5`, `unet-attention-4d-atlantic.hdf5`
  into `replication/models/` (already there if you cloned this repo).
- Run (from repo root): `python3 replication/predictor.py [MODEL ID] [INPUT IMAGE PATH]`
  - Model ID: `1` = RGB, `2` = 4-band Amazon-trained, `3` = 4-band Atlantic Forest-trained.
  - e.g. `python3 replication/predictor.py 2 test.tif`

### Obtaining pre-processed data (replication)
Run from repo root: `python3 replication/preprocess-4band-amazon-data.py`,
`python3 replication/preprocess-4band-atlantic-forest-data.py`, or
`python3 replication/preprocess-rgb-data.py`.

### Heilongjiang pipeline
The `experiment/` notebooks are designed to run on Google Colab (they mount Drive and clone this
repo automatically) — open via GitHub from Colab directly. The `.py` scripts can be run locally
from the repo root, e.g. `python3 experiment/preprocess-heilongjiang-sentinel2-data.py` — see the
docstring at the top of each script for its exact data dependencies.
