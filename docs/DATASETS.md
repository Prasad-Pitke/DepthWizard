# Datasets

## GAMUS (recommended training/validation benchmark)

Per the problem statement's reference repo
([IMG-PROCESS-SAC/SIH-DepthWizard-2026](https://github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026)),
**GAMUS** is the recommended open dataset for training/validating the
monocular depth backbone and bridging the natural-image → remote-sensing
domain gap.

- Dataset: [earthflow/GAMUS on HuggingFace](https://huggingface.co/datasets/earthflow/GAMUS) (80GB, cc-by-4.0)
- Paper: Xiong et al. 2023, ["GAMUS: A Geometry-aware Multi-modal Semantic Segmentation Benchmark for Remote Sensing Data"](https://arxiv.org/abs/2305.14914)
- Content: 11,507 tiles (1024×1024) across 5 US cities (Washington DC,
  Philadelphia, Oklahoma, Jacksonville, NYC) — each tile is an RGB
  orthophoto paired with a LiDAR-derived **nDSM / AGL** (height Above
  Ground Level — building/vegetation height with bare-earth terrain
  already subtracted out) and a 6-class land-cover raster (ground,
  low-vegetation, building, water, road, tree).
- Split: 6,304 train / 1,059 val / 4,144 test tiles.

This is a strong fit for two of this project's three milestones:
**Elevation Extraction** (fine-tuning target) and **DSM Estimation
Accuracy validation** (real LiDAR ground truth, pre-stratified by
land-cover class that maps directly onto the urban/sparse/forested
categories this project's evaluation already reports — see
`docs/EVALUATION.md`). GAMUS has no hilly/mountainous terrain (it's
flat US-city LiDAR), so pair it with a second dataset (e.g. a
DFC19-style or SRTM-anchored scene) if you need hilly-terrain coverage
in your final evaluation.

### On-disk layout

Verified directly against the HuggingFace repo's file tree (not just
the dataset card, which doesn't document this):

```
<gamus_root>/
  images/{train,val,test}/<TILE_ID>_RGB.h5    # e.g. DC_01_25_RGB.h5
  heights/{train,val,test}/<TILE_ID>_AGL.h5   # e.g. DC_01_25_AGL.h5
  classes/{train,val,test}/<TILE_ID>_CLS.h5   # e.g. DC_01_25_CLS.h5
```

Each `.h5` file holds exactly one tile. `depthwizard/datasets/gamus.py`
auto-detects the internal HDF5 dataset key name (tries common names,
then falls back to "first dataset in the file") since that key wasn't
documented anywhere inspectable without downloading the full 80GB
dataset — pass an explicit key if your copy differs (see the module
docstring).

### Downloading

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='earthflow/GAMUS', repo_type='dataset', local_dir='./gamus_data')
"
```

To pull just a slice for fast iteration (e.g. one city's train split):

```python
snapshot_download(repo_id="earthflow/GAMUS", repo_type="dataset",
                   local_dir="./gamus_data",
                   allow_patterns=["images/train/DC_*", "heights/train/DC_*", "classes/train/DC_*"])
```

### Validating a backbone against GAMUS

```bash
cd backend
pip install h5py huggingface_hub
python scripts/validate_on_gamus.py \
    --gamus-root ./gamus_data --split test \
    --backbone classical --n-tiles 50 --n-gcps 40 \
    --output gamus_validation_report.json
```

This samples `--n-gcps` random pixels per tile as simulated survey
points, calibrates on those (via the same `calibrate_with_gcps` path
the main pipeline uses), and reports RMSE/MAE/correlation on the
**held-out remaining pixels** — not on the calibration points
themselves, which would be circular. Results are aggregated overall
and per land-cover class (mapped to urban/sparse/forested via
`gamus.GAMUS_TO_LANDSCAPE`). This script was built and tested against
a synthetic fixture matching GAMUS's exact file layout in this
project's test environment (no GPU / no HuggingFace access there); run
it against the real dataset before citing its numbers.

### Fine-tuning a backbone on GAMUS

```bash
pip install torch transformers h5py huggingface_hub
python scripts/finetune_on_gamus.py \
    --gamus-root ./gamus_data \
    --base-model Intel/dpt-hybrid-midas \
    --epochs 5 --batch-size 4 --lr 1e-5 \
    --output-dir ./checkpoints/dpt-hybrid-gamus
```

Fine-tunes from a pretrained DPT/MiDaS checkpoint using a scale-invariant
log loss against GAMUS's AGL ground truth. **This script requires
`torch`+`transformers` and a GPU for practical training time; it was
written and structurally checked (arg parsing, data flow, checkpoint
I/O) but the actual training loop was not run in this project's build
environment** (no GPU, no `download.pytorch.org`/`huggingface.co`
access there — see `README.md`). Run it on your own GPU host, then
point `DepthWizardPipeline` at the resulting checkpoint — the exact
load-back snippet is printed at the end of a training run, and looks
like:

```python
from transformers import pipeline
from PIL import Image
import numpy as np
from depthwizard.pipeline import DepthWizardPipeline

hf_pipe = pipeline("depth-estimation", model="./checkpoints/dpt-hybrid-gamus/epoch_5")

def custom_depth_fn(rgb: np.ndarray) -> np.ndarray:
    return np.array(hf_pipe(Image.fromarray(rgb))["depth"], dtype=np.float32)

pipeline = DepthWizardPipeline(backbone="auto", backbone_kwargs={"custom_model": custom_depth_fn})
```

The `custom_model=` path in `depth_model.py` is deliberately
implemented to need **no torch import at inference time** — once
you've fine-tuned a model, wrapping it as a plain `ndarray -> ndarray`
function lets you deploy the calibration/DSM/mesh/viewer stack on a
machine that doesn't have (or can't run) torch at all, only the
training step needs a GPU.

## ISRO RGB-band optical satellite imagery (final evaluation target)

The pipeline already ingests any RGB GeoTIFF/PNG/JPG generically (see
`geo_io.py`), which should cover standard ISRO optical products
(e.g. Cartosat-2/3, ResourceSat-2 LISS-III/IV) without code changes,
but a few things are worth checking once you have a real sample:

- **Radiometric depth**: ISRO optical products are commonly delivered
  at 10-12 bit (not 8-bit). `geo_io._to_uint8()` already applies a
  2nd/98th-percentile contrast stretch for any non-uint8 input, but
  it's worth spot-checking that stretch against a real ISRO scene —
  a narrower or wider percentile window may serve high-dynamic-range
  imagery better than the current default.
- **Band order / count**: confirm whether the product delivers
  RGB directly or as separate band files / a different band order —
  `_load_geotiff()` currently assumes the first 3 bands of a multi-band
  file are R,G,B in that order.
- **CRS/projection**: ISRO products are typically delivered
  geocoded (UTM or geographic WGS84); `rasterio` (used throughout
  `geo_io.py`) handles arbitrary CRSes transparently, so this should
  need no changes, but it's worth confirming the CRS round-trips
  correctly through `write_dsm_geotiff()` for your specific product.
- **Metadata sidecar files**: some ISRO products ship georeferencing
  or radiometric calibration info in a separate `.xml`/`.txt` file
  rather than embedded in the GeoTIFF/IMG itself. The current pipeline
  only reads what's embedded in the raster; if the evaluation imagery
  relies on a sidecar file for correct georeferencing, that'll need a
  small loader addition once you can share a sample product (even just
  the file naming convention and whether the sidecar is required for
  correct geolocation).

**If you can share a sample ISRO product (or even just its exact
sensor name / band count / bit depth / file format), I can adapt
`geo_io.py` precisely instead of leaving this as general guidance** —
the message that mentioned this got cut off before the specifics came
through.
