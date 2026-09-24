# DepthWizard

**Single-view height estimation and 3D flythrough for optical remote-sensing imagery.**

DepthWizard turns a single overhead RGB image — a plain PNG/JPG or a
georeferenced GeoTIFF — into an elevation model and a navigable 3D
terrain you can fly through in a browser. No stereo pair, no LiDAR, no
InSAR pass required.

```
 RGB image (PNG/JPG/GeoTIFF)
        │
        ▼
 monocular depth backbone  ──►  relative depth/relief map
        │
        ▼
 scale calibration (coarse DEM / GCPs / none)
        │
        ▼
 DSM (absolute, GeoTIFF)  or  rDSM (relative)
        │
        ▼
 textured 3D mesh (GLB) + web viewer bundle
        │
        ▼
 Three.js flythrough  +  RMSE/MAE/correlation validation
```

## Repository layout

```
depthwizard/
├── backend/
│   ├── depthwizard/          # Core Python package (the actual pipeline)
│   │   ├── depth_model.py    # Pluggable monocular depth backbones
│   │   ├── calibration.py    # Relative depth -> absolute elevation
│   │   ├── geo_io.py         # PNG/JPG/GeoTIFF I/O, DSM export
│   │   ├── mesh.py           # Terrain mesh + GLB + web viewer bundle
│   │   ├── evaluation.py     # RMSE/MAE/correlation, stratified by land class
│   │   ├── pipeline.py       # End-to-end orchestration
│   │   └── cli.py            # Command-line interface
│   ├── api/server.py         # FastAPI backend (upload/process/validate)
│   ├── tests/                # pytest suite (15 tests)
│   └── requirements.txt
├── frontend/                 # Three.js Interactive Visualization Platform
│   ├── index.html
│   ├── css/style.css
│   └── js/{main,terrain,api}.js
├── docker/                   # Standalone deployment
│   ├── Dockerfile
│   └── docker-compose.yml
├── sample_data/              # Synthetic demo scenes (see below)
└── docs/
    ├── ARCHITECTURE.md
    ├── USER_GUIDE.md
    └── EVALUATION.md
```

## Quick start

### Option A — Docker (recommended, one command)

```bash
cd docker
docker compose up --build
```

Then open **http://localhost:8000** — the FastAPI backend serves the
Three.js frontend from the same process.

### Option B — Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn api.server:app --reload --port 8000
```

Open **http://localhost:8000**.

### Option C — CLI only (no web UI)

```bash
cd backend
python -m depthwizard.cli \
    --input scene.png \
    --output out/ \
    --backbone classical
```

For a georeferenced scene with scale calibration:

```bash
python -m depthwizard.cli \
    --input scene.tif --output out/ \
    --backbone dpt_hybrid \
    --reference-dem srtm_tile.tif \
    --vertical-exaggeration 1.5
```

Outputs land in `out/`:
- `dsm.tif` — the DSM/rDSM as a GeoTIFF (single-band float32)
- `terrain.glb` — textured 3D mesh (importable into Unity / Babylon.js / Blender)
- `viewer_bundle/` — `texture.jpg` + `heightmap.png` + `meta.json` for the web viewer
- `calibration_report.json` — method used, fitted scale/shift, RMSE/MAE/R² if a reference was supplied

## Depth backbones

| Key | Type | Requires | Notes |
|---|---|---|---|
| `classical` | Classical CV (shading + texture + shadow-length relief) | Nothing extra — pure OpenCV/NumPy | Runs anywhere, zero GPU, zero download. Used by default in this offline demo. **Not** a substitute for a learned model in production. |
| `depth_anything_v2_small` | Learned foundation model | `torch`, `transformers` | Recommended default once torch is installed. |
| `dpt_hybrid` / `dpt_large` | Learned (MiDaS/DPT family) | `torch`, `transformers` | Strong general-purpose relative depth. |
| `midas_small` / `midas_hybrid` | Learned (torch.hub MiDaS) | `torch` | Loaded via `intel-isl/MiDaS` torch.hub repo. |
| `auto` | Tries a learned model first, falls back to `classical` | — | Safe default for mixed environments. |

Swapping backbones is a one-line change (`--backbone depth_anything_v2_small`
on the CLI, or the dropdown in the web UI) — see `docs/ARCHITECTURE.md` for why
the interface is designed this way and how to plug in a fine-tuned checkpoint.

## Datasets

See `docs/DATASETS.md` for:
- Integration with **GAMUS** (the recommended training/validation benchmark — real LiDAR AGL ground truth, 5 US cities, HDF5 tiles), including a working loader (`depthwizard/datasets/gamus.py`), a validation script (`scripts/validate_on_gamus.py`), and a fine-tuning scaffold (`scripts/finetune_on_gamus.py`).
- Notes on ingesting **ISRO RGB-band optical satellite imagery** for final evaluation.

## Reconstruction modes

DepthWizard supports two ways of turning elevation into 3D geometry,
selectable per-run:

- **`relief` (default)** — a continuous per-pixel heightfield. Good for
  terrain, hills, gentle slopes, and scenes without hard height
  discontinuities.
- **`buildings`** — detects rectangular rooftop footprints and extrudes
  each as a clean flat-roofed box on a flat ground plane, with height
  estimated from cast-shadow length. Use this for dense urban/rooftop
  scenes, where `relief`'s per-pixel approach produces spiky artifacts
  because the true geometry is discontinuous (flat roof → vertical
  wall → flat ground), not smoothly varying terrain. See
  `docs/ARCHITECTURE.md §4` for why, and `docs/EVALUATION.md` for a
  before/after.

```bash
python -m depthwizard.cli --input scene.png --output out/ --mode buildings \
    --building-min-height-m 4 --building-max-height-m 35
```

**Building heights from a single top-down photo are a shadow-length-based
estimate, not a measurement** — see the caveat in
`depthwizard/building_extraction.py` and the note below on what's
fundamentally recoverable from a single nadir image.

## A note on what single-view reconstruction can and can't do

A straight-down (nadir) photograph contains no view of building
facades — you only ever see rooftops. That means there is no direct
visual cue for *how tall* a rooftop is; every approach here (continuous
relief or building-block extrusion) is inferring height from indirect,
imperfect cues (shading, shadow length, local texture), not measuring
it. Both modes will always fall short of what a true oblique/multi-view
photograph or LiDAR survey can give you. If a task genuinely requires
photograph-quality oblique 3D (visible facades, precise heights), that
requires either stereo/multi-view imagery, LiDAR, or a survey with
known building heights (fed in as GCPs) — not single-view estimation
from any model, learned or classical. Being upfront about this
boundary is more useful than tuning parameters to chase a result the
input image doesn't contain enough information to produce.

## Why this demo ships a classical fallback

The evaluation environment this project was authored in has no GPU and a
constrained, allow-listed network with no access to model hubs (HuggingFace,
`download.pytorch.org`, etc.), and only ~a few GB of free disk — not enough
to safely install `torch` with its default CUDA dependency closure. Rather
than ship untested stub code for the "real" depth model path, `depth_model.py`
implements a genuine, working, dependency-light relief estimator so that
**the full pipeline you're looking at actually runs, end-to-end, right now** —
ingest → depth → calibration → DSM → mesh → validation. The learned-model
code path (`TorchHubDepthBackbone`) is fully implemented and is the
recommended path for production; it just wasn't the one exercised in this
sandbox's automated tests.

## License / attribution

This is a reference implementation delivered as source code and documentation
for the DepthWizard problem statement. See `docs/ARCHITECTURE.md` for design
rationale and `docs/EVALUATION.md` for how it maps to the stated evaluation
criteria.
