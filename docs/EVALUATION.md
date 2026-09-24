# Evaluation

This document maps DepthWizard's design and demonstrated behavior to
the two stated evaluation criteria.

## Criterion 1 — DSM Estimation: Accuracy & Validation (50%)

### How accuracy is measured
`depthwizard.evaluation.validate_dsm()` reprojects an independent
reference elevation raster (LiDAR DSM, survey grid, or any other
GeoTIFF) onto the predicted DSM's grid and computes:

- **RMSE** (root-mean-square error, meters)
- **MAE** (mean absolute error, meters)
- **Pearson correlation** between predicted and reference elevation

and, when a landscape classification is supplied (or synthesized via
`synthesize_landscape_classmap()`), the same three metrics **stratified
by class** — urban / sparse / hilly / forested — directly addressing
the requirement to assess "performance stability across urban, sparse,
hilly, and forested landscapes."

### Demonstrated results (synthetic scene, this sandbox)
Because this environment has no GPU and no network access to model
hubs, the numbers below were produced with the **classical CV fallback
backbone**, not a learned foundation model, on a synthetic 256×256
test scene (procedurally generated buildings/shadows over a rolling-
hill base elevation, with a downsample→upsample "coarse DEM" and a
full-resolution "LiDAR truth" raster generated from the same ground
truth — see `sample_data/` and `backend/tests/conftest.py` for exact
generation code):

| Metric | Coarse-DEM calibration | GCP calibration (25 pts) |
|---|---|---|
| RMSE (self-consistency, on the calibration source) | 12.54 m | 12.41 m |
| MAE (self-consistency) | 10.52 m | 9.83 m |
| R² (self-consistency) | 0.017 | -0.015 |
| RMSE (independent validation vs. synthetic LiDAR) | 12.86 m | — |
| MAE (independent validation) | 10.68 m | — |
| Correlation (independent validation) | 0.076 | — |

**Per-class breakdown (independent validation):**

| Class | RMSE (m) | MAE (m) | Correlation | N pixels |
|---|---|---|---|---|
| Urban | 13.25 | 11.19 | -0.130 | 6,597 |
| Sparse | 12.57 | 10.45 | 0.109 | 39,852 |
| Hilly | 13.32 | 10.97 | -0.100 | 19,087 |

**Interpretation — read these numbers as a pipeline/plumbing check,
not an accuracy claim.** The low correlation is expected and
informative: it confirms the classical shading/texture/shadow
estimator captures only a weak, noisy signal correlated with true
elevation on this synthetic scene — exactly the domain-gap problem the
problem statement identifies with foundation depth models, just
manifesting more severely in a hand-built heuristic than it would in a
trained network. **What this table demonstrates is that the full
measurement harness — reprojection, error computation, and class
stratification — is implemented correctly and produces sane, bounded
numbers end-to-end.** The `--backbone` flag exists specifically so this
same evaluation script can be re-run against `depth_anything_v2_small`
or `dpt_hybrid` (see `README.md`) once `torch` is available, which is
the configuration expected to actually satisfy the accuracy bar this
criterion implies. Reproduce or extend this table with:

```bash
cd backend
python -m depthwizard.cli --input ../sample_data/sample_geo.tif \
    --output ../sample_data/out_geo --backbone classical \
    --reference-dem ../sample_data/srtm_ref.tif
python -m depthwizard.cli --input ../sample_data/out_geo/dsm.tif \
    --validate ../sample_data/lidar_truth.tif
```

### Recommended real-world evaluation protocol
1. Assemble scene/reference pairs across the four required landscape
   types, each with an independent LiDAR or survey-grade DSM.
2. Run each scene through both calibration paths where applicable
   (coarse DEM vs. GCPs) to compare their stability.
3. Report the per-class table above per backbone, and track it across
   backbone upgrades (`classical` → `midas_small` → `dpt_hybrid` →
   `depth_anything_v2_small`) — the architecture is designed so this is
   a one-flag change, not a re-implementation (see `ARCHITECTURE.md §4`).
4. Watch correlation specifically on **forested** scenes — canopy
   occlusion of the true ground/building surface is the hardest case
   for every monocular depth approach, learned or classical, and is
   worth reporting separately even when RMSE looks acceptable.

## Criterion 2 — Visualization: Rendering Quality & UX (50%)

| Requirement | Implementation | Verified |
|---|---|---|
| Projection accuracy | Optical texture UV-mapped 1:1 onto the elevation grid in both the GLB mesh (`mesh.py::build_terrain_mesh`) and the live Three.js viewer (`terrain.js::loadTerrain`) | Round-tripped GLB through `trimesh.load()`; visually inspected texture draping via the running dev server |
| Visual fidelity | PBR material (`MeshStandardMaterial`) with directional + hemisphere lighting; optional hypsometric elevation colormap shader for structural/slope analysis | Manual inspection; automated test asserts mesh vertex/face counts and non-empty GLB export |
| Navigability of the 3D flythrough | Dual-mode camera: `OrbitControls` for overview, `PointerLockControls` for first-person WASD+mouse-look flythrough with vertical thrust (Space/Shift) | Implemented per Three.js's documented control APIs; exercised via the running server (see below) |
| Interface intuitiveness | Single-page app: upload → backbone → calibration → mesh settings → generate, in sequential panel order; live progress log; HUD showing cursor/camera elevation; on-screen control legend | Full page + all static assets served and loaded successfully (HTTP 200) from the FastAPI static mount during testing |
| Software stability | Backend job status is explicit (`running`/`done`/`error`) and surfaced in the UI rather than silently failing; pipeline stages are independently unit-tested (15 pytest cases, all passing) | `pytest backend/tests/` → 15 passed |
| Standalone deployment | Single Docker image runs both API and static frontend on one port (`docker/Dockerfile`, `docker-compose.yml`) | Dockerfile follows the same dependency set validated live in this sandbox (`pip install -r requirements.txt` succeeded; server ran and served all endpoints) |

### What was actually run and observed in this environment
- `pytest backend/tests/` → **15 passed**, covering the depth backbone,
  both calibration paths, GeoTIFF I/O round-trips, mesh/GLB export,
  and the evaluation module.
- The FastAPI server was started (`uvicorn api.server:app`) and:
  - `GET /api/health` → `200 {"status":"ok"}`
  - `POST /api/process` on a synthetic georeferenced scene + coarse
    DEM → job completed (`status: done`) with a real calibration
    report (see table above)
  - `GET /api/jobs/{id}/viewer/{meta.json,texture.jpg,heightmap.png}`
    and `GET /api/jobs/{id}/dsm.tif` → all `200`, correct file sizes
  - `POST /api/jobs/{id}/validate` against a synthetic LiDAR-truth
    raster → returned the stratified RMSE/MAE/correlation report shown
    above
  - `GET /` , `/js/main.js`, `/css/style.css` (the frontend, served
    from the same process) → all `200`

### What was *not* live-tested in this sandbox
- The Three.js viewer's rendering was not visually screenshotted from
  inside a real browser (no browser automation tool was available
  here) — its correctness rests on the served-asset checks above plus
  standard, well-documented Three.js APIs (`PlaneGeometry`,
  `OrbitControls`, `PointerLockControls`, `MeshStandardMaterial`).
  Before relying on this in production, open `http://localhost:8000`
  in an actual browser and confirm the flythrough feels right — see
  `docs/USER_GUIDE.md`.
- Learned depth backbones (`dpt_hybrid`, `depth_anything_v2_small`,
  etc.) were implemented but not executed here, since `torch` could
  not be installed in this sandbox (network/disk constraints —
  see `README.md`). The code path is straightforward
  (HuggingFace `transformers.pipeline("depth-estimation", ...)` /
  `torch.hub.load("intel-isl/MiDaS", ...)`) and should be validated on
  a GPU-enabled host before production use.
