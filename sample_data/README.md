# sample_data/

Synthetic demo/test fixtures (no real satellite imagery required to try
the pipeline).

- `sample_scene.png` — synthetic non-georeferenced "urban" scene
  (procedural buildings + cast shadows on a flat base). Run through
  the pipeline with no calibration source to produce a relative rDSM.
- `sample_buildings.png` — synthetic scene with 8 rectangular
  rooftops of varying size and shadow length, for exercising
  `--mode buildings` (footprint detection + shadow-height extrusion —
  see `docs/ARCHITECTURE.md §4`). `out_buildings/` is the corresponding
  example output, including `buildings.json` (detected footprints +
  estimated heights).
- `sample_dubai_complex.jpg` — a real (non-synthetic) satellite photo
  with dense, tightly-packed hangar buildings, used to find and fix
  the watershed-splitting bug documented in `docs/ARCHITECTURE.md §4`'s
  fix history (adjacent buildings merging into one giant rejected
  blob). `out_dubai_complex/` is the corresponding example output —
  178 buildings detected, vs. ~2 usable detections before that fix.
- `sample_geo.tif` — synthetic georeferenced scene (EPSG:32633, 2m/px)
  with the same building/shadow pattern over a rolling-hill base
  elevation.
- `srtm_ref.tif` — a synthetic "coarse reference DEM" for
  `sample_geo.tif`: the true elevation field downsampled to 32×32 then
  upsampled back, simulating what a real SRTM 30-90m tile would offer
  relative to a much higher-resolution optical scene.
- `lidar_truth.tif` — the full-resolution "true" elevation field used
  to generate `sample_geo.tif`'s relief, used as independent ground
  truth for the validation step.
- `gcps.csv` — 25 synthetic ground control points sampled from the
  same true elevation field, for testing the GCP calibration path.
- `out_relative/`, `out_geo/` — example pipeline outputs (DSM GeoTIFF,
  GLB mesh, viewer bundle, calibration report) from running the CLI on
  `sample_scene.png` (no calibration) and `sample_geo.tif` +
  `srtm_ref.tif` (coarse-DEM calibration) respectively. Regenerate with:

  ```bash
  cd ../backend
  python -m depthwizard.cli --input ../sample_data/sample_scene.png \
      --output ../sample_data/out_relative --backbone classical
  python -m depthwizard.cli --input ../sample_data/sample_geo.tif \
      --output ../sample_data/out_geo --backbone classical \
      --reference-dem ../sample_data/srtm_ref.tif
  ```

Regeneration code for all synthetic fixtures is in
`backend/tests/conftest.py` (pytest fixtures) — the same generation
logic, at larger scale, produced these standalone files.

**These are procedurally generated stand-ins, not real remote-sensing
imagery.** They exist purely so the pipeline, API, and tests are fully
exercisable without requiring you to source a licensed satellite scene
or LiDAR tile first. Swap in real imagery (and a real SRTM/LiDAR tile)
by pointing `--input` / `--reference-dem` / `--validate` at your own
files — the pipeline doesn't care where the data came from.
