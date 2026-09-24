# User Guide

## Starting the app

```bash
cd docker && docker compose up --build
# or, without Docker:
cd backend && pip install -r requirements.txt && uvicorn api.server:app --port 8000
```

Open **http://localhost:8000** in a browser.

## 1. Upload imagery

Drag a file onto the **Input Imagery** dropzone, or click it to browse.
Supported formats:

- **PNG / JPG** — treated as non-georeferenced. Output will be a
  *relative* DSM (rDSM) unless you also provide GCPs (see below —
  GCPs require pixel coordinates, so they work for ungeoreferenced
  images too).
- **GeoTIFF (.tif/.tiff)** — if it carries a valid CRS + transform,
  you can additionally supply a reference DEM for absolute-scale
  calibration.

## 2. Choose a depth backbone

- **Classical relief estimator** — works immediately, no download,
  runs on CPU. This is the default and what's exercised by this
  project's automated tests.
- **Depth-Anything-V2 / DPT-Hybrid / DPT-Large** — learned foundation
  models. Requires `torch` + `transformers` to be installed on the
  server (see `README.md`); the first request downloads and caches
  the checkpoint.
- **Auto** — tries a learned model, falls back to classical if
  unavailable.

## 3. Scale calibration (GeoTIFF inputs)

Three options, mutually exclusive:

- **None** — output is a relative DSM, useful for visualization and
  relative height comparisons only.
- **Coarse reference DEM** — upload a local SRTM (or any other) DEM
  GeoTIFF tile covering the same area. The pipeline reprojects it onto
  your image's grid and fits scale/shift automatically.
- **Ground Control Points** — upload a CSV with columns `row, col,
  elevation_m` (pixel row/column in the *input image*, and the known
  elevation in meters at that pixel). At least 4 points are required;
  15-20+ well-distributed points give a materially more stable fit.
  Example:

  ```csv
  row,col,elevation_m
  12,340,412.6
  88,120,405.1
  ...
  ```

## 4. Mesh & rendering settings

**Reconstruction mode:**
- **Continuous relief** (default) — a per-pixel heightfield. Best for
  terrain, hills, gentle slopes.
- **Building blocks** — detects rectangular rooftop footprints and
  extrudes clean boxes using a shadow-length height estimate. Use this
  for dense urban/rooftop scenes — if "Continuous relief" gives you a
  spiky, needle-like mesh, this is the fix, not a parameter tweak.
  When selected, two extra sliders appear:
  - **Min/max building height (m)** — the plausible height range
    footprints are rescaled into (shadow length alone is a *relative*
    signal without a known sun angle and pixel size, so this sets the
    absolute range it's mapped onto).

  Building-block mode replaces the vertical-exaggeration slider (it
  doesn't apply — buildings are extruded to their estimated real
  height directly) and disables the Wireframe/Elevation-Colormap
  toggles (they're designed for a continuous heightfield's shader,
  not a set of discrete boxes).

  **If you're getting a lot of false-positive boxes on roads/parking
  lots**, the web UI's two sliders won't fix that — you'll need the
  CLI's advanced detection flags (`--building-min-area-px`,
  `--building-min-extent`, `--building-max-aspect-ratio`,
  `--building-min-width-px`, `--building-min-separation-px`), which
  aren't yet exposed in the web form. See `docs/ARCHITECTURE.md §5`
  for why this happens (real pavement and real rooftops can be
  photometrically near-identical) and what those flags actually do.
  There's no universally-correct setting; it's a genuine trade-off
  you tune per image.

- **Vertical exaggeration** (relief mode only) — multiplies elevation
  before meshing. Useful for subtle terrain (hilly/rural scenes); set
  back to 1.0 for a physically accurate read of building/structure
  heights.
- **Mesh resolution** (relief mode only) — number of grid samples
  along the longer image edge used for the 3D mesh and viewer. Higher
  = more geometric detail but slower to render; the underlying DSM
  GeoTIFF is always kept at full resolution regardless of this
  setting.

## 5. Generate & explore

Click **Generate DSM & 3D Terrain**. Progress is logged in the panel.
Once complete:

- The 3D viewport loads the terrain automatically.
- **Orbit mode** (default): left-drag to rotate, scroll to zoom,
  right-drag to pan.
- **Flythrough mode**: click the **Flythrough** toolbar button (or
  press `Tab`) to enter first-person navigation. Click the viewport to
  lock the mouse, then:
  - `W`/`A`/`S`/`D` or arrow keys — move
  - Mouse — look around
  - `Space` / `Shift` — move up / down
  - `Esc` — release mouse lock
  - `Tab` — return to orbit mode
- **Wireframe** — toggles a wireframe overlay for inspecting mesh
  density and slope facets.
- **Elevation Colormap** — swaps the optical texture for a hypsometric
  tint (blue → green → yellow → brown → white) keyed to elevation,
  useful for quickly reading relief without the optical clutter.
- The HUD (top-right) reports the elevation under your cursor (orbit
  mode) or directly beneath the camera (flythrough mode), plus current
  camera height.

## 6. Download results

- **Download DSM (.tif)** — the elevation GeoTIFF, for use in GIS
  software (QGIS, ArcGIS) or downstream analysis.
- **Download Mesh (.glb)** — the textured 3D mesh, importable into
  Unity, Babylon.js, Blender, or any glTF-compatible tool.

## 7. Validate against a reference

If you have an independent elevation source (LiDAR-derived DSM, survey
grid, or any other GeoTIFF elevation raster covering the same area),
upload it in the **Validate Against Reference** section and click
**Run Validation**. You'll get:

- Overall RMSE, MAE, and Pearson correlation.
- A breakdown by landscape class (urban / sparse / hilly / forested),
  computed from a heuristic classifier over the input image and
  elevation (see `docs/ARCHITECTURE.md` for its limitations).

## Command-line usage (no browser needed)

```bash
cd backend
python -m depthwizard.cli --input scene.png --output out/
python -m depthwizard.cli --input scene.tif --output out/ --reference-dem srtm.tif
python -m depthwizard.cli --input scene.tif --output out/ --gcps gcps.csv
python -m depthwizard.cli --input out/dsm.tif --validate lidar_ref.tif
```

Run `python -m depthwizard.cli --help` for the full flag list.

## Troubleshooting

- **"Not enough valid coarse-DEM samples to calibrate scale"** — your
  reference DEM likely doesn't overlap the input image's extent, or
  its CRS/transform is missing. Confirm the two files cover the same
  ground footprint.
- **"At least 4 GCPs are required..."** — check your CSV has a header
  row (`row,col,elevation_m`) and at least 4 data rows with valid
  integers/floats.
- **Job stuck on "running"** — check server logs; a learned backbone
  selected without `torch`/`transformers` installed will raise on the
  first request (the "auto" backbone instead falls back silently).
- **Heightmap looks blocky in the viewer** — raise "Mesh resolution";
  note this affects only the 3D mesh, not the underlying DSM GeoTIFF.
