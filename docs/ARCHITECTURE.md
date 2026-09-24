# Architecture

## 1. Design goals

1. **One pipeline, two input regimes.** Non-georeferenced PNG/JPG and
   georeferenced GeoTIFF share every stage except calibration. This
   keeps the codebase small and means improvements to the depth
   backbone or mesh export benefit both regimes automatically.
2. **Backbone-agnostic.** Monocular depth foundation models are an
   actively moving target (MiDaS → DPT → Depth-Anything → Depth-Anything-V2
   → whatever ships next quarter). `depth_model.py` defines a single
   `DepthBackbone.predict(rgb) -> depth` contract so the rest of the
   system — calibration, DSM export, meshing, evaluation — never
   changes when the backbone does.
3. **Absolute scale is recovered, not assumed.** Foundation depth
   models are trained on affine-invariant losses; their raw output
   has no metric meaning. `calibration.py` treats scale recovery as
   its own explicit, testable stage with two independent strategies
   (coarse DEM registration, GCP fitting) rather than baking a fixed
   scale factor into the depth model.
4. **Graceful degradation.** If no reference elevation is available,
   the system still produces a usable, correctly-labeled relative
   DSM (rDSM) rather than fabricating false absolute values.
5. **Mesh generation is decoupled from elevation resolution.** DSMs
   can be tens of megapixels; a 1:1 mesh at that density is
   unnavigable in real time. `mesh.py` explicitly downsamples for the
   *rendering* mesh while the *analysis* DSM (GeoTIFF) stays
   full-resolution.

## 2. Pipeline stages

### 2.1 Ingest (`geo_io.py`)
- PNG/JPG → loaded as plain RGB, `is_georeferenced=False`.
- GeoTIFF → read via `rasterio`; CRS + affine transform captured;
  `is_georeferenced` is set based on whether the transform is a
  non-trivial affine (guards against "GeoTIFF container, no real
  geo-metadata" edge cases).
- Non-uint8 sources (e.g. 16-bit panchromatic/multispectral) are
  contrast-stretched (2nd/98th percentile) to a displayable/model-
  ready 8-bit RGB.

### 2.2 Depth estimation (`depth_model.py`)
Two backbone families implement `DepthBackbone`:

- **`TorchHubDepthBackbone`** — the production path. Loads a
  HuggingFace `transformers` depth-estimation pipeline (preferred; it
  caches weights locally and works fully offline after first
  download) or a `torch.hub` MiDaS checkpoint. Torch is imported
  lazily inside the constructor so importing `depthwizard` never
  requires torch to be installed — a system with no learned-model
  support degrades to the classical backbone instead of crashing at
  import time.
- **`ClassicalReliefBackbone`** — the CPU-only fallback used
  throughout this repo's tests and demo. It fuses three classical
  monocular cues:
  1. *Multi-scale shading relief*: a Laplacian-pyramid gradient
     integration approximating shape-from-shading, capturing both
     fine (rooftop-scale) and coarse (hill-scale) structure.
  2. *Local texture energy*: variance-based roughness prior — urban
     rooftops, rubble, and forest canopy read as "rougher" than roads,
     water, or bare fields.
  3. *Cast-shadow length*: dark regions adjacent to strong edges are
     treated as shadows; their extent is a coarse, physically-motivated
     height proxy independent of the shading model.
  These are percentile-normalized (robust to outlier pixels), weighted,
  summed, median-despiked, and edge-preserving filtered against
  luminance. **This is intentionally not presented
  as a research contribution** — it is a working stand-in so the full
  system can be demonstrated without a GPU or model download; expect
  a real foundation model to substantially outperform it, particularly
  on illumination-ambiguous scenes (overcast imagery, nadir views with
  minimal shadow).

  **Known failure mode, fixed:** on flat, uniformly-lit nadir imagery
  (typical of satellite/drone orthophotos — dense rooftops, parking
  lots, roads with sharp painted lines and no real elevation change),
  raw Sobel-gradient edges from those non-elevation features can
  dominate a naive min-max normalization, crushing the whole scene
  toward zero except a handful of comb-like spikes at every sharp
  linear edge — visually, a mesh made of needles. The backbone now
  guards against this with (1) gradient-magnitude clipping at the 97th
  percentile before integration, (2) percentile-based (not min-max)
  normalization throughout, and (3) a median de-spike pass before the
  final guided filter. `docs/EVALUATION.md` has a before/after on a
  synthetic reproduction of this failure mode.

Backbones agree on a single depth convention (farther/lower = larger
value) so `calibration.py` never needs to know which backbone produced
its input.

### 2.3 Scale calibration (`calibration.py`)
Foundation depth models predict depth up to an unknown global affine
transform: `true ≈ a·predicted + b`. Two recovery strategies:

- **Coarse-DEM registration.** The relative height field is split into
  low-frequency (comparable to what a 30-90m DEM like SRTM can
  resolve) and high-frequency (fine detail only the vision model
  captures) components via Gaussian blur. A robust (IRLS, soft-L1)
  affine fit anchors the low-frequency component to the coarse DEM;
  the same fitted scale is applied to the high-frequency residual so
  fine structure isn't discarded. This mirrors how satellite
  photogrammetry pipelines commonly blend a coarse absolute reference
  with a high-resolution relative surface.
- **GCP fitting.** Given ≥4 (recommended 15-20+) `(row, col,
  elevation_m)` points, an IRLS affine fit maps predicted relative
  height directly to elevation. Cheaper to acquire than a DEM tile in
  some workflows (e.g. a handful of RTK-GPS survey shots); accuracy
  scales with point count and spatial spread.
- **Uncalibrated fallback.** With neither source, output is explicitly
  labeled `is_absolute=False` and rescaled to a nominal relief range —
  suitable for visualization and *relative* structural analysis
  (which building is taller than which), not for absolute measurement.

Both calibrated paths report RMSE/MAE/R² **on the calibration source
itself** (self-consistency, in `calibration_report.json`) —
this is distinct from, and should not be confused with, independent
validation against held-out LiDAR (`evaluation.py`, `docs/EVALUATION.md`).

### 2.4 DSM/rDSM export (`geo_io.py`)
Elevation is written as a single-band `float32` GeoTIFF. Georeferenced
inputs propagate their original CRS + affine transform, so the DSM
overlays correctly in GIS tools (QGIS, ArcGIS) against other layers.
Non-georeferenced inputs still get a valid GeoTIFF (local pixel grid,
EPSG:3857 container) so the file format contract stays uniform.

### 2.5 Visualization layer (`mesh.py`, `frontend/`)
Two artifacts are produced from the same elevation + texture pair:

- **`terrain.glb`** — a textured triangle mesh (downsampled to
  `mesh_resolution`, default 512 samples on the long edge), suitable
  for import into Unity, Babylon.js, Blender, or any glTF-compatible
  engine, per the problem statement's "integrate with a rendering
  engine" requirement.
- **Viewer bundle** (`texture.jpg` + `heightmap.png` + `meta.json`) —
  a lighter-weight representation for the bundled Three.js app. The
  frontend builds a `PlaneGeometry`, displaces each vertex by
  bilinearly sampling the heightmap, and drapes the original optical
  image as a texture. Building the mesh client-side (rather than
  shipping a pre-baked GLB to the browser) keeps the payload small and
  lets a future version trade mesh density for frame rate live.

The Three.js app (`frontend/js/main.js` + `terrain.js`) provides:
- **Orbit mode** (`OrbitControls`) for general inspection.
- **Flythrough mode** (`PointerLockControls`) — WASD + mouse-look +
  Space/Shift for vertical movement, for first-person navigation as
  called for in the problem statement.
- **Wireframe** and **hypsometric elevation colormap** shader toggles
  for structural/slope analysis.
- A live HUD reporting cursor/ground elevation and camera height,
  and a validation panel that uploads an independent reference raster
  and renders the resulting RMSE/MAE/correlation table in-browser.

### 2.6 Validation (`evaluation.py`)
Given a predicted DSM and an independent reference (LiDAR DSM, survey
grid, or any GeoTIFF elevation raster), `validate_dsm()` reprojects the
reference onto the prediction's grid and reports overall RMSE/MAE/
Pearson correlation, plus a breakdown by landscape class (urban /
sparse / hilly / forested) when a class map is supplied.
`synthesize_landscape_classmap()` is a heuristic (color + local relief
statistics) stratifier used when no real land-cover raster is
available; production deployments should substitute a genuine
land-cover product (e.g. ESA WorldCover, Dynamic World) for a
reproducible, literature-comparable stratified evaluation.

## 3. Service architecture (`api/server.py`)

FastAPI app, in-process background threads per job (adequate for a
demo/single-node deployment; swap `threading.Thread` for a task queue —
Celery/RQ — for multi-worker production deployments where jobs must
survive a server restart). Endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/process` | Upload image (+ optional reference DEM / GCP CSV) → returns `job_id` |
| GET | `/api/jobs/{id}` | Poll status (`running` / `done` / `error`) + result metadata |
| GET | `/api/jobs/{id}/viewer/{file}` | Serve `texture.jpg` / `heightmap.png` / `meta.json` |
| GET | `/api/jobs/{id}/dsm.tif` | Download the DSM GeoTIFF |
| GET | `/api/jobs/{id}/terrain.glb` | Download the textured mesh |
| POST | `/api/jobs/{id}/validate` | Upload a reference raster → RMSE/MAE/correlation report |

The same FastAPI process serves the static frontend, so the whole
system deploys as a single container (`docker/Dockerfile`) — satisfying
the "deployable as a unified module" / "standalone deployment"
requirement.

## 4. Two reconstruction geometries: relief vs. building blocks

The continuous per-pixel heightfield approach in §2.5 has a structural
weakness on scenes dominated by dense rectilinear rooftops: it assumes
the surface varies *smoothly*, but the true geometry there is
*discontinuous* — flat roof, then a vertical drop, then flat ground.
Feeding a discontinuous signal through per-pixel gradient-based relief
estimation (even with the robustness fixes in §2.2) produces visible
"comb" artifacts at every roof edge, because the estimator has no
concept of "this edge is a wall, not a slope."

`building_extraction.py` implements a second, purpose-built geometry
for exactly this case:
1. **Footprint detection** — segments locally-uniform, non-green,
   non-elongated, roughly-rectangular blobs (`extract_building_footprints`),
   explicitly filtering out roads (elongated), vegetation (green/
   textured), and open ground (large, low-extent blobs).
2. **Shadow-length height estimation** — samples a cast-shadow-length
   proxy (the same cue as the classical relief backbone's shadow term,
   factored out into a standalone function) in a ring around each
   footprint, then either (a) converts it to an absolute height via
   `height = shadow_px * pixel_size_m * tan(sun_elevation_deg)` if
   both are known, or (b) ranks footprints by shadow signal and area
   and rescales into a user-supplied plausible height band otherwise.
3. **Extrusion** — each footprint polygon is extruded to a flat-topped
   box (`trimesh.creation.extrude_polygon` via `shapely`), roof-textured
   with the corresponding patch of the original image, walls given a
   flat neutral shade (single-view nadir imagery contains no facade
   information to texture them honestly with).

This produces genuinely box-shaped, discontinuous-geometry-correct
buildings instead of melted spikes — see `docs/EVALUATION.md` for a
concrete before/after and `README.md`'s "what single-view
reconstruction can and can't do" for the honest ceiling on this
approach: it still cannot recover facade detail or a truly
photograph-accurate height, because a nadir image contains no direct
view of a building's vertical extent.

**Two bugs found and fixed after initial real-world testing** (the
synthetic test scenes used during development didn't surface either):

1. **Rotated thin slivers passing footprint filtering.** The original
   filter computed extent/aspect-ratio from `cv2.boundingRect`
   (axis-aligned). A thin diagonal road stripe or parking-lot line can
   have a deceptively square-looking *axis-aligned* bounding box while
   still being a thin sliver in its true, rotated extent — so it
   passed the aspect-ratio check meant to reject exactly that shape.
   Fixed by computing extent/aspect from `cv2.minAreaRect` (the
   oriented rectangle) instead, plus an added `min_width_px` check on
   the oriented rectangle's short side as a second, independent guard.
   Regression test: `test_extract_building_footprints_rejects_rotated_thin_stripes`.
2. **Ground texture silently discarded on export.** The original
   implementation built one `Trimesh` per building plus the ground
   plane, then merged them all with `trimesh.util.concatenate()`.
   That function downgrades *any* mix of `TextureVisuals` (the ground
   plane's UV-mapped optical photo) and `ColorVisuals` (the flat-shaded
   building walls) into a single `ColorVisuals` for the whole merged
   mesh — silently dropping the ground texture and rendering it as
   flat gray. Fixed by returning a `trimesh.Scene` with the ground and
   each building as separate named geometries/primitives, which both
   glTF and three.js's `GLTFLoader` support natively with per-primitive
   materials. Regression test:
   `test_build_building_block_mesh_has_ground_and_extrusions` now
   explicitly asserts the ground keeps its `TextureVisuals` after a
   full GLB export/reload round-trip.

**Remaining honest caveat**: `extract_building_footprints()` is a
hand-crafted heuristic (uniformity + color + shape filters +
watershed instance splitting), validated against both synthetic test
scenes and a real dense satellite photo (178 buildings detected —
`sample_data/sample_dubai_complex.jpg`, see the fix history below).
It has NOT been validated against imagery with substantially different
characteristics (different sensor, different lighting/season, very
different building density or material palette) — expect to need
per-image parameter tuning (`min_area_px`, `min_extent`,
`max_aspect_ratio`, `min_width_px`, `min_separation_px`) on imagery
that looks meaningfully different from what's been tested. For
production-grade footprint accuracy at scale across arbitrary imagery,
this heuristic should eventually be replaced with a trained
building-footprint segmentation model (e.g. a U-Net fine-tuned on
SpaceNet or GAMUS's building class — see `docs/DATASETS.md`) — that is
a materially bigger undertaking than what's implemented here.

**Fix history (chronological, each found via real-image testing, not
anticipated in advance):**
1. *Rotated thin slivers passing filtering* — axis-aligned bounding-box
   aspect/extent checks don't reject thin *rotated* slivers (a diagonal
   road stripe can look square in axis-aligned terms). Fixed by
   switching to `cv2.minAreaRect` (oriented rectangle) for all shape
   checks, plus a `min_width_px` guard on the oriented rect's short side.
2. *Ground texture silently discarded on export* — merging the ground
   plane and building meshes with `trimesh.util.concatenate()`
   downgrades any mix of `TextureVisuals` + `ColorVisuals` to flat
   vertex colors for the whole merged mesh. Fixed by exporting a
   `trimesh.Scene` with the ground and each building as separate named
   glTF primitives instead.
3. *Adjacent buildings merging into one giant rejected blob* — simple
   thresholding + morphological closing bridges a whole row of
   buildings with only a few pixels of gap into a single connected
   component, which then fails the rectangularity filters wholesale
   (its extent is low because it spans multiple buildings plus gaps) —
   so an entire row of real buildings was silently dropped rather than
   detected individually. This was the dominant real-world failure:
   on a real test photo with dense adjacent hangar buildings, it meant
   ~2 usable detections out of dozens of visible buildings. Fixed with
   marker-based watershed splitting (`_split_into_instances()`,
   `scikit-image`'s `peak_local_max` + `watershed` on a distance
   transform of the candidate mask) before filtering — 178 buildings
   detected on the same test photo afterward.
4. *Jagged/thin extrusions from noisy contours* — `approxPolyDP` on a
   small or JPEG-compression-noisy blob can produce a many-vertex,
   near-irregular polygon rather than a clean quad; extruded, this
   produced visually broken near-vertical sliver shapes rather than
   boxes. Fixed by using the oriented rectangle's 4 corners
   (`cv2.boxPoints`) as the footprint polygon instead of the raw traced
   contour — every extruded shape is now guaranteed to be a clean box,
   since the whole premise of this mode is that real rooftops are
   approximately rectangular anyway.

Each fix above has a regression test in
`backend/tests/test_building_extraction.py`, including one that runs
against the real test photo (`test_real_world_image_detects_many_buildings_not_just_a_few`).

5. **Critical: ground plane and building extrusions authored in the
   wrong "up" axis.** The ground plane and every building extrusion
   were built with the north-south image extent varying along world
   Y (up/down) instead of world Z (forward/back), with the true
   extrusion height (from `trimesh.creation.extrude_polygon`, which
   extrudes along its own local Z) never remapped onto world Y either.
   Every other part of this codebase — `build_terrain_mesh`'s explicit
   `PlaneGeometry.rotateX(-π/2)`, the Three.js scene's camera, controls,
   and lighting — assumes the standard Y-up convention. The visible
   symptom, only obvious once you actually rotate the camera to a
   side-on view, was the **entire model rendering as a vertical
   wall/fin** standing upright instead of buildings on a flat ground
   plane — from most default viewing angles it could look deceptively
   plausible, which is why this shipped once already. Fixed by (a)
   building the ground plane with Y held at 0 for every vertex and X/Z
   carrying the horizontal extent, and (b) reordering each extruded
   building's vertex columns from trimesh's natural `(X, Z_footprint,
   H)` to `(X, H, Z_footprint)` after extrusion, landing the height on
   world Y to match. Regression test: `test_ground_plane_is_horizontal_yup`
   explicitly asserts the ground has constant Y and that buildings
   extrude upward along Y, not Z.

## 5. A known, hard limitation: distinguishing rooftops from pavement

On dense real-world scenes (see `sample_data/sample_dubai_complex.jpg`),
a meaningful share of false-positive footprints land on open pavement —
parking aprons, taxiways, large paved lots — rather than actual
buildings. This was investigated directly (not just guessed at): a
parking apron and an adjacent hangar rooftop in the test photo are
**photometrically almost identical** (both light uniform gray, both
locally smooth under the Laplacian-uniformity test), and a
"real-row-of-buildings-with-gaps" blob and a "one big paved lot" blob
can have overlapping extent/area signatures once morphological closing
has already bridged them together pre-split — there isn't a clean
color/shading-only rule that separates them on this image. This is a
genuine ceiling on a hand-crafted heuristic, not a parameter-tuning
oversight; see the "Fix history" `_split_into_instances` shadow-signal
and parent-blob-extent experiments in the project's development history
for what was tried and why it didn't cleanly generalize.

Two practical mitigations, in order of effort:
1. **Tune the exposed detection parameters** (`--building-min-area-px`,
   `--building-min-extent`, `--building-max-aspect-ratio`,
   `--building-min-width-px`, `--building-min-separation-px` — all
   threaded through `pipeline.run()`, the CLI, and the API). Raising
   `min_area_px` and `min_extent` together measurably helps (on the
   test photo: 178 → 40 detections, visibly less pavement clutter,
   while keeping the hangar rows and the circular complex) at the cost
   of losing some smaller/fainter real buildings. There is no single
   "correct" setting — it's a real precision/recall trade-off specific
   to each image's characteristics.
2. **Use an external building-footprint source instead of the CV
   heuristic entirely** for well-mapped locations — e.g. OpenStreetMap
   building polygons for the scene's geographic bounding box, using
   this codebase's shadow-length estimator only for height (not
   footprint shape). This would give genuinely exact footprints
   wherever OSM coverage exists, sidestepping the photometric ambiguity
   entirely. **This is not implemented in this codebase** — it's a
   concrete, scoped next step if footprint accuracy on real imagery
   needs to exceed what any color/shading heuristic can offer, and was
   deliberately not attempted half-built rather than shipped untested.

## 6. Known limitations & production upgrade path

| Limitation (current demo) | Production upgrade |
|---|---|
| Classical relief backbone by default | Install `torch`+`transformers`, switch `--backbone` to `depth_anything_v2_small`/`dpt_hybrid`; no other code changes needed |
| SRTM/GCP fetch is manual (user supplies a local file) | Add an automated SRTM/Copernicus DEM tile-fetch module keyed off the GeoTIFF's bounding box |
| In-process threading for jobs | Task queue (Celery/RQ) + object storage (S3/GCS) for outputs |
| Heuristic landscape classmap for stratified eval | Real land-cover raster (ESA WorldCover / Dynamic World) |
| Global affine calibration | Per-tile or spatially-varying (thin-plate-spline) calibration for large scenes with heteroscedastic scale error |
| 16-bit heightmap PNG loses precision when canvas-decoded to 8-bit in-browser | Ship a `.exr`/raw float texture + WebGL float-texture path for full-precision picking |
