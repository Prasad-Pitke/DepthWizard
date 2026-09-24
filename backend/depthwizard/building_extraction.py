"""
building_extraction.py
------------------------
An alternative reconstruction path for scenes dominated by discrete
rectilinear structures (dense urban rooftops, hangars, warehouses,
tightly-packed housing) — the case where per-pixel continuous relief
estimation (`depth_model.py`) produces visually poor results, because
the true underlying geometry is *discontinuous* (flat roof -> vertical
wall -> flat ground) rather than smoothly varying terrain. See
`docs/ARCHITECTURE.md §5` for the reasoning and its limits.

Pipeline:
  1. `extract_building_footprints()` — segments compact, roughly-
     rectangular, non-elongated bright/uniform blobs from the RGB
     image as building candidates, excluding roads (thin/elongated),
     vegetation (green, high local texture), and open ground.
  2. `estimate_heights()` — for each footprint, samples a classical
     cast-shadow-length proxy in the region immediately adjacent to
     the footprint (reusing the same shadow cue as the continuous
     relief backbone) and maps it to a plausible height via
     percentile-based scaling against a configurable height range.

**This is a heuristic, not a measurement.** Shadow length is a
physically real cue (height = shadow_length * tan(sun_elevation)) but
only if the sun elevation angle and ground sample distance are known;
without them (the common case for an arbitrary uploaded PNG/JPG),
`estimate_heights()` falls back to *relative* shadow-length ranking
rescaled into a user-supplied plausible height band. If you know the
true sun elevation angle and pixel size for your imagery, pass them
to get a physically-grounded absolute estimate instead — see
`estimate_heights(..., sun_elevation_deg=..., pixel_size_m=...)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class BuildingFootprint:
    polygon: np.ndarray       # Nx2 float array, (x, y) in pixel coordinates
    bbox: tuple               # (x, y, w, h) in pixel coordinates
    area_px: float
    height_m: float = 0.0
    shadow_signal: float = 0.0  # raw (uncalibrated) shadow-length proxy, for debugging/inspection


def extract_building_footprints(
    rgb: np.ndarray,
    min_area_px: int = 60,
    max_area_frac: float = 0.15,
    min_extent: float = 0.35,
    max_aspect_ratio: float = 6.0,
    min_width_px: float = 6.0,
    min_separation_px: int = 12,
) -> List[BuildingFootprint]:
    """Detects compact, roughly-rectangular building-like blobs.

    Two-stage approach:
      1. Segment a candidate mask (locally-uniform, non-green,
         non-shadow regions — flat rooftops/pavement).
      2. **Split merged blobs into individual buildings via marker-based
         watershed** (`min_separation_px` controls granularity). This
         step is essential on real imagery: a row of adjacent
         buildings/hangars with only a few pixels of gap between them
         routinely merges into one giant blob under simple thresholding
         + morphological closing, and that merged blob then fails the
         rectangularity checks below (its extent is low because it
         spans multiple buildings plus the gaps between them) — so
         without splitting, an entire row of real buildings would be
         silently dropped rather than each detected individually.

    Filters applied per split region, all computed from the region's
    **oriented** minimum-area rectangle (`cv2.minAreaRect`) rather than
    axis-aligned bounding box (an axis-aligned box can look deceptively
    square for a thin rotated sliver — see the regression test for this):
      - `min_area_px` / `max_area_frac`: drop noise specks and drop
        anything so large it's more likely open ground than a building.
      - `min_extent` (contour area / oriented-rect area): rejects
        irregular/L-shaped blobs.
      - `max_aspect_ratio` (oriented rect long side / short side):
        rejects long thin shapes (roads, parking-lot stripes) at any
        rotation angle. Set higher than you might expect (default 6.0)
        because genuine buildings — warehouses, hangars, terraces — are
        often legitimately elongated; `min_width_px` is what actually
        distinguishes a real elongated building (tens of pixels wide)
        from a road/stripe (a few pixels wide), not the aspect ratio.
      - `min_width_px` (oriented rect short side, in pixels): the
        primary defense against thin slivers regardless of length.

    The returned footprint polygon is always the 4 corners of the
    oriented rectangle (`cv2.boxPoints`), NOT a raw traced contour —
    real building rooftops are approximately rectangular (that's the
    premise of this reconstruction mode), and snapping to the oriented
    rectangle guarantees a clean box shape even when the underlying
    mask is noisy (JPEG blocking artifacts, antenna/vent clutter on a
    roof, etc. previously produced jagged many-vertex polygons that
    extruded into visually broken, near-vertical sliver shapes).
    """
    h, w = rgb.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    grad = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    grad_abs = np.abs(grad)
    uniform_thresh = max(float(np.percentile(grad_abs, 70)), 1e-6)
    uniform = grad_abs <= uniform_thresh
    not_green = ~((hsv[..., 0] > 35) & (hsv[..., 0] < 95) & (hsv[..., 1] > 40))
    not_dark = gray > np.percentile(gray, 20)

    candidate_mask = (uniform & not_green & not_dark).astype(np.uint8) * 255
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    labels = _split_into_instances(candidate_mask, min_separation_px)

    footprints = []
    for lbl in range(1, labels.max() + 1):
        region_mask = (labels == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)  # watershed regions are already single-blob; this is just a safety net

        area = cv2.contourArea(cnt)
        if area < min_area_px or area > max_area_frac * img_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        short_side, long_side = sorted((rw, rh))
        if short_side < min_width_px:
            continue
        rect_area = rw * rh
        if rect_area <= 0:
            continue
        extent = area / rect_area
        if extent < min_extent:
            continue
        aspect = long_side / max(short_side, 1e-6)
        if aspect > max_aspect_ratio:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)  # axis-aligned; kept only for shadow-ring sampling
        polygon = cv2.boxPoints(rect).astype(np.float64)  # clean rectangle corners, not the raw contour

        footprints.append(BuildingFootprint(polygon=polygon, bbox=(x, y, bw, bh), area_px=area))

    return footprints


def _split_into_instances(mask_u8: np.ndarray, min_separation_px: int) -> np.ndarray:
    """Marker-based watershed to split a binary mask where multiple
    adjacent objects (e.g. a row of buildings with thin gaps) have
    merged into single connected components under simple thresholding.
    Returns an int32 label image (0 = background, 1..N = instances).
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    mask_bool = mask_u8 > 0
    if not mask_bool.any():
        return np.zeros(mask_u8.shape, dtype=np.int32)

    dist = ndi.distance_transform_edt(mask_bool)
    coords = peak_local_max(dist, min_distance=min_separation_px, labels=mask_bool)
    if len(coords) == 0:
        # Nothing peaky enough to seed (e.g. one small blob) — treat
        # the whole mask as a single instance rather than dropping it.
        return ndi.label(mask_bool)[0]

    markers = np.zeros(mask_u8.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    return watershed(-dist, markers, mask=mask_bool)


def _shadow_length_proxy_map(rgb: np.ndarray) -> np.ndarray:
    """Cast-shadow extent map, shared logic with
    depth_model.ClassicalReliefBackbone._shadow_height_proxy (kept as
    a standalone function here so building_extraction.py doesn't need
    to instantiate a full depth backbone just for this one cue)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dark_mask = (gray < np.percentile(gray, 25)).astype(np.uint8)
    dist = cv2.distanceTransform(dark_mask, cv2.DIST_L2, 5)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 40, 120)
    edge_dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    proximity_to_edge = np.exp(-edge_dist / 15.0)
    return dist * proximity_to_edge


def estimate_heights(
    rgb: np.ndarray,
    footprints: List[BuildingFootprint],
    min_height_m: float = 4.0,
    max_height_m: float = 35.0,
    sun_elevation_deg: Optional[float] = None,
    pixel_size_m: Optional[float] = None,
    search_radius_px: int = 18,
) -> List[BuildingFootprint]:
    """Fills in `height_m` for each footprint in place (and returns the
    same list) using the shadow-length proxy sampled in a ring just
    outside each footprint.

    If both `sun_elevation_deg` and `pixel_size_m` are supplied, the
    raw shadow-length signal (converted from a proxy distance-transform
    unit into an approximate pixel count) is treated as a real shadow
    length and converted to meters via `height = shadow_px * pixel_size_m
    * tan(sun_elevation_deg)`. This is still approximate — the proxy
    isn't a precise shadow-tip measurement — but it's a physically
    motivated absolute estimate rather than a pure ranking.

    Otherwise (the common case — sun angle/GSD unknown for an arbitrary
    upload), footprints are ranked by their shadow signal and rescaled
    percentile-wise into [min_height_m, max_height_m], with a secondary
    nudge from footprint area (larger buildings skew taller) since
    shadow signal alone is noisy on small blobs.
    """
    if not footprints:
        return footprints

    shadow_map = _shadow_length_proxy_map(rgb)
    h, w = rgb.shape[:2]
    signals = np.zeros(len(footprints), dtype=np.float64)

    for i, fp in enumerate(footprints):
        x, y, bw, bh = fp.bbox
        x0, y0 = max(0, x - search_radius_px), max(0, y - search_radius_px)
        x1, y1 = min(w, x + bw + search_radius_px), min(h, y + bh + search_radius_px)
        footprint_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        shifted_poly = (fp.polygon - [x0, y0]).astype(np.int32)
        cv2.fillPoly(footprint_mask, [shifted_poly], 1)
        dilated = cv2.dilate(footprint_mask, np.ones((search_radius_px, search_radius_px), np.uint8))
        ring = (dilated > 0) & (footprint_mask == 0)

        region_signal = shadow_map[y0:y1, x0:x1]
        # Use max (not a mid/high percentile) because the ring wraps
        # all the way around the footprint but a cast shadow only
        # occupies the down-sun side — a percentile over the full ring
        # is dominated by the shadow-free majority and collapses to 0.
        signals[i] = float(region_signal[ring].max()) if ring.any() else 0.0
        fp.shadow_signal = signals[i]

    if sun_elevation_deg is not None and pixel_size_m is not None:
        sun_rad = np.radians(sun_elevation_deg)
        for fp in footprints:
            shadow_px = fp.shadow_signal  # approximate; see docstring
            fp.height_m = float(np.clip(
                shadow_px * pixel_size_m / max(np.tan(sun_rad), 1e-3), min_height_m, max_height_m
            ))
        return footprints

    # Relative ranking fallback.
    lo, hi = np.percentile(signals, 5), np.percentile(signals, 95)
    span = (hi - lo) if hi > lo else 1.0
    norm_signal = np.clip((signals - lo) / span, 0, 1)

    areas = np.array([fp.area_px for fp in footprints], dtype=np.float64)
    area_lo, area_hi = np.percentile(areas, 5), np.percentile(areas, 95)
    area_span = (area_hi - area_lo) if area_hi > area_lo else 1.0
    norm_area = np.clip((areas - area_lo) / area_span, 0, 1)

    combined = 0.7 * norm_signal + 0.3 * norm_area
    for fp, c in zip(footprints, combined):
        fp.height_m = float(min_height_m + c * (max_height_m - min_height_m))

    return footprints
