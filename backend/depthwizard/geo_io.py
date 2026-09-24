"""
geo_io.py
----------
Image + geospatial I/O helpers.

Handles:
  * Loading PNG/JPG (no georeference) and GeoTIFF (with georeference)
    into a common in-memory representation (`LoadedImage`).
  * Writing the resulting DSM/rDSM back out as a GeoTIFF (for
    georeferenced inputs; carries the original CRS/transform) or a
    plain single-band float TIFF / 16-bit PNG heightmap (for
    non-georeferenced inputs).
  * Resampling an external coarse reference DEM (e.g. an SRTM tile
    the user supplies locally) onto the input image's grid for use by
    the calibration module.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("depthwizard.geo_io")

GEOTIFF_EXTS = {".tif", ".tiff"}
PLAIN_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class LoadedImage:
    rgb: np.ndarray                 # HxWx3 uint8
    is_georeferenced: bool
    crs: Optional[str] = None
    transform: Optional[object] = None   # affine.Affine, if georeferenced
    width: int = 0
    height: int = 0
    source_path: str = ""


def load_image(path: str) -> LoadedImage:
    ext = os.path.splitext(path)[1].lower()
    if ext in GEOTIFF_EXTS:
        return _load_geotiff(path)
    elif ext in PLAIN_EXTS:
        return _load_plain(path)
    else:
        raise ValueError(f"Unsupported input extension '{ext}'. Use PNG/JPG or GeoTIFF.")


def _load_plain(path: str) -> LoadedImage:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return LoadedImage(rgb=arr, is_georeferenced=False,
                        width=arr.shape[1], height=arr.shape[0], source_path=path)


def _load_geotiff(path: str) -> LoadedImage:
    import rasterio
    with rasterio.open(path) as src:
        has_geo = src.crs is not None and src.transform is not None and not src.transform.is_identity
        bands = src.read()  # (bands, H, W)
        if bands.shape[0] >= 3:
            rgb = np.moveaxis(bands[:3], 0, -1)
        else:
            # single-band panchromatic -> replicate to pseudo-RGB
            rgb = np.repeat(bands[0][:, :, None], 3, axis=2)

        rgb = _to_uint8(rgb)
        return LoadedImage(
            rgb=rgb,
            is_georeferenced=bool(has_geo),
            crs=str(src.crs) if src.crs else None,
            transform=src.transform if has_geo else None,
            width=src.width, height=src.height, source_path=path,
        )


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi <= lo:
        lo, hi = arr.min(), arr.max() + 1e-6
    arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0
    return arr.astype(np.uint8)


def resample_reference_dem(dem_path: str, target: LoadedImage) -> np.ndarray:
    """Reproject+resample a reference DEM (e.g. a local SRTM GeoTIFF tile)
    onto the exact pixel grid of `target`. Requires `target` to be
    georeferenced."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    if not target.is_georeferenced:
        raise ValueError("Cannot resample a reference DEM onto a non-georeferenced image.")

    dest = np.full((target.height, target.width), np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as dem_src:
        reproject(
            source=rasterio.band(dem_src, 1),
            destination=dest,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=target.transform,
            dst_crs=target.crs,
            resampling=Resampling.bilinear,
            src_nodata=dem_src.nodata,
            dst_nodata=np.nan,
        )
    return dest


def write_dsm_geotiff(path: str, elevation: np.ndarray, ref: LoadedImage):
    """Write elevation as a single-band float32 GeoTIFF, carrying the
    source image's CRS/transform when available."""
    import rasterio
    from rasterio.crs import CRS

    profile = {
        "driver": "GTiff",
        "height": elevation.shape[0],
        "width": elevation.shape[1],
        "count": 1,
        "dtype": "float32",
        "compress": "lzw",
        "nodata": np.nan,
    }
    if ref.is_georeferenced:
        profile["crs"] = CRS.from_string(ref.crs)
        profile["transform"] = ref.transform
    else:
        # No real-world CRS: use a local unit "pixel" grid so the file
        # is still a valid, openable GeoTIFF (EPSG:3857-style local grid).
        from affine import Affine
        profile["crs"] = CRS.from_epsg(3857)
        profile["transform"] = Affine.translation(0, 0) @ Affine.scale(1, -1)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(elevation.astype(np.float32), 1)


def write_heightmap_png16(path: str, elevation: np.ndarray) -> dict:
    """Write a normalized 16-bit grayscale PNG heightmap for the web
    viewer (Three.js reads this as a displacement texture) and return
    the {min, max} used for normalization so the frontend can rescale
    back to real units."""
    lo, hi = float(np.nanmin(elevation)), float(np.nanmax(elevation))
    span = (hi - lo) if hi > lo else 1.0
    norm = np.nan_to_num((elevation - lo) / span, nan=0.0)
    arr16 = (np.clip(norm, 0, 1) * 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img16 = Image.new("I;16", (arr16.shape[1], arr16.shape[0]))
    img16.frombytes(arr16.tobytes())
    img16.save(path)
    return {"min": lo, "max": hi}


def write_preview_images(depth_map: np.ndarray, elevation: np.ndarray, out_dir: str) -> dict:
    """Writes human-inspectable 8-bit preview PNGs so a user can sanity
    -check the intermediate depth map and the final (possibly
    calibrated) height map *before* they're turned into a 3D mesh:

      depth_map_preview.png    grayscale, raw backbone output (near=dark)
      height_map_preview.png   grayscale, final elevation (low=dark, high=bright)
      height_map_colormap.png  hypsometric-tinted version of the above

    Uses robust percentile stretching (not raw min/max) so a handful
    of outlier pixels don't wash out the whole preview.
    """
    import cv2

    os.makedirs(out_dir, exist_ok=True)

    def robust_u8(a: np.ndarray) -> np.ndarray:
        finite = a[np.isfinite(a)]
        lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
        if hi <= lo:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a)) if np.nanmax(a) > np.nanmin(a) else lo + 1.0
        norm = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1)
        return (norm * 255).astype(np.uint8)

    depth_u8 = robust_u8(depth_map)
    height_u8 = robust_u8(elevation)

    cv2.imwrite(os.path.join(out_dir, "depth_map_preview.png"), depth_u8)
    cv2.imwrite(os.path.join(out_dir, "height_map_preview.png"), height_u8)
    colormap = cv2.applyColorMap(height_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(out_dir, "height_map_colormap.png"), colormap)

    return {
        "depth_map_preview": "depth_map_preview.png",
        "height_map_preview": "height_map_preview.png",
        "height_map_colormap": "height_map_colormap.png",
    }
