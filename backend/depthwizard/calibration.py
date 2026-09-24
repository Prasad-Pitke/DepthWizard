"""
calibration.py
----------------
Converts a scale-and-shift-ambiguous relative depth/relief map into
absolute metric elevation (a DSM), for georeferenced inputs.

Foundation depth models predict depth only up to an unknown affine
transform:  true_depth ≈ a * predicted_depth + b   (locally, ideally
globally). DepthWizard estimates (a, b) — or a smooth per-pixel
correction field for terrain with heteroscedastic scale error — using
whichever reference elevation source is available:

  * A coarse DEM (e.g. SRTM 30m/90m) resampled onto the image grid:
    used as weak, low-frequency supervision. We solve a robust
    least-squares fit of the *low-pass* component of the relative
    depth to the coarse DEM, then add back the model's high-frequency
    relative structure (the fine detail SRTM cannot resolve) scaled by
    the same factor. This is the classic "guided/global registration"
    strategy used for merging low-res absolute DEMs with high-res
    relative depth from vision models.

  * A sparse set of Ground Control Points (GCPs): (row, col, elevation)
    triples, e.g. surveyed points or LiDAR spot heights. A robust
    (RANSAC + least-squares) affine or low-order-polynomial fit maps
    predicted-depth -> elevation.

  * If neither is available, elevations are left in *relative* units
    (rDSM), scaled to a plausible 0-height_hint range (still useful for
    visualization, not for absolute measurement).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

logger = logging.getLogger("depthwizard.calibration")


@dataclass
class CalibrationResult:
    elevation: np.ndarray          # HxW float32, meters (or relative units if uncalibrated)
    is_absolute: bool
    method: str
    scale: float
    shift: float
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    n_points: Optional[int] = None
    notes: str = ""


def relative_to_height(depth_map: np.ndarray) -> np.ndarray:
    """Invert a depth map (far=large) into an unscaled height map
    (up=large), min-max normalized to [0, 1]."""
    h = depth_map.max() - depth_map
    h = h - h.min()
    m = h.max()
    return h / m if m > 1e-8 else h


def _robust_affine_fit(x: np.ndarray, y: np.ndarray, n_iter: int = 3
                        ) -> Tuple[float, float, np.ndarray]:
    """Iteratively re-weighted least squares affine fit y ≈ a*x + b,
    downweighting outliers each iteration (cheap RANSAC substitute)."""
    weights = np.ones_like(x)
    a, b = 1.0, 0.0
    for _ in range(n_iter):
        def resid(p):
            return weights * (p[0] * x + p[1] - y)
        res = least_squares(resid, x0=[a, b], loss="soft_l1", f_scale=np.std(y) * 0.5 + 1e-6)
        a, b = res.x
        err = np.abs(a * x + b - y)
        mad = np.median(np.abs(err - np.median(err))) + 1e-6
        weights = 1.0 / (1.0 + (err / (3 * 1.4826 * mad)) ** 2)
    return a, b, weights


class ScaleCalibrator:
    """Fits relative-depth -> absolute-elevation mappings."""

    def calibrate_with_coarse_dem(
        self,
        depth_map: np.ndarray,
        coarse_dem: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        low_freq_sigma: float = 25.0,
    ) -> CalibrationResult:
        """
        depth_map:  HxW relative depth (far=large), same grid as coarse_dem
        coarse_dem: HxW absolute elevation resampled/reprojected onto the
                    image grid (e.g. bilinear-resampled SRTM tile).
        """
        rel_height = relative_to_height(depth_map)
        if valid_mask is None:
            valid_mask = np.isfinite(coarse_dem)

        # Split relative height into low-frequency (comparable to what
        # a 30-90m DEM can resolve) and high-frequency (fine structure
        # only the vision model captures) components.
        low_freq = ndimage.gaussian_filter(rel_height, sigma=low_freq_sigma)
        high_freq = rel_height - low_freq

        x = low_freq[valid_mask].ravel()
        y = coarse_dem[valid_mask].ravel()
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size < 50:
            raise ValueError("Not enough valid coarse-DEM samples to calibrate scale.")

        a, b, weights = _robust_affine_fit(x, y)

        # Absolute elevation = affine-fit low-freq term (anchored to the
        # coarse DEM's absolute datum) + same-scale high-freq detail.
        elevation = a * low_freq + b + a * high_freq

        pred_on_samples = a * x + b
        residual = pred_on_samples - y
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        mae = float(np.mean(np.abs(residual)))
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-9
        r2 = float(1 - ss_res / ss_tot)

        return CalibrationResult(
            elevation=elevation.astype(np.float32),
            is_absolute=True,
            method="coarse_dem_affine",
            scale=float(a), shift=float(b),
            rmse=rmse, mae=mae, r2=r2, n_points=int(x.size),
            notes="Low-frequency component anchored to coarse reference DEM; "
                  "high-frequency detail from the vision backbone scaled by the same factor.",
        )

    def calibrate_with_gcps(
        self,
        depth_map: np.ndarray,
        gcps: Sequence[Tuple[int, int, float]],
    ) -> CalibrationResult:
        """
        gcps: list of (row, col, elevation_meters) ground control points.
        Requires >= 4 points; recommends >= 15-20 well-distributed points
        for a stable fit.
        """
        if len(gcps) < 4:
            raise ValueError("At least 4 GCPs are required for scale calibration; "
                              "15-20 well-distributed points is recommended for stability.")
        rel_height = relative_to_height(depth_map)
        rows = np.array([g[0] for g in gcps])
        cols = np.array([g[1] for g in gcps])
        elevs = np.array([g[2] for g in gcps], dtype=np.float64)
        preds = rel_height[rows, cols]

        a, b, weights = _robust_affine_fit(preds, elevs)
        elevation = a * rel_height + b

        fit_pred = a * preds + b
        residual = fit_pred - elevs
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        mae = float(np.mean(np.abs(residual)))
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((elevs - elevs.mean()) ** 2) + 1e-9
        r2 = float(1 - ss_res / ss_tot)

        return CalibrationResult(
            elevation=elevation.astype(np.float32),
            is_absolute=True,
            method="gcp_affine",
            scale=float(a), shift=float(b),
            rmse=rmse, mae=mae, r2=r2, n_points=len(gcps),
            notes=f"Global affine fit from {len(gcps)} ground control points "
                  "(robust IRLS, soft-L1 loss).",
        )

    def uncalibrated(self, depth_map: np.ndarray, height_hint_m: float = 50.0) -> CalibrationResult:
        """No reference elevation available: return a relative DSM (rDSM),
        rescaled into a plausible relief range purely for visualization."""
        rel_height = relative_to_height(depth_map) * height_hint_m
        return CalibrationResult(
            elevation=rel_height.astype(np.float32),
            is_absolute=False,
            method="relative_only",
            scale=float(height_hint_m), shift=0.0,
            notes="No georeference/DEM/GCPs supplied; output is a relative DSM (rDSM) "
                  "scaled to a nominal relief range for visualization only.",
        )
