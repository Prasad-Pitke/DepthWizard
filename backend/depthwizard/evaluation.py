"""
evaluation.py
--------------
Validates a predicted DSM against a reference elevation source
(LiDAR-derived DSM, survey GCPs, or a coarse DEM used as weak ground
truth). Reports the metrics called out in the evaluation criteria:
RMSE, MAE, correlation, and stability across landscape classes
(urban / sparse / hilly / forested), using an optional per-pixel
class map (e.g. from NDVI thresholding, a land-cover raster, or a
segmentation model) to stratify the error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class ValidationReport:
    overall_rmse: float
    overall_mae: float
    overall_corr: float
    n_pixels: int
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self):
        return {
            "overall": {
                "rmse": self.overall_rmse,
                "mae": self.overall_mae,
                "correlation": self.overall_corr,
                "n_pixels": self.n_pixels,
            },
            "per_class": self.per_class,
        }


def validate_dsm(predicted: np.ndarray, reference: np.ndarray,
                  class_map: Optional[np.ndarray] = None,
                  class_names: Optional[Dict[int, str]] = None) -> ValidationReport:
    """
    predicted / reference: HxW float arrays, same grid, meters.
    class_map: optional HxW int array labeling each pixel's landscape
               class (e.g. 0=urban, 1=sparse, 2=hilly, 3=forested).
    class_names: maps class_map integer values to display names.
    """
    valid = np.isfinite(predicted) & np.isfinite(reference)
    p, r = predicted[valid], reference[valid]
    if p.size == 0:
        raise ValueError("No overlapping valid pixels between predicted and reference elevation.")

    residual = p - r
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    corr = float(np.corrcoef(p, r)[0, 1]) if p.size > 1 else float("nan")

    report = ValidationReport(overall_rmse=rmse, overall_mae=mae, overall_corr=corr, n_pixels=int(p.size))

    if class_map is not None:
        names = class_names or {}
        for cls_val in np.unique(class_map[valid]):
            m = valid & (class_map == cls_val)
            if m.sum() < 20:
                continue
            pc, rc = predicted[m], reference[m]
            res_c = pc - rc
            label = names.get(int(cls_val), f"class_{int(cls_val)}")
            report.per_class[label] = {
                "rmse": float(np.sqrt(np.mean(res_c ** 2))),
                "mae": float(np.mean(np.abs(res_c))),
                "correlation": float(np.corrcoef(pc, rc)[0, 1]) if pc.size > 1 else float("nan"),
                "n_pixels": int(m.sum()),
            }
    return report


def synthesize_landscape_classmap(rgb: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """Cheap heuristic landscape stratifier for demo/CI purposes, used
    only when no real land-cover raster is supplied: buckets pixels
    into urban / sparse / hilly / forested using color + local relief
    statistics. In production this should be replaced by an actual
    land-cover classification (e.g. ESA WorldCover) for a fair,
    reproducible stratified evaluation.
    """
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    greenness = hsv[..., 0]
    green_mask = (greenness > 35) & (greenness < 95) & (hsv[..., 1] > 40)

    local_std = cv2.blur(gray, (15, 15))
    local_var = cv2.blur(gray ** 2, (15, 15)) - local_std ** 2
    high_freq = local_var > np.percentile(local_var, 70)

    grad = np.abs(cv2.Sobel(elevation, cv2.CV_32F, 1, 1, ksize=5))
    hilly = grad > np.percentile(grad, 70)

    cls = np.zeros(rgb.shape[:2], dtype=np.int32)
    cls[:] = 1                          # 1 = sparse (default/background)
    cls[green_mask] = 3                 # 3 = forested / vegetated
    cls[(~green_mask) & high_freq] = 0  # 0 = urban (non-green, high-frequency texture)
    cls[hilly & (~green_mask)] = 2      # 2 = hilly / bare terrain relief
    return cls


CLASS_NAMES = {0: "urban", 1: "sparse", 2: "hilly", 3: "forested"}
