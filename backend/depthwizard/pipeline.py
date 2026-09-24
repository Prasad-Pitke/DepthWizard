"""
pipeline.py
------------
End-to-end orchestration:

    image (PNG/JPG/GeoTIFF)
        -> depth backbone -> relative depth
        -> scale calibration (coarse DEM / GCPs / none)
        -> DSM (GeoTIFF, absolute) or rDSM (relative)
        -> viewer bundle (heightmap + texture + meta) and/or GLB mesh
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from . import geo_io
from .calibration import ScaleCalibrator, CalibrationResult
from .depth_model import get_backbone, DepthBackbone
from .mesh import (MeshBuildConfig, build_terrain_mesh, build_building_block_mesh,
                    export_glb, export_viewer_bundle)
from .evaluation import validate_dsm, synthesize_landscape_classmap, CLASS_NAMES

logger = logging.getLogger("depthwizard.pipeline")


@dataclass
class PipelineOutput:
    output_dir: str
    dsm_path: str
    viewer_bundle_dir: str
    glb_path: Optional[str]
    is_absolute: bool
    calibration_method: str
    calibration_rmse: Optional[float]
    calibration_mae: Optional[float]
    calibration_r2: Optional[float]
    reconstruction_mode: str = "relief"
    n_buildings_detected: Optional[int] = None
    buildings_path: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class DepthWizardPipeline:
    def __init__(self, backbone: str = "auto", backbone_kwargs: Optional[dict] = None,
                 mesh_config: Optional[MeshBuildConfig] = None):
        self.backbone: DepthBackbone = get_backbone(backbone, **(backbone_kwargs or {}))
        self.calibrator = ScaleCalibrator()
        self.mesh_config = mesh_config or MeshBuildConfig()

    def run(
        self,
        image_path: str,
        output_dir: str,
        reference_dem_path: Optional[str] = None,
        gcps: Optional[Sequence[Tuple[int, int, float]]] = None,
        pixel_size_m: Optional[float] = None,
        export_mesh: bool = True,
        vertical_exaggeration: float = 1.0,
        reconstruction_mode: str = "relief",
        building_min_height_m: float = 4.0,
        building_max_height_m: float = 25.0,
        building_sun_elevation_deg: Optional[float] = None,
        building_min_area_px: int = 60,
        building_max_area_frac: float = 0.15,
        building_min_extent: float = 0.35,
        building_max_aspect_ratio: float = 6.0,
        building_min_width_px: float = 6.0,
        building_min_separation_px: int = 12,
    ) -> PipelineOutput:
        if reconstruction_mode not in ("relief", "buildings"):
            raise ValueError(f"reconstruction_mode must be 'relief' or 'buildings', got '{reconstruction_mode}'")

        os.makedirs(output_dir, exist_ok=True)
        logger.info("Loading image: %s", image_path)
        loaded = geo_io.load_image(image_path)

        logger.info("Running depth backbone '%s' ...", self.backbone.name)
        depth_map = self.backbone.predict(loaded.rgb)
        calib = self._calibrate(loaded, depth_map, reference_dem_path, gcps)
        logger.info("Calibration: method=%s absolute=%s rmse=%s",
                    calib.method, calib.is_absolute, calib.rmse)

        # --- Absolute (GeoTIFF) or relative DSM output --------------------
        # Produced regardless of reconstruction_mode: it's a useful,
        # cheap-to-compute diagnostic artifact even when the 3D mesh
        # itself is built via building-block extrusion instead.
        dsm_path = os.path.join(output_dir, "dsm.tif")
        geo_io.write_dsm_geotiff(dsm_path, calib.elevation, loaded)

        with open(os.path.join(output_dir, "calibration_report.json"), "w") as f:
            json.dump({
                "method": calib.method, "is_absolute": calib.is_absolute,
                "scale": calib.scale, "shift": calib.shift,
                "rmse": calib.rmse, "mae": calib.mae, "r2": calib.r2,
                "n_points": calib.n_points, "notes": calib.notes,
            }, f, indent=2)

        # --- Viewer bundle (always produced) -------------------------------
        px_size = pixel_size_m or (self._infer_pixel_size(loaded) if loaded.is_georeferenced else 1.0)
        mesh_cfg = MeshBuildConfig(
            max_mesh_resolution=self.mesh_config.max_mesh_resolution,
            horizontal_scale_m=px_size,
            vertical_exaggeration=vertical_exaggeration,
        )
        viewer_dir = os.path.join(output_dir, "viewer_bundle")
        export_viewer_bundle(calib.elevation, loaded.rgb, viewer_dir, mesh_cfg)
        geo_io.write_preview_images(depth_map, calib.elevation, viewer_dir)

        glb_path = None
        n_buildings = None
        buildings_path = None
        if export_mesh:
            glb_path = os.path.join(output_dir, "terrain.glb")
            if reconstruction_mode == "buildings":
                from . import building_extraction
                logger.info("Detecting building footprints...")
                footprints = building_extraction.extract_building_footprints(
                    loaded.rgb,
                    min_area_px=building_min_area_px, max_area_frac=building_max_area_frac,
                    min_extent=building_min_extent, max_aspect_ratio=building_max_aspect_ratio,
                    min_width_px=building_min_width_px, min_separation_px=building_min_separation_px,
                )
                footprints = building_extraction.estimate_heights(
                    loaded.rgb, footprints,
                    min_height_m=building_min_height_m, max_height_m=building_max_height_m,
                    sun_elevation_deg=building_sun_elevation_deg,
                    pixel_size_m=px_size if building_sun_elevation_deg is not None else None,
                )
                logger.info("Detected %d building footprints", len(footprints))
                mesh = build_building_block_mesh(loaded.rgb, footprints, horizontal_scale_m=px_size)
                n_buildings = len(footprints)
                buildings_path = os.path.join(output_dir, "buildings.json")
                with open(buildings_path, "w") as f:
                    json.dump([{
                        "polygon_px": fp.polygon.tolist(), "bbox_px": fp.bbox,
                        "area_px": fp.area_px, "height_m": fp.height_m,
                        "shadow_signal": fp.shadow_signal,
                    } for fp in footprints], f, indent=2)
            else:
                mesh = build_terrain_mesh(calib.elevation, loaded.rgb, mesh_cfg)
            export_glb(mesh, glb_path)

        return PipelineOutput(
            output_dir=output_dir, dsm_path=dsm_path, viewer_bundle_dir=viewer_dir,
            glb_path=glb_path, is_absolute=calib.is_absolute, calibration_method=calib.method,
            calibration_rmse=calib.rmse, calibration_mae=calib.mae, calibration_r2=calib.r2,
            reconstruction_mode=reconstruction_mode, n_buildings_detected=n_buildings,
            buildings_path=buildings_path,
        )

    def validate(self, dsm_path: str, reference_dem_path: str, rgb_for_stratification: Optional[np.ndarray] = None):
        import rasterio
        from rasterio.warp import reproject, Resampling

        with rasterio.open(dsm_path) as pred_src:
            pred = pred_src.read(1)
            ref = np.full_like(pred, np.nan, dtype=np.float32)
            with rasterio.open(reference_dem_path) as ref_src:
                reproject(
                    source=rasterio.band(ref_src, 1), destination=ref,
                    src_transform=ref_src.transform, src_crs=ref_src.crs,
                    dst_transform=pred_src.transform, dst_crs=pred_src.crs,
                    resampling=Resampling.bilinear, dst_nodata=np.nan,
                )
        class_map = None
        if rgb_for_stratification is not None:
            class_map = synthesize_landscape_classmap(rgb_for_stratification, pred)
        report = validate_dsm(pred, ref, class_map=class_map, class_names=CLASS_NAMES)
        return report

    # -- internals -----------------------------------------------------
    def _calibrate(self, loaded: geo_io.LoadedImage, depth_map: np.ndarray,
                    reference_dem_path: Optional[str], gcps) -> CalibrationResult:
        if gcps:
            return self.calibrator.calibrate_with_gcps(depth_map, gcps)
        if reference_dem_path and loaded.is_georeferenced:
            coarse_dem = geo_io.resample_reference_dem(reference_dem_path, loaded)
            return self.calibrator.calibrate_with_coarse_dem(depth_map, coarse_dem)
        return self.calibrator.uncalibrated(depth_map)

    def _infer_pixel_size(self, loaded: geo_io.LoadedImage) -> float:
        t = loaded.transform
        if t is None:
            return 1.0
        # Affine pixel size (assumes north-up or near-north-up raster)
        return float(np.hypot(t.a, t.b))
