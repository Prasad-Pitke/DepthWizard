"""
cli.py
-------
Command-line entry point.

Examples:
    python -m depthwizard.cli --input scene.png --output out/
    python -m depthwizard.cli --input scene.tif --output out/ --reference-dem srtm_tile.tif
    python -m depthwizard.cli --input scene.tif --output out/ --gcps gcps.csv
    python -m depthwizard.cli --input out/dsm.tif --validate lidar_ref.tif --rgb scene.tif
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys

from .pipeline import DepthWizardPipeline


def _load_gcps_csv(path):
    gcps = []
    with open(path) as f:
        reader = csv.reader(f)
        header_skipped = False
        for row in reader:
            if not row:
                continue
            try:
                r, c, e = int(row[0]), int(row[1]), float(row[2])
            except ValueError:
                if not header_skipped:
                    header_skipped = True
                    continue
                raise
            gcps.append((r, c, e))
    return gcps


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="DepthWizard: single-view height estimation + 3D flythrough")
    p.add_argument("--input", required=True, help="Input image: PNG, JPG, or GeoTIFF")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--backbone", default="auto",
                    help="auto | classical | depth_anything_v2_small | dpt_hybrid | dpt_large | midas_small | midas_hybrid")
    p.add_argument("--reference-dem", default=None, help="Local coarse reference DEM GeoTIFF (e.g. SRTM tile)")
    p.add_argument("--gcps", default=None, help="CSV of row,col,elevation_m ground control points")
    p.add_argument("--pixel-size-m", type=float, default=None, help="Override ground sample distance (m/px)")
    p.add_argument("--vertical-exaggeration", type=float, default=1.0)
    p.add_argument("--mesh-resolution", type=int, default=512)
    p.add_argument("--mode", default="relief", choices=["relief", "buildings"],
                   help="'relief': continuous per-pixel heightfield (default, good for terrain/hills). "
                        "'buildings': detect building footprints and extrude clean boxes using a "
                        "shadow-length height proxy (better for dense urban rooftop scenes where "
                        "'relief' produces spiky artifacts — see docs/ARCHITECTURE.md).")
    p.add_argument("--building-min-height-m", type=float, default=4.0)
    p.add_argument("--building-max-height-m", type=float, default=25.0)
    p.add_argument("--building-sun-elevation-deg", type=float, default=None,
                   help="If known (with --pixel-size-m), converts shadow length to an absolute "
                        "height estimate instead of a relative ranking.")
    p.add_argument("--building-min-area-px", type=int, default=60,
                   help="Reject candidate footprints smaller than this (noise/fragment suppression).")
    p.add_argument("--building-max-area-frac", type=float, default=0.15,
                   help="Reject candidates larger than this fraction of the image area (likely open ground, not a single building).")
    p.add_argument("--building-min-extent", type=float, default=0.35,
                   help="Min (contour area / oriented-rect area). Raise to demand more rectangular shapes; "
                        "lower if real buildings are being rejected as 'not rectangular enough'.")
    p.add_argument("--building-max-aspect-ratio", type=float, default=6.0,
                   help="Max oriented long-side/short-side ratio. Real elongated buildings (warehouses, "
                        "hangars) need this fairly high; min-width-px is the primary defense against roads/stripes.")
    p.add_argument("--building-min-width-px", type=float, default=6.0,
                   help="Min oriented-rect short side, in pixels. Primary filter against thin road/stripe false positives.")
    p.add_argument("--building-min-separation-px", type=int, default=12,
                   help="Watershed marker spacing — roughly the minimum gap (in px) needed to split two "
                        "adjacent buildings into separate footprints. Lower = more aggressive splitting "
                        "(risk of over-segmenting a single building); higher = more conservative (risk of "
                        "merging adjacent buildings). Tune this first if buildings in a tight row aren't "
                        "being separated, or if a single building is being split into multiple pieces.")
    p.add_argument("--no-mesh", action="store_true", help="Skip GLB mesh export (viewer bundle still produced)")
    p.add_argument("--validate", default=None, help="Reference elevation GeoTIFF to validate --input DSM against")
    args = p.parse_args(argv)

    if args.validate:
        pipeline = DepthWizardPipeline(backbone="classical")  # backbone unused for validation-only mode
        import rasterio
        rgb = None
        report = pipeline.validate(args.input, args.validate, rgb_for_stratification=rgb)
        print(report.to_dict())
        return

    gcps = _load_gcps_csv(args.gcps) if args.gcps else None
    from .mesh import MeshBuildConfig
    pipeline = DepthWizardPipeline(backbone=args.backbone,
                                    mesh_config=MeshBuildConfig(max_mesh_resolution=args.mesh_resolution))
    result = pipeline.run(
        image_path=args.input, output_dir=args.output,
        reference_dem_path=args.reference_dem, gcps=gcps,
        pixel_size_m=args.pixel_size_m, export_mesh=not args.no_mesh,
        vertical_exaggeration=args.vertical_exaggeration,
        reconstruction_mode=args.mode,
        building_min_height_m=args.building_min_height_m,
        building_max_height_m=args.building_max_height_m,
        building_sun_elevation_deg=args.building_sun_elevation_deg,
        building_min_area_px=args.building_min_area_px,
        building_max_area_frac=args.building_max_area_frac,
        building_min_extent=args.building_min_extent,
        building_max_aspect_ratio=args.building_max_aspect_ratio,
        building_min_width_px=args.building_min_width_px,
        building_min_separation_px=args.building_min_separation_px,
    )
    print(result.to_dict())


if __name__ == "__main__":
    sys.exit(main())
