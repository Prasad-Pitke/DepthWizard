"""
validate_on_gamus.py
----------------------
Evaluates a DepthWizard depth backbone against the GAMUS benchmark's
real LiDAR-derived height ground truth (AGL), using the same
RMSE/MAE/correlation + landscape-class-stratified reporting as
`depthwizard.evaluation`, so results here are directly comparable to
`docs/EVALUATION.md`.

Evaluation protocol (per tile):
  1. Run the backbone to get a relative depth map for the tile's RGB.
  2. Sample `--n-gcps` random pixels' true AGL values as if they were
     survey/GCP measurements (mimicking the GCP calibration path a
     real deployment would use — GAMUS provides no independent coarse
     DEM to test the other calibration path against).
  3. Fit scale/shift from those sampled points only.
  4. Validate the calibrated height against the REMAINING (held-out)
     pixels' true AGL — this is what "accuracy on unseen ground truth"
     actually means; validating against the same points used for
     calibration would be circular.
  5. Aggregate RMSE/MAE/correlation across all tiles, overall and
     per landscape class (via GAMUS's class raster, if present).

Usage:
    python scripts/validate_on_gamus.py \
        --gamus-root ./gamus_data --split test \
        --backbone classical --n-tiles 50 --n-gcps 40 \
        --output gamus_validation_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from depthwizard.datasets import gamus  # noqa: E402
from depthwizard.depth_model import get_backbone  # noqa: E402
from depthwizard.calibration import ScaleCalibrator  # noqa: E402
from depthwizard.evaluation import validate_dsm  # noqa: E402


def evaluate_tile(tile: gamus.GamusTile, backbone, n_gcps: int, rng: np.random.RandomState):
    depth = backbone.predict(tile.rgb)
    h, w = tile.height_agl.shape

    valid_mask = np.isfinite(tile.height_agl)
    valid_idx = np.argwhere(valid_mask)
    if len(valid_idx) < n_gcps * 3:
        return None  # not enough valid ground truth to both calibrate and validate

    rng.shuffle(valid_idx)
    gcp_idx, holdout_idx = valid_idx[:n_gcps], valid_idx[n_gcps:]

    gcps = [(int(r), int(c), float(tile.height_agl[r, c])) for r, c in gcp_idx]
    calib = ScaleCalibrator().calibrate_with_gcps(depth, gcps)

    holdout_mask = np.zeros((h, w), dtype=bool)
    holdout_mask[holdout_idx[:, 0], holdout_idx[:, 1]] = True

    pred_masked = np.where(holdout_mask, calib.elevation, np.nan)
    truth_masked = np.where(holdout_mask, tile.height_agl, np.nan)

    class_map = None
    if tile.landscape_class is not None:
        class_map = np.where(holdout_mask, tile.landscape_class, -1)

    report = validate_dsm(pred_masked, truth_masked, class_map=class_map,
                           class_names=gamus.LANDSCAPE_CLASS_NAMES)
    return report


def main():
    p = argparse.ArgumentParser(description="Validate a DepthWizard backbone against GAMUS.")
    p.add_argument("--gamus-root", required=True, help="Path to a local GAMUS download")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--backbone", default="classical")
    p.add_argument("--n-tiles", type=int, default=50, help="Number of tiles to sample (0 = all)")
    p.add_argument("--n-gcps", type=int, default=40, help="Simulated GCPs sampled per tile")
    p.add_argument("--city-prefix", default=None, help="e.g. 'DC' to restrict to one city")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="gamus_validation_report.json")
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    backbone = get_backbone(args.backbone)

    all_tile_ids = gamus.list_tile_ids(args.gamus_root, args.split)
    if args.city_prefix:
        all_tile_ids = [t for t in all_tile_ids if t.startswith(args.city_prefix)]
    if args.n_tiles and args.n_tiles < len(all_tile_ids):
        rng.shuffle(all_tile_ids)
        all_tile_ids = all_tile_ids[: args.n_tiles]

    print(f"Evaluating backbone '{backbone.name}' on {len(all_tile_ids)} GAMUS '{args.split}' tiles…")

    all_residuals = []
    per_class_residuals = {}
    n_ok, n_skipped = 0, 0

    for i, tile_id in enumerate(all_tile_ids):
        try:
            tile = gamus.load_tile(args.gamus_root, args.split, tile_id)
            report = evaluate_tile(tile, backbone, args.n_gcps, rng)
        except Exception as e:
            print(f"  [{i+1}/{len(all_tile_ids)}] {tile_id}: SKIPPED ({e})")
            n_skipped += 1
            continue
        if report is None:
            n_skipped += 1
            continue
        n_ok += 1
        # re-derive raw residuals is expensive to thread through; we
        # instead re-aggregate from each tile's summary stats using a
        # pooled-variance-style combination, weighted by pixel count.
        all_residuals.append((report.overall_rmse, report.overall_mae, report.overall_corr, report.n_pixels))
        for cls_name, m in report.per_class.items():
            per_class_residuals.setdefault(cls_name, []).append((m["rmse"], m["mae"], m["correlation"], m["n_pixels"]))
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(all_tile_ids)}] tiles processed…")

    def weighted_agg(rows):
        rmses, maes, corrs, ns = zip(*rows)
        ns = np.array(ns, dtype=np.float64)
        return {
            "rmse": float(np.average(rmses, weights=ns)),
            "mae": float(np.average(maes, weights=ns)),
            "correlation": float(np.average([c for c in corrs if np.isfinite(c)],
                                             weights=[n for c, n in zip(corrs, ns) if np.isfinite(c)])
                                  if any(np.isfinite(corrs)) else float("nan")),
            "n_pixels": int(ns.sum()),
            "n_tiles": len(rows),
        }

    result = {
        "backbone": backbone.name,
        "split": args.split,
        "n_tiles_evaluated": n_ok,
        "n_tiles_skipped": n_skipped,
        "n_gcps_per_tile": args.n_gcps,
        "overall": weighted_agg(all_residuals) if all_residuals else None,
        "per_class": {k: weighted_agg(v) for k, v in per_class_residuals.items()},
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone. {n_ok} tiles evaluated, {n_skipped} skipped.")
    if result["overall"]:
        o = result["overall"]
        print(f"Overall: RMSE={o['rmse']:.2f}m  MAE={o['mae']:.2f}m  Corr={o['correlation']:.3f}  "
              f"({o['n_pixels']:,} px across {o['n_tiles']} tiles)")
        for cls_name, m in result["per_class"].items():
            print(f"  {cls_name:>10}: RMSE={m['rmse']:.2f}m  MAE={m['mae']:.2f}m  Corr={m['correlation']:.3f}  "
                  f"({m['n_pixels']:,} px)")
    print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
