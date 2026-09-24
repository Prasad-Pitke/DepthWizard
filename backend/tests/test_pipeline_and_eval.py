import os
import numpy as np
import pytest

from depthwizard.pipeline import DepthWizardPipeline
from depthwizard.mesh import MeshBuildConfig, build_terrain_mesh, export_glb, export_viewer_bundle
from depthwizard.evaluation import validate_dsm, synthesize_landscape_classmap, CLASS_NAMES


def test_pipeline_end_to_end_relative(tmp_png, tmp_path):
    out_dir = str(tmp_path / "out")
    pipeline = DepthWizardPipeline(backbone="classical", mesh_config=MeshBuildConfig(max_mesh_resolution=64))
    result = pipeline.run(image_path=tmp_png, output_dir=out_dir)
    assert os.path.isfile(result.dsm_path)
    assert os.path.isfile(result.glb_path)
    assert os.path.isdir(result.viewer_bundle_dir)
    assert not result.is_absolute
    assert result.reconstruction_mode == "relief"
    assert os.path.isfile(os.path.join(result.viewer_bundle_dir, "meta.json"))
    assert os.path.isfile(os.path.join(result.viewer_bundle_dir, "heightmap.png"))
    assert os.path.isfile(os.path.join(result.viewer_bundle_dir, "texture.jpg"))


def test_pipeline_buildings_mode(tmp_path):
    import numpy as np
    import cv2
    rng = np.random.RandomState(5)
    h, w = 200, 300
    img = np.full((h, w, 3), 150, dtype=np.uint8)
    for _ in range(5):
        bw, bh = rng.randint(30, 60), rng.randint(25, 45)
        x, y = rng.randint(10, w - bw - 20), rng.randint(10, h - bh - 20)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (200, 195, 185), -1)
        cv2.rectangle(img, (x + bw, y + 8), (x + bw + 10, y + bh), (50, 50, 50), -1)
    img_path = str(tmp_path / "buildings.png")
    from PIL import Image
    Image.fromarray(img).save(img_path)

    out_dir = str(tmp_path / "out")
    pipeline = DepthWizardPipeline(backbone="classical")
    result = pipeline.run(image_path=img_path, output_dir=out_dir, reconstruction_mode="buildings")
    assert result.reconstruction_mode == "buildings"
    assert result.n_buildings_detected is not None
    assert os.path.isfile(result.glb_path)
    assert os.path.isfile(result.buildings_path)

    import trimesh
    mesh = trimesh.load(result.glb_path)
    assert mesh is not None


def test_pipeline_invalid_reconstruction_mode_raises(tmp_png, tmp_path):
    pipeline = DepthWizardPipeline(backbone="classical")
    with pytest.raises(ValueError):
        pipeline.run(image_path=tmp_png, output_dir=str(tmp_path / "out"), reconstruction_mode="bogus")


def test_pipeline_end_to_end_absolute_with_dem(tmp_geotiff_with_dem, tmp_path):
    fx = tmp_geotiff_with_dem
    out_dir = str(tmp_path / "out")
    pipeline = DepthWizardPipeline(backbone="classical", mesh_config=MeshBuildConfig(max_mesh_resolution=64))
    result = pipeline.run(image_path=fx["image"], output_dir=out_dir, reference_dem_path=fx["dem"])
    assert result.is_absolute
    assert result.calibration_method == "coarse_dem_affine"
    assert result.calibration_rmse is not None


def test_pipeline_validate_against_reference(tmp_geotiff_with_dem, tmp_path):
    fx = tmp_geotiff_with_dem
    out_dir = str(tmp_path / "out")
    pipeline = DepthWizardPipeline(backbone="classical", mesh_config=MeshBuildConfig(max_mesh_resolution=64))
    pipeline.run(image_path=fx["image"], output_dir=out_dir, reference_dem_path=fx["dem"])

    import depthwizard.geo_io as geo_io
    loaded = geo_io.load_image(fx["image"])
    report = pipeline.validate(os.path.join(out_dir, "dsm.tif"), fx["truth"], rgb_for_stratification=loaded.rgb)
    d = report.to_dict()
    assert "overall" in d
    assert d["overall"]["rmse"] >= 0
    assert set(d["per_class"].keys()).issubset(set(CLASS_NAMES.values()))


def test_mesh_build_and_glb_export(tmp_path):
    h, w = 32, 40
    elevation = (np.sin(np.linspace(0, 6, h))[:, None] + np.cos(np.linspace(0, 6, w))[None, :]).astype(np.float32)
    texture = (np.random.RandomState(0).rand(h, w, 3) * 255).astype(np.uint8)
    mesh = build_terrain_mesh(elevation, texture, MeshBuildConfig(max_mesh_resolution=40))
    assert mesh.vertices.shape[0] == h * w
    assert mesh.faces.shape[0] == (h - 1) * (w - 1) * 2

    out_path = str(tmp_path / "terrain.glb")
    export_glb(mesh, out_path)
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0

    # round-trip load
    import trimesh
    reloaded = trimesh.load(out_path)
    assert reloaded is not None


def test_viewer_bundle_export(tmp_path):
    h, w = 32, 32
    elevation = np.linspace(0, 100, h * w, dtype=np.float32).reshape(h, w)
    texture = (np.random.RandomState(1).rand(h, w, 3) * 255).astype(np.uint8)
    out_dir = str(tmp_path / "bundle")
    meta = export_viewer_bundle(elevation, texture, out_dir, MeshBuildConfig())
    assert meta["elevation_min_m"] == pytest.approx(0.0, abs=1.0)
    assert meta["elevation_max_m"] == pytest.approx(100.0, abs=1.0)
    for fname in ("texture.jpg", "heightmap.png", "meta.json"):
        assert os.path.isfile(os.path.join(out_dir, fname))


def test_validate_dsm_metrics_are_sane():
    rng = np.random.RandomState(0)
    ref = rng.rand(50, 50).astype(np.float32) * 100
    pred = ref + rng.randn(50, 50).astype(np.float32) * 2  # small noise
    report = validate_dsm(pred, ref)
    assert report.overall_rmse < 5
    assert report.overall_corr > 0.9


def test_synthesize_landscape_classmap_returns_expected_labels():
    rgb = (np.random.RandomState(0).rand(40, 40, 3) * 255).astype(np.uint8)
    elevation = np.random.RandomState(0).rand(40, 40).astype(np.float32) * 30
    cls = synthesize_landscape_classmap(rgb, elevation)
    assert cls.shape == (40, 40)
    assert set(np.unique(cls)).issubset(set(CLASS_NAMES.keys()))
