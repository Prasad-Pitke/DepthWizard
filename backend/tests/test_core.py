import numpy as np
import pytest

from depthwizard.depth_model import ClassicalReliefBackbone, get_backbone
from depthwizard.calibration import ScaleCalibrator, relative_to_height
from depthwizard import geo_io


def test_classical_backbone_shape_and_range(tmp_png):
    loaded = geo_io.load_image(tmp_png)
    backbone = ClassicalReliefBackbone()
    depth = backbone.predict(loaded.rgb)
    assert depth.shape == loaded.rgb.shape[:2]
    assert np.isfinite(depth).all()
    assert depth.max() > depth.min()


def test_get_backbone_falls_back_to_classical_when_torch_missing():
    backbone = get_backbone("auto")
    assert backbone.name in ("classical", "depth_anything_v2_small")


def test_custom_model_backbone_never_imports_torch(tmp_png, monkeypatch):
    """Regression test: the whole point of the custom-callable backbone
    path is to work on a machine where torch is unavailable or broken.
    predict() must not import torch when backend == 'custom'."""
    import builtins
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("predict() must not import torch on the custom-callable path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    def fake_model(rgb):
        return np.ones(rgb.shape[:2], dtype=np.float32)

    backbone = get_backbone("auto", custom_model=fake_model)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    out = backbone.predict(rgb)
    assert out.shape == (16, 16)


def test_relative_to_height_normalizes_and_inverts():
    depth = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)  # far=large
    h = relative_to_height(depth)
    assert h.min() == pytest.approx(0.0)
    assert h.max() == pytest.approx(1.0)
    # depth=0 (closest) should map to the highest relative height
    assert h[0, 0] == 1.0


def test_uncalibrated_returns_relative_dsm():
    depth = np.random.RandomState(0).rand(32, 32).astype(np.float32)
    calib = ScaleCalibrator().uncalibrated(depth, height_hint_m=20.0)
    assert not calib.is_absolute
    assert calib.elevation.max() <= 20.0 + 1e-3
    assert calib.elevation.min() >= 0.0


def test_gcp_calibration_recovers_affine_scale():
    # synthetic relative depth that is an exact affine function of a
    # known height field, verify the fitted (a,b) match closely
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w]
    true_height = (xx + yy).astype(np.float32)
    a_true, b_true = 2.5, 10.0
    true_elev = a_true * true_height + b_true
    depth = true_height.max() - true_height  # convert to depth convention

    rng = np.random.RandomState(2)
    rows = rng.randint(0, h, 40)
    cols = rng.randint(0, w, 40)
    gcps = [(int(r), int(c), float(true_elev[r, c])) for r, c in zip(rows, cols)]

    calib = ScaleCalibrator().calibrate_with_gcps(depth, gcps)
    assert calib.is_absolute
    assert calib.rmse < 1.0  # near-perfect synthetic recovery


def test_coarse_dem_calibration_runs_and_reports_metrics(tmp_geotiff_with_dem):
    fx = tmp_geotiff_with_dem
    loaded = geo_io.load_image(fx["image"])
    coarse = geo_io.resample_reference_dem(fx["dem"], loaded)
    backbone = ClassicalReliefBackbone()
    depth = backbone.predict(loaded.rgb)
    calib = ScaleCalibrator().calibrate_with_coarse_dem(depth, coarse)
    assert calib.is_absolute
    assert calib.rmse is not None and calib.rmse >= 0
    assert calib.elevation.shape == fx["shape"]


def test_geotiff_roundtrip_write_read(tmp_geotiff_with_dem, tmp_path):
    fx = tmp_geotiff_with_dem
    loaded = geo_io.load_image(fx["image"])
    elevation = np.random.RandomState(3).rand(*fx["shape"]).astype(np.float32) * 100
    out_path = str(tmp_path / "dsm.tif")
    geo_io.write_dsm_geotiff(out_path, elevation, loaded)

    reloaded = geo_io.load_image(fx["image"])  # sanity: original still loads
    import rasterio
    with rasterio.open(out_path) as ds:
        arr = ds.read(1)
        assert arr.shape == fx["shape"]
        assert str(ds.crs) == loaded.crs


def test_non_georeferenced_png_detected_correctly(tmp_png):
    loaded = geo_io.load_image(tmp_png)
    assert loaded.is_georeferenced is False
    assert loaded.rgb.shape[2] == 3
