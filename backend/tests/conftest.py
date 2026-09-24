import os
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_png(tmp_path):
    h, w = 128, 128
    rng = np.random.RandomState(0)
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    img += (rng.randn(h, w, 1) * 5).astype(np.uint8)
    for _ in range(10):
        x, y = rng.randint(10, w - 25), rng.randint(10, h - 25)
        sz = rng.randint(8, 18)
        img[y:y + sz, x:x + sz] = [190, 180, 160]
    path = os.path.join(tmp_path, "scene.png")
    Image.fromarray(img).save(path)
    return path


@pytest.fixture
def tmp_geotiff_with_dem(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    h, w = 128, 128
    rng = np.random.RandomState(1)
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    true_elev = 50 + 8 * np.sin(xx / 25) + 6 * np.cos(yy / 20)

    transform = from_origin(400000, 4400000, 1.0, 1.0)
    img_path = os.path.join(tmp_path, "scene.tif")
    with rasterio.open(img_path, "w", driver="GTiff", height=h, width=w, count=3,
                        dtype="uint8", crs="EPSG:32633", transform=transform) as dst:
        for b in range(3):
            dst.write(img[:, :, b], b + 1)

    import cv2
    coarse = cv2.resize(cv2.resize(true_elev.astype(np.float32), (16, 16)), (w, h))
    dem_path = os.path.join(tmp_path, "dem.tif")
    with rasterio.open(dem_path, "w", driver="GTiff", height=h, width=w, count=1,
                        dtype="float32", crs="EPSG:32633", transform=transform) as dst:
        dst.write(coarse, 1)

    truth_path = os.path.join(tmp_path, "truth.tif")
    with rasterio.open(truth_path, "w", driver="GTiff", height=h, width=w, count=1,
                        dtype="float32", crs="EPSG:32633", transform=transform) as dst:
        dst.write(true_elev.astype(np.float32), 1)

    return {"image": img_path, "dem": dem_path, "truth": truth_path, "shape": (h, w)}
