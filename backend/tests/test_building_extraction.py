import numpy as np
import cv2
import pytest

from depthwizard.building_extraction import extract_building_footprints, estimate_heights
from depthwizard.mesh import build_building_block_mesh


@pytest.fixture
def synthetic_buildings_scene():
    rng = np.random.RandomState(3)
    h, w = 200, 300
    img = np.full((h, w, 3), 150, dtype=np.uint8)
    specs = []
    for _ in range(5):
        bw, bh = rng.randint(30, 60), rng.randint(25, 45)
        x, y = rng.randint(10, w - bw - 25), rng.randint(10, h - bh - 25)
        shadow_len = rng.randint(3, 20)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (200, 195, 185), -1)
        cv2.rectangle(img, (x + bw, y + shadow_len), (x + bw + shadow_len, y + bh), (50, 50, 50), -1)
        specs.append((x, y, bw, bh, shadow_len))
    return img, specs


@pytest.fixture
def synthetic_row_scene():
    """A row of 5 adjacent buildings with only a 3px gap between them —
    reproduces the real-world 'row of hangars merges into one blob'
    failure mode on a small, fast, fully-synthetic image."""
    img = np.full((120, 400, 3), 150, dtype=np.uint8)
    x = 20
    for _ in range(5):
        bw = 60
        cv2.rectangle(img, (x, 30), (x + bw, 90), (195, 190, 180), -1)
        x += bw + 3  # 3px gap — tight enough to bridge under morphological closing
    return img


def test_extract_building_footprints_finds_rectangles(synthetic_buildings_scene):
    img, specs = synthetic_buildings_scene
    footprints = extract_building_footprints(img)
    assert len(footprints) >= 1
    for fp in footprints:
        x, y, w, h = fp.bbox
        assert w > 0 and h > 0
        assert fp.area_px > 0
        assert fp.polygon.shape[1] == 2


def test_extract_building_footprints_rejects_thin_roads():
    img = np.full((100, 200, 3), 150, dtype=np.uint8)
    cv2.rectangle(img, (10, 45), (190, 55), (180, 180, 180), -1)  # a thin road-like stripe
    footprints = extract_building_footprints(img, max_aspect_ratio=4.5)
    for fp in footprints:
        x, y, w, h = fp.bbox
        aspect = max(w, h) / max(1, min(w, h))
        assert aspect <= 4.5 + 1e-6


def test_estimate_heights_produces_bounded_values(synthetic_buildings_scene):
    img, specs = synthetic_buildings_scene
    footprints = extract_building_footprints(img)
    footprints = estimate_heights(img, footprints, min_height_m=4.0, max_height_m=35.0)
    for fp in footprints:
        assert 4.0 - 1e-6 <= fp.height_m <= 35.0 + 1e-6


def test_estimate_heights_empty_list_is_noop():
    result = estimate_heights(np.zeros((10, 10, 3), dtype=np.uint8), [])
    assert result == []


def test_build_building_block_mesh_has_ground_and_extrusions(synthetic_buildings_scene):
    img, specs = synthetic_buildings_scene
    footprints = extract_building_footprints(img)
    footprints = estimate_heights(img, footprints, min_height_m=4.0, max_height_m=35.0)
    scene = build_building_block_mesh(img, footprints, horizontal_scale_m=1.0)
    assert "ground" in scene.geometry
    ground = scene.geometry["ground"]
    assert ground.vertices.shape[0] == 4
    assert np.allclose(ground.vertices[:, 1], 0.0)  # Y-up: ground is flat (constant Y), not constant Z
    # the ground plane must keep its optical texture (regression test for
    # the trimesh.util.concatenate() bug that silently downgraded it to
    # flat vertex colors — see mesh.py docstring)
    from trimesh.visual import TextureVisuals
    assert isinstance(ground.visual, TextureVisuals)
    assert ground.visual.material.baseColorTexture is not None

    building_names = [n for n in scene.geometry if n != "ground"]
    if footprints:
        assert len(building_names) > 0
        for n in building_names:
            geom = scene.geometry[n]
            assert geom.vertices[:, 1].max() > 0  # extruded upward along Y, not Z


def test_build_building_block_mesh_with_no_footprints_is_flat_ground():
    img = (np.random.RandomState(0).rand(50, 60, 3) * 255).astype(np.uint8)
    scene = build_building_block_mesh(img, [], horizontal_scale_m=1.0)
    assert list(scene.geometry.keys()) == ["ground"]
    ground = scene.geometry["ground"]
    assert ground.vertices.shape[0] == 4
    assert np.allclose(ground.vertices[:, 1], 0.0)  # Y-up: ground is flat (constant Y), not constant Z


def test_ground_plane_is_horizontal_yup(synthetic_buildings_scene):
    """Regression test for a critical coordinate-system bug: the ground
    plane and building extrusions were authored in a Z-up frame (ground
    lying in the XY plane, extrusion height along Z) while the rest of
    the codebase (build_terrain_mesh, and the Three.js viewer's camera/
    controls/lighting) all assume Y-up. The visible symptom was the
    entire model rendering as a vertical wall/fin when viewed from the
    side, instead of buildings standing on a flat ground plane. This
    test asserts the ground has constant Y (flat, horizontal) and that
    buildings extrude along Y from 0 up to their estimated height, with
    their footprint occupying the X/Z plane like the ground does."""
    img, specs = synthetic_buildings_scene
    footprints = extract_building_footprints(img)
    footprints = estimate_heights(img, footprints, min_height_m=4.0, max_height_m=35.0)
    scene = build_building_block_mesh(img, footprints, horizontal_scale_m=1.0)

    ground = scene.geometry["ground"]
    assert np.allclose(ground.vertices[:, 1], 0.0), "ground plane must be flat in Y (horizontal)"
    assert np.ptp(ground.vertices[:, 0]) > 0 and np.ptp(ground.vertices[:, 2]) > 0, \
        "ground plane must span X and Z (not collapsed onto a vertical plane)"

    for name, geom in scene.geometry.items():
        if name == "ground":
            continue
        assert geom.vertices[:, 1].min() >= -1e-6, "buildings must not extend below the ground (Y=0)"
        assert geom.vertices[:, 1].max() > 0, "buildings must extrude upward along Y, not Z"


def test_extract_building_footprints_rejects_rotated_thin_stripes():
    """Regression test: axis-aligned bounding-box filtering fails to
    reject thin *rotated* slivers (a diagonal stripe can have a
    deceptively square-looking axis-aligned bbox). Filtering must use
    the oriented minAreaRect instead."""
    img = np.full((300, 500, 3), 150, dtype=np.uint8)
    for i in range(6):
        cv2.line(img, (20 + i * 50, 20), (60 + i * 50, 280), (190, 185, 175), 4)
    cv2.rectangle(img, (420, 150), (480, 200), (200, 195, 185), -1)  # one real compact building

    footprints = extract_building_footprints(img)
    assert len(footprints) == 1
    x, y, w, h = footprints[0].bbox
    assert x > 400  # it's the compact rectangle, not one of the diagonal stripes


def test_extract_building_footprints_splits_merged_row(synthetic_row_scene):
    """Regression test: a row of adjacent buildings with only a few
    pixels of gap between them must not merge into one giant
    low-rectangularity blob that then gets rejected wholesale — the
    watershed splitting step must recover individual footprints."""
    footprints = extract_building_footprints(synthetic_row_scene)
    # 5 buildings placed with a 3px gap; expect most/all recovered as
    # separate, reasonably-sized footprints rather than 0 or 1 merged blob
    assert len(footprints) >= 4
    for fp in footprints:
        assert fp.area_px < 3600 * 1.2  # each is one ~60x60 building, not multiple merged


def test_real_world_image_detects_many_buildings_not_just_a_few():
    """Regression test using a real (non-synthetic) satellite photo
    with dense adjacent hangar-style buildings — the original failure
    mode this fix targets. Before the watershed-splitting fix, entire
    rows of adjacent buildings merged into a handful of giant
    low-extent blobs that were then rejected, leaving only a few tiny
    noise-sized footprints detected."""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data", "sample_dubai_complex.jpg")
    if not os.path.isfile(path):
        pytest.skip("sample_dubai_complex.jpg not present in this checkout")
    rgb = np.array(__import__("PIL.Image", fromlist=["Image"]).open(path).convert("RGB"))
    footprints = extract_building_footprints(rgb)
    assert len(footprints) > 50  # was effectively ~2 usable detections before the fix
    areas = [fp.area_px for fp in footprints]
    assert max(areas) < 0.1 * rgb.shape[0] * rgb.shape[1]  # nothing merged into a giant blob
