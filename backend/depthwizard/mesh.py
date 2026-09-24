"""
mesh.py
--------
Turns an (optical texture, elevation map) pair into 3D assets the
visualization layer can consume:

  * A decimated triangle mesh exported as glTF/GLB (draped with the
    original optical image as a texture) — importable by Unity,
    Babylon.js, or any glTF-compatible engine.
  * A lightweight heightmap+texture pair for the bundled Three.js
    viewer, which displaces a GPU-tessellated plane in a vertex
    shader (cheaper than shipping a huge pre-baked mesh, and lets the
    viewer trade mesh resolution for frame rate live).

Both paths downsample the elevation grid for the *mesh* representation
(full-resolution elevation/texture is preserved separately for
analysis/validation), since naive per-pixel meshing of a multi-
megapixel DSM produces meshes far too dense for real-time navigation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class MeshBuildConfig:
    max_mesh_resolution: int = 512     # grid samples along the longer edge
    horizontal_scale_m: float = 1.0    # meters per pixel (georeferenced) or 1.0 (relative)
    vertical_exaggeration: float = 1.0


def _downsample_grid(elevation: np.ndarray, max_res: int) -> np.ndarray:
    import cv2
    h, w = elevation.shape
    scale = max_res / max(h, w)
    if scale >= 1.0:
        return elevation
    new_w, new_h = max(2, int(w * scale)), max(2, int(h * scale))
    return cv2.resize(elevation, (new_w, new_h), interpolation=cv2.INTER_AREA)


def build_terrain_mesh(elevation: np.ndarray, rgb_texture: np.ndarray,
                        cfg: MeshBuildConfig = MeshBuildConfig()) -> "trimesh.Trimesh":
    """Builds a textured triangle mesh from a heightfield + RGB texture."""
    import trimesh
    from trimesh.visual import TextureVisuals
    from PIL import Image as PILImage

    grid = _downsample_grid(elevation, cfg.max_mesh_resolution)
    h, w = grid.shape
    xs = np.arange(w) * cfg.horizontal_scale_m
    ys = np.arange(h) * cfg.horizontal_scale_m
    xx, yy = np.meshgrid(xs, ys)
    zz = np.nan_to_num(grid, nan=np.nanmin(grid)) * cfg.vertical_exaggeration

    vertices = np.stack([xx, -yy, zz], axis=-1).reshape(-1, 3)  # -y so image-down = world-forward
    uv = np.stack([xx / xs.max() if xs.max() > 0 else xx,
                   1.0 - (yy / ys.max() if ys.max() > 0 else yy)], axis=-1).reshape(-1, 2)

    faces = []
    for r in range(h - 1):
        for c in range(w - 1):
            i0 = r * w + c
            i1 = r * w + (c + 1)
            i2 = (r + 1) * w + c
            i3 = (r + 1) * w + (c + 1)
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])
    faces = np.array(faces, dtype=np.int64)

    tex_img = PILImage.fromarray(rgb_texture)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=tex_img, metallicFactor=0.0, roughnessFactor=1.0)
    visual = TextureVisuals(uv=uv, material=material)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh.fix_normals()
    return mesh


def export_glb(mesh, out_path: str):
    """Exports either a trimesh.Trimesh or a trimesh.Scene to GLB —
    both implement .export(), so this works unchanged for either."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mesh.export(out_path, file_type="glb")


def export_viewer_bundle(elevation: np.ndarray, rgb_texture: np.ndarray, out_dir: str,
                          cfg: MeshBuildConfig = MeshBuildConfig()) -> dict:
    """Writes the lightweight bundle the Three.js viewer expects:
    - texture.jpg   (draped optical image)
    - heightmap.png (16-bit normalized displacement map)
    - meta.json     (min/max elevation, scale, mesh resolution)
    Returns the meta dict.
    """
    import json
    import cv2
    from .geo_io import write_heightmap_png16

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "texture.jpg"),
                cv2.cvtColor(rgb_texture, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    hmap_meta = write_heightmap_png16(os.path.join(out_dir, "heightmap.png"), elevation)

    meta = {
        "elevation_min_m": hmap_meta["min"],
        "elevation_max_m": hmap_meta["max"],
        "horizontal_scale_m": cfg.horizontal_scale_m,
        "vertical_exaggeration": cfg.vertical_exaggeration,
        "width_px": int(elevation.shape[1]),
        "height_px": int(elevation.shape[0]),
        "mesh_resolution": cfg.max_mesh_resolution,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# --------------------------------------------------------------------------
# 3. Building-block reconstruction (footprint extrusion)
# --------------------------------------------------------------------------
def build_building_block_mesh(
    rgb_texture: np.ndarray,
    footprints,
    horizontal_scale_m: float = 1.0,
    ground_texture_downsample: int = 512,
):
    """Builds a flat textured ground plane plus one extruded box per
    detected building footprint (see `building_extraction.py`).

    This is the reconstruction path for scenes where a continuous
    per-pixel heightfield (`build_terrain_mesh`) produces poor results
    because the true geometry is discontinuous — dense rectilinear
    rooftops with essentially no smooth transition to the ground.
    Building tops are draped with the corresponding patch of the
    optical texture (so roof color/markings are preserved); walls are
    given a flat neutral shade since single-view nadir imagery contains
    no facade information to texture them with honestly.

    Returns a `trimesh.Scene`, NOT a single merged `trimesh.Trimesh`.
    This matters: `trimesh.util.concatenate()` silently downgrades a
    mix of `TextureVisuals` (the ground plane) and `ColorVisuals` (the
    building walls) into a single `ColorVisuals` for the whole merged
    mesh — which discards the ground's UV-mapped optical texture
    entirely and renders it as flat gray. A `Scene` keeps each part as
    its own glTF primitive with its own material, which both glTF and
    three.js's GLTFLoader support natively, so the ground texture and
    the building materials all render correctly.
    """
    import trimesh
    from trimesh.creation import extrude_polygon
    from trimesh.visual import TextureVisuals
    from shapely.geometry import Polygon
    from PIL import Image as PILImage

    h, w = rgb_texture.shape[:2]

    # Ground plane: a single flat quad, textured with the full optical
    # image, at world scale. Uses the Y-up convention that the rest of
    # this codebase (build_terrain_mesh, and the Three.js viewer's
    # camera/controls/lighting) all assume: X = right, Y = up,
    # Z = "north" (image row, negated so image-down maps to world-
    # -south, matching build_terrain_mesh's PlaneGeometry rotation).
    # Y is held at 0 for every vertex so the plane lies flat — NOT the
    # column that varies with image row.
    #
    # BUG THIS FIXES: an earlier version put the north-south extent in
    # the Y column instead of Z, with Y=0 only along one edge — which
    # authored the ground plane standing *upright* in a Y-up world
    # rather than lying flat. Every building extrusion inherited the
    # same mistake (trimesh's extrude_polygon extrudes along its own Z,
    # which was never remapped onto world Y either). The visible symptom
    # was the entire model rendering as a vertical wall/fin when viewed
    # from the side. Regression test: test_ground_plane_is_horizontal_yup.
    ground_verts = np.array([
        [0, 0, 0],
        [w * horizontal_scale_m, 0, 0],
        [w * horizontal_scale_m, 0, -h * horizontal_scale_m],
        [0, 0, -h * horizontal_scale_m],
    ])
    ground_faces = np.array([[0, 2, 1], [0, 3, 2]])
    ground_uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    tex_img = PILImage.fromarray(rgb_texture)
    ground_material = trimesh.visual.material.PBRMaterial(baseColorTexture=tex_img, metallicFactor=0.0, roughnessFactor=1.0)
    ground_mesh = trimesh.Trimesh(vertices=ground_verts, faces=ground_faces,
                                   visual=TextureVisuals(uv=ground_uv, material=ground_material), process=False)

    scene = trimesh.Scene()
    scene.add_geometry(ground_mesh, node_name="ground", geom_name="ground")
    wall_color = [190, 185, 175, 255]  # neutral flat facade tone — see docstring

    n_added = 0
    for i, fp in enumerate(footprints):
        if fp.height_m <= 0 or len(fp.polygon) < 3:
            continue
        # Image pixel coords (x right, y down) -> world (X, Z) ground
        # position, matching the ground plane above exactly (same
        # horizontal_scale_m, same sign on the row->Z mapping).
        poly_world = fp.polygon.copy().astype(np.float64)
        poly_world[:, 0] *= horizontal_scale_m
        poly_world[:, 1] *= -horizontal_scale_m
        try:
            shp = Polygon(poly_world)
            if not shp.is_valid:
                shp = shp.buffer(0)
            if shp.is_empty or shp.area <= 0:
                continue
            block = extrude_polygon(shp, height=fp.height_m)
        except Exception:
            continue  # skip degenerate polygons rather than fail the whole mesh

        # extrude_polygon extrudes the 2D (X, Z) footprint along its
        # OWN Z axis by `height` — i.e. it returns vertices as
        # (X, Z_footprint, H) in trimesh's natural (Z-up) convention.
        # Reorder columns to (X, H, Z_footprint) so the extrusion
        # height lands on world Y, matching the ground plane's Y-up
        # convention exactly. This is the other half of the fix above.
        v = np.asarray(block.vertices)
        block.vertices = v[:, [0, 2, 1]]

        block.visual = trimesh.visual.ColorVisuals(block, vertex_colors=np.tile(wall_color, (len(block.vertices), 1)))
        name = f"building_{i}"
        scene.add_geometry(block, node_name=name, geom_name=name)
        n_added += 1

    for name, geom in scene.geometry.items():
        if name != "ground":
            geom.fix_normals()
    return scene
