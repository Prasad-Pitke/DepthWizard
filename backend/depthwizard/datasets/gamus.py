"""
gamus.py
---------
Loader for the GAMUS benchmark (Xiong et al. 2023, "GAMUS: A
Geometry-aware Multi-modal Semantic Segmentation Benchmark for Remote
Sensing Data"), recommended as the training/validation dataset for
DepthWizard's monocular depth backbone (see the SIH2026 problem
statement / IMG-PROCESS-SAC/SIH-DepthWizard-2026 reference repo).

Dataset: https://huggingface.co/datasets/earthflow/GAMUS
Paper:   https://arxiv.org/abs/2305.14914

GAMUS pairs 1024x1024 RGB aerial-orthophoto tiles (5 US cities: DC,
Philadelphia, Oklahoma, Jacksonville, NYC) with LiDAR-derived nDSM
("AGL" — height Above Ground Level, i.e. building/vegetation height
with the bare-earth terrain subtracted out) and a 6-class land-cover
map (ground, low-vegetation, building, water, road, tree). This is
exactly the domain-gap bridge the problem statement asks for: it lets
you validate (or fine-tune) a depth backbone against real LiDAR ground
truth stratified by land-cover class, which maps cleanly onto the
"urban / sparse / hilly / forested" stratification used elsewhere in
this codebase (see `evaluation.py`).

On-disk layout (verified against the actual HF repo file tree):

    <gamus_root>/
      images/{train,val,test}/<TILE_ID>_RGB.h5      # e.g. DC_01_25_RGB.h5
      heights/{train,val,test}/<TILE_ID>_AGL.h5      # e.g. DC_01_25_AGL.h5
      classes/{train,val,test}/<TILE_ID>_CLS.h5      # e.g. DC_01_25_CLS.h5  (may be absent)

Each `.h5` file holds a single tile. The internal dataset key name
inside each HDF5 file was not documented anywhere we could inspect
without downloading the (80GB) dataset, so `_read_h5_array()` below
auto-detects it: it tries a short list of common key names first, and
otherwise falls back to "the first dataset found in the file" — which
is correct for the extremely common case of one array per file. If
your copy of GAMUS uses a different, multi-dataset layout, pass an
explicit `key=` to `load_tile()`.

Downloading GAMUS:

    pip install huggingface_hub
    python -c "
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id='earthflow/GAMUS', repo_type='dataset',
                           local_dir='./gamus_data')
    "

or with the CLI: `huggingface-cli download earthflow/GAMUS --repo-type dataset --local-dir ./gamus_data`

This is an 80GB dataset — plan disk/bandwidth accordingly, or use
`allow_patterns=["images/train/DC_*", "heights/train/DC_*"]` in
`snapshot_download` to pull a city/split subset for quick iteration.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

# GAMUS's six land-cover classes (paper §3.2, listed in the order given
# in the paper: ground, low-vegetation, building, water, road, tree).
# NOTE: the *integer value* assigned to each class inside the actual
# `_CLS.h5` rasters was not documented anywhere inspectable without
# downloading the dataset; this ordering (0..5) is our best-evidence
# assumption from the paper's listing order. If your evaluation numbers
# look inverted/scrambled per class, check a few tiles' unique class
# values against the visual land-cover and adjust GAMUS_CLASS_NAMES.
GAMUS_CLASS_NAMES = {
    0: "ground",
    1: "low_vegetation",
    2: "building",
    3: "water",
    4: "road",
    5: "tree",
}

# Maps GAMUS's 6 classes onto the urban/sparse/hilly/forested
# stratification used by depthwizard.evaluation, so GAMUS validation
# results are directly comparable to the rest of this codebase's
# reporting. GAMUS has no "hilly" concept (it's flat US-city LiDAR
# data) so nothing maps there — that's expected and worth noting in
# any report generated from this dataset.
GAMUS_TO_LANDSCAPE = {
    "ground": "sparse",
    "low_vegetation": "sparse",
    "building": "urban",
    "water": "sparse",
    "road": "urban",
    "tree": "forested",
}


@dataclass
class GamusTile:
    tile_id: str
    rgb: np.ndarray                     # HxWx3 uint8
    height_agl: np.ndarray               # HxW float32, meters above ground
    landscape_class: Optional[np.ndarray] = None  # HxW int32, depthwizard.evaluation-compatible


def _read_h5_array(path: str, key: Optional[str] = None) -> np.ndarray:
    import h5py

    common_keys = ("rgb", "image", "img", "data", "agl", "height", "ndsm",
                   "cls", "class", "classes", "label", "labels", "array")
    with h5py.File(path, "r") as f:
        if key is not None:
            return np.array(f[key])
        for k in common_keys:
            if k in f:
                return np.array(f[k])
        # Fall back to the first dataset found anywhere in the file.
        found = {}
        f.visititems(lambda name, obj: found.setdefault(name, obj) if hasattr(obj, "shape") else None)
        if not found:
            raise ValueError(f"No readable dataset found inside '{path}' "
                              f"(top-level keys: {list(f.keys())}). Pass an explicit key=.")
        first_key = next(iter(found))
        return np.array(found[first_key])


def list_tile_ids(gamus_root: str, split: str = "train") -> list:
    """Returns tile IDs (e.g. 'DC_01_25') present in images/{split}/."""
    img_dir = os.path.join(gamus_root, "images", split)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"'{img_dir}' not found. Expected layout: <gamus_root>/images/{{train,val,test}}/*_RGB.h5 "
            f"— see depthwizard.datasets.gamus module docstring for how to download GAMUS."
        )
    tile_ids = []
    for fname in sorted(os.listdir(img_dir)):
        if fname.endswith("_RGB.h5"):
            tile_ids.append(fname[: -len("_RGB.h5")])
    return tile_ids


def load_tile(gamus_root: str, split: str, tile_id: str,
              rgb_key: Optional[str] = None, agl_key: Optional[str] = None,
              cls_key: Optional[str] = None) -> GamusTile:
    rgb_path = os.path.join(gamus_root, "images", split, f"{tile_id}_RGB.h5")
    agl_path = os.path.join(gamus_root, "heights", split, f"{tile_id}_AGL.h5")
    cls_path = os.path.join(gamus_root, "classes", split, f"{tile_id}_CLS.h5")

    rgb = _read_h5_array(rgb_path, rgb_key)
    if rgb.dtype != np.uint8:
        # Some exports store RGB as float [0,1] or [0,255]; normalize defensively.
        rgb = rgb.astype(np.float32)
        if rgb.max() <= 1.0 + 1e-3:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[0] < rgb.shape[-1]:
        rgb = np.moveaxis(rgb, 0, -1)  # CHW -> HWC
    rgb = rgb[..., :3]

    height = _read_h5_array(agl_path, agl_key).astype(np.float32)
    height = np.clip(height, 0, None)  # AGL should be >= 0; guard against nodata sentinels like -9999

    landscape = None
    if os.path.isfile(cls_path):
        raw_cls = _read_h5_array(cls_path, cls_key).astype(np.int32)
        landscape = np.zeros_like(raw_cls)
        for cls_id, name in GAMUS_CLASS_NAMES.items():
            landscape[raw_cls == cls_id] = _LANDSCAPE_ID[GAMUS_TO_LANDSCAPE[name]]

    return GamusTile(tile_id=tile_id, rgb=rgb, height_agl=height, landscape_class=landscape)


# Stable integer IDs matching depthwizard.evaluation.CLASS_NAMES so
# GAMUS-derived class maps plug directly into validate_dsm(class_names=...).
_LANDSCAPE_ID = {"urban": 0, "sparse": 1, "hilly": 2, "forested": 3}
LANDSCAPE_CLASS_NAMES = {v: k for k, v in _LANDSCAPE_ID.items()}


def iter_tiles(gamus_root: str, split: str = "test", limit: Optional[int] = None,
               city_prefix: Optional[str] = None) -> Iterator[GamusTile]:
    """Yields GamusTile objects one at a time (no torch required) —
    intended for validation/evaluation scripts, not training loops."""
    tile_ids = list_tile_ids(gamus_root, split)
    if city_prefix:
        tile_ids = [t for t in tile_ids if t.startswith(city_prefix)]
    if limit:
        tile_ids = tile_ids[:limit]
    for tid in tile_ids:
        yield load_tile(gamus_root, split, tid)


# ---------------------------------------------------------------------
# Optional torch Dataset wrapper (lazy import — only needed for
# scripts/finetune_on_gamus.py; the rest of this module works without
# torch installed).
# ---------------------------------------------------------------------
def make_torch_dataset(gamus_root: str, split: str = "train", transform=None,
                        city_prefix: Optional[str] = None):
    import torch
    from torch.utils.data import Dataset

    class GamusTorchDataset(Dataset):
        def __init__(self):
            self.tile_ids = list_tile_ids(gamus_root, split)
            if city_prefix:
                self.tile_ids = [t for t in self.tile_ids if t.startswith(city_prefix)]

        def __len__(self):
            return len(self.tile_ids)

        def __getitem__(self, idx):
            tile = load_tile(gamus_root, split, self.tile_ids[idx])
            rgb, height = tile.rgb, tile.height_agl
            if transform:
                rgb, height = transform(rgb, height)
            rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
            height_t = torch.from_numpy(height).float()
            return rgb_t, height_t

    return GamusTorchDataset()
