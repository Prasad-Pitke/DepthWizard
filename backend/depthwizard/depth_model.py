"""
depth_model.py
---------------
Pluggable monocular relative-depth backbones.

DepthWizard is designed around a single contract:

    backbone.predict(rgb: np.ndarray[H,W,3] uint8) -> np.ndarray[H,W] float32

Any model satisfying that contract can be dropped in. Two implementations
ship in this module:

1. ``TorchHubDepthBackbone`` — wraps a real foundation depth model
   (MiDaS / DPT-Hybrid / Depth-Anything-V2) loaded via torch.hub or
   HuggingFace `transformers`. This is the recommended backbone for
   production / GPU deployments. It is imported lazily so the rest of
   the system has zero hard dependency on torch being installed.

2. ``ClassicalReliefBackbone`` — a dependency-light, CPU-only fallback
   that estimates relative relief from photometric and structural cues
   (multi-scale shading gradients + local variance + guided smoothing).
   It is intentionally simple and is NOT a substitute for a learned
   foundation model in production, but it lets the full pipeline
   (ingest -> relative depth -> calibration -> DSM -> mesh) run
   end-to-end on any machine with no GPU and no multi-hundred-MB model
   download, which is invaluable for CI, offline demos, and
   constrained deployment targets (edge boxes, air-gapped sites).

The pipeline always talks to backbones through ``get_backbone()`` /
``DepthBackbone``, so switching from the classical fallback to a real
learned model in production is a one-line configuration change:

    pipeline = DepthWizardPipeline(backbone="dpt_hybrid")   # learned
    pipeline = DepthWizardPipeline(backbone="classical")    # fallback
    pipeline = DepthWizardPipeline(backbone="auto")         # try learned, else fallback
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("depthwizard.depth_model")


class DepthBackbone(abc.ABC):
    """Abstract contract every depth backbone must satisfy."""

    name: str = "base"

    @abc.abstractmethod
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return a relative (scale/shift-ambiguous) depth/disparity map.

        Args:
            rgb: HxWx3 uint8 RGB image.
        Returns:
            HxW float32 array. Convention: LARGER value = FARTHER /
            LOWER elevation (i.e. this is a depth map, not a height
            map). The pipeline inverts and normalizes it downstream.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# 1. Learned foundation-model backbone (torch / timm / transformers)
# --------------------------------------------------------------------------
class TorchHubDepthBackbone(DepthBackbone):
    """Wraps a pretrained monocular depth foundation model.

    Supports three loading strategies, tried in order, first success wins:
      a) HuggingFace `transformers` pipeline for Depth-Anything-V2 /
         DPT (works fully offline once weights are cached).
      b) `torch.hub` MiDaS (intel-isl/MiDaS repo, DPT_Hybrid / MiDaS_small).
      c) A user-supplied callable / torch.nn.Module passed via
         `custom_model`.

    This class is import-safe: torch is imported inside __init__, so
    merely importing depthwizard.depth_model never requires torch.
    """

    SUPPORTED = {
        "dpt_hybrid": ("Intel/dpt-hybrid-midas", "transformers"),
        "dpt_large": ("Intel/dpt-large", "transformers"),
        "depth_anything_v2_small": ("depth-anything/Depth-Anything-V2-Small-hf", "transformers"),
        "midas_small": ("MiDaS_small", "torchhub"),
        "midas_hybrid": ("DPT_Hybrid", "torchhub"),
    }

    def __init__(self, model_key: str = "depth_anything_v2_small", device: str | None = None,
                 custom_model=None):
        self.name = model_key
        self._device_str = device
        self._model = custom_model
        self._transform = None
        self._backend = "custom" if custom_model is not None else None
        if custom_model is None:
            self._load(model_key)

    def _load(self, model_key: str):
        import torch  # noqa: F401  (raises ImportError if torch unavailable)

        if model_key not in self.SUPPORTED:
            raise ValueError(f"Unknown model_key '{model_key}'. Options: {list(self.SUPPORTED)}")
        hf_id, backend = self.SUPPORTED[model_key]
        device = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device_str = device

        if backend == "transformers":
            from transformers import pipeline as hf_pipeline
            self._model = hf_pipeline(task="depth-estimation", model=hf_id, device=device)
            self._backend = "transformers"
        else:  # torchhub (MiDaS)
            midas = torch.hub.load("intel-isl/MiDaS", hf_id)
            midas.to(device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._transform = (
                transforms.dpt_transform if "Hybrid" in hf_id or "Large" in hf_id
                else transforms.small_transform
            )
            self._model = midas
            self._backend = "torchhub"

        logger.info("Loaded depth backbone '%s' via %s on %s", model_key, self._backend, device)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        if self._backend == "transformers":
            import torch  # noqa: F401 (transformers pipeline manages its own torch usage)
            from PIL import Image
            pil_img = Image.fromarray(rgb)
            out = self._model(pil_img)
            depth = np.array(out["depth"], dtype=np.float32)
            # HF depth-estimation pipelines return *predicted depth*
            # (near=low value) for DPT/MiDaS-family checkpoints.
            return depth

        elif self._backend == "torchhub":
            import torch
            inp = self._transform(rgb).to(self._device_str)
            with torch.no_grad():
                pred = self._model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
                ).squeeze()
            disparity = pred.cpu().numpy().astype(np.float32)
            # MiDaS outputs *disparity* (near=high). Convert to depth
            # convention (near=low) so all backbones agree.
            eps = 1e-6
            depth = 1.0 / (disparity + disparity.max() * 0.0 + eps)
            depth = depth.max() - depth  # re-orient: farther = larger value
            return depth

        else:  # custom callable — deliberately requires NO torch import.
            # This is what makes it possible to plug in a fine-tuned
            # checkpoint (see scripts/finetune_on_gamus.py) or any other
            # depth function on a machine where torch is unavailable or
            # broken, as long as the *inference* side has already been
            # reduced to a plain numpy-in/numpy-out callable.
            return np.asarray(self._model(rgb), dtype=np.float32)


# --------------------------------------------------------------------------
# 2. Dependency-light classical fallback backbone
# --------------------------------------------------------------------------
@dataclass
class ClassicalReliefConfig:
    shading_weight: float = 0.55
    texture_weight: float = 0.25
    shadow_weight: float = 0.20
    smooth_sigma: float = 3.0
    pyramid_levels: int = 4


class ClassicalReliefBackbone(DepthBackbone):
    """CPU-only, zero-heavy-dependency relative relief estimator.

    Combines three classical monocular depth cues that are known to
    correlate with elevation in nadir/oblique aerial imagery:

    1. **Multi-scale shading gradient** — under roughly-directional
       illumination (sun angle), the illuminated side of raised
       structures (buildings, ridgelines) is brighter and its gradient
       points toward the light source; integrating a Laplacian-pyramid
       "shape-from-shading" style relief field over multiple scales
       captures both fine (individual rooftops) and coarse (hills)
       structure.
    2. **Local texture / high-frequency energy** — vegetation canopies,
       urban rooftops and rubble typically have far higher local
       variance than flat ground, water or roads; used as a
       structural-roughness prior that boosts predicted relief in
       textured regions.
    3. **Cast-shadow length** — long dark regions adjacent to a bright
       edge (a shadow) scale roughly with the height of the object
       casting it, giving a coarse, physically-grounded height signal
       independent of the shading model above.

    The three cue maps are percentile-normalized (robust to outlier
    pixels), weighted, summed, median-despiked, and guided-filtered
    against the luminance channel to keep object edges crisp without
    letting a handful of extreme gradient pixels (e.g. painted road
    stripes, rooftop seams on flat uniformly-lit nadir imagery) dominate
    the whole map. Output is inverted to depth convention (farther/lower
    = larger value) to match the abstract DepthBackbone contract.
    """

    name = "classical"

    def __init__(self, config: ClassicalReliefConfig | None = None):
        self.cfg = config or ClassicalReliefConfig()

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape

        relief = self._multiscale_shading_relief(gray)
        texture = self._local_texture_energy(gray)
        shadow = self._shadow_height_proxy(rgb, gray)

        relief, texture, shadow = self._norm(relief), self._norm(texture), self._norm(shadow)
        cfg = self.cfg
        height = (cfg.shading_weight * relief +
                  cfg.texture_weight * texture +
                  cfg.shadow_weight * shadow)

        # De-spike pass: a 3x3 median kills the single-pixel-wide
        # "comb" spikes that raw Sobel-gradient edges produce on
        # nadir imagery with sharp linear features (rooftop edges,
        # road/parking-lot markings, curbs) that have nothing to do
        # with real elevation change.
        height = cv2.medianBlur((self._norm(height) * 255).astype(np.uint8), 5).astype(np.float32) / 255.0

        # Guided/edge-preserving smoothing keyed off luminance so we
        # don't blur genuine structural edges (building outlines).
        # cv2.ximgproc.guidedFilter requires opencv-contrib-python; on
        # some installs (opencv-python and opencv-contrib-python both
        # present, or only the non-contrib package) the `ximgproc`
        # namespace can exist but be empty, or be missing entirely.
        # Try/except (not hasattr) is what actually guards against
        # both failure modes, and falls back to a plain edge-preserving
        # bilateral filter, which needs no extra module.
        height_u8 = (self._norm(height) * 255).astype(np.uint8)
        try:
            height = cv2.ximgproc.guidedFilter(
                guide=(gray * 255).astype(np.uint8), src=height_u8, radius=12, eps=400,
            ).astype(np.float32)
        except (AttributeError, cv2.error):
            height = cv2.bilateralFilter(height_u8, d=11, sigmaColor=60, sigmaSpace=15).astype(np.float32)

        height = self._norm(height)
        # Convert HEIGHT (higher=up) to DEPTH convention (higher=farther/lower)
        depth = height.max() - height
        return depth.astype(np.float32)

    @staticmethod
    def _norm(a: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
        """Percentile-based normalization to [0, 1]. Unlike raw min-max,
        this is robust to the small number of extreme-outlier pixels
        that sharp linear features (rooftop/road edges, parking-lot
        stripes) produce in a Sobel-gradient relief map — without this,
        a handful of hot pixels compress the entire rest of the scene
        toward zero and the "signal" becomes indistinguishable from
        noise once meshed."""
        lo, hi = np.percentile(a, lo_pct), np.percentile(a, hi_pct)
        if hi <= lo:
            lo, hi = a.min(), a.max()
        span = hi - lo
        if span < 1e-8:
            return np.zeros_like(a)
        return np.clip((a - lo) / span, 0.0, 1.0)

    def _multiscale_shading_relief(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        pyramid = [gray]
        for _ in range(cfg.pyramid_levels - 1):
            pyramid.append(cv2.pyrDown(pyramid[-1]))

        relief_acc = np.zeros_like(gray)
        for level_img in pyramid:
            gx = cv2.Sobel(level_img, cv2.CV_32F, 1, 0, ksize=5)
            gy = cv2.Sobel(level_img, cv2.CV_32F, 0, 1, ksize=5)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)
            # Clip gradient outliers before integration. Sharp linear
            # features unrelated to elevation (rooftop edges, road/lot
            # markings, curb lines) produce gradient spikes an order of
            # magnitude larger than genuine gradual relief; left
            # unclipped, those few pixels dominate the whole map after
            # normalization and everything else reads as flat.
            cap = np.percentile(grad_mag, 97)
            grad_mag = np.clip(grad_mag, 0, cap if cap > 1e-6 else grad_mag.max())
            integrated = cv2.GaussianBlur(grad_mag, (0, 0), sigmaX=cfg.smooth_sigma)
            integrated = cv2.resize(integrated, (gray.shape[1], gray.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)
            relief_acc += integrated
        # combine gradient-energy relief with raw luminance (brighter
        # facades facing the sun read as "raised")
        relief_acc += 0.6 * cv2.GaussianBlur(gray, (0, 0), sigmaX=cfg.smooth_sigma * 2)
        return relief_acc

    def _local_texture_energy(self, gray: np.ndarray) -> np.ndarray:
        mean = cv2.blur(gray, (9, 9))
        sq_mean = cv2.blur(gray * gray, (9, 9))
        variance = np.clip(sq_mean - mean ** 2, 0, None)
        return cv2.GaussianBlur(variance, (0, 0), sigmaX=1.5)

    def _shadow_height_proxy(self, rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
        # Dark regions adjacent to strong edges are treated as shadows;
        # their local extent is used as a coarse height proxy.
        dark_mask = (gray < np.percentile(gray, 25)).astype(np.uint8)
        dist = cv2.distanceTransform(dark_mask, cv2.DIST_L2, 5)
        edges = cv2.Canny((gray * 255).astype(np.uint8), 40, 120)
        edge_dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        proximity_to_edge = np.exp(-edge_dist / 15.0)
        return dist * proximity_to_edge


def get_backbone(name: str = "auto", **kwargs) -> DepthBackbone:
    """Factory returning a ready-to-use DepthBackbone.

    name:
      "auto"        -> try a learned model, silently fall back to classical
      "classical"   -> force the dependency-light CPU fallback
      <model_key>   -> force a specific learned backbone (see
                       TorchHubDepthBackbone.SUPPORTED), raises if
                       torch/transformers are unavailable
    """
    if name == "classical":
        return ClassicalReliefBackbone(**kwargs)

    if name == "auto":
        try:
            return TorchHubDepthBackbone(model_key=kwargs.pop("model_key", "depth_anything_v2_small"), **kwargs)
        except Exception as e:  # torch / transformers not installed, no network, etc.
            logger.warning("Learned backbone unavailable (%s); falling back to classical backbone.", e)
            return ClassicalReliefBackbone()

    return TorchHubDepthBackbone(model_key=name, **kwargs)
