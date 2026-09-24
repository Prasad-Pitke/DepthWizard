"""
finetune_on_gamus.py
----------------------
Fine-tunes a learned monocular depth model on GAMUS to close the
domain gap between natural egocentric imagery (what foundation depth
models are trained on) and top-down remote-sensing imagery — this is
the "Scale Calibration" / "Elevation Extraction" milestone's deeper
version: instead of only post-hoc calibrating a frozen model's output,
this actually adapts the model's weights to remote-sensing structure
(building edges, road networks, canopy texture) using GAMUS's real
LiDAR AGL as supervision.

REQUIRES: torch + transformers (NOT installed by default — see
backend/requirements.txt). This script was written and structurally
validated (argument parsing, data flow, checkpoint I/O) but the actual
training loop was NOT executed in the environment this project was
built in, since that sandbox had no GPU and no access to
download.pytorch.org / huggingface.co. Run this on your own GPU host.

Approach:
  - Loads a pretrained depth-estimation model from `transformers`
    (default: Intel/dpt-hybrid-midas) as the starting point — fine-
    tuning from a strong natural-image prior converges faster and
    generalizes better than training from scratch on GAMUS's ~6k tiles.
  - Optimizes a **scale-invariant log loss (SILog)**, the standard loss
    for monocular depth regression when the model's own output isn't
    guaranteed to be in a metric scale — appropriate here since GAMUS's
    AGL is metric but the frozen head initially outputs unscaled
    relative depth.
  - Randomly crops 1024x1024 GAMUS tiles down to a smaller training
    resolution (default 384) for GPU memory reasons; this also acts as
    lightweight data augmentation.
  - Saves a checkpoint you can point `depth_model.TorchHubDepthBackbone`
    at via `custom_model=` (see the __main__ docstring below for the
    load-back snippet).

Usage:
    python scripts/finetune_on_gamus.py \
        --gamus-root ./gamus_data \
        --base-model Intel/dpt-hybrid-midas \
        --epochs 5 --batch-size 4 --lr 1e-5 \
        --output-dir ./checkpoints/dpt-hybrid-gamus
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def silog_loss(pred, target, valid_mask, lambd: float = 0.5, eps: float = 1e-6):
    """Scale-invariant log loss (Eigen et al. 2014). `pred`/`target`
    must be strictly positive where `valid_mask` is True."""
    import torch

    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    diff = torch.log(pred[valid_mask]) - torch.log(target[valid_mask])
    if diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return torch.sqrt(torch.mean(diff ** 2) - lambd * torch.mean(diff) ** 2 + eps)


def main():
    p = argparse.ArgumentParser(description="Fine-tune a depth backbone on GAMUS (requires torch+transformers).")
    p.add_argument("--gamus-root", required=True)
    p.add_argument("--base-model", default="Intel/dpt-hybrid-midas")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--crop-size", type=int, default=384)
    p.add_argument("--city-prefix", default=None)
    p.add_argument("--output-dir", default="./checkpoints/gamus-finetuned")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    except ImportError as e:
        print("This script requires `torch` and `transformers`, which are not installed "
              "in this environment. Install them on a GPU host with:\n"
              "    pip install torch transformers\n"
              f"(Original import error: {e})")
        sys.exit(1)

    from depthwizard.datasets.gamus import make_torch_dataset

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(args.base_model)
    model = AutoModelForDepthEstimation.from_pretrained(args.base_model).to(device)
    model.train()

    def transform(rgb, height):
        import numpy as np
        h, w = rgb.shape[:2]
        cs = args.crop_size
        if h > cs and w > cs:
            y0 = np.random.randint(0, h - cs)
            x0 = np.random.randint(0, w - cs)
            rgb = rgb[y0:y0 + cs, x0:x0 + cs]
            height = height[y0:y0 + cs, x0:x0 + cs]
        return rgb, height

    dataset = make_torch_dataset(args.gamus_root, split="train", transform=transform,
                                  city_prefix=args.city_prefix)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training on {len(dataset)} GAMUS tiles for {args.epochs} epochs…")
    for epoch in range(args.epochs):
        running_loss, n_batches = 0.0, 0
        for rgb_batch, height_batch in loader:
            rgb_batch = rgb_batch.to(device)
            height_batch = height_batch.to(device)

            outputs = model(pixel_values=rgb_batch)
            pred = outputs.predicted_depth
            if pred.shape[-2:] != height_batch.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=height_batch.shape[-2:], mode="bicubic", align_corners=False
                ).squeeze(1)

            valid_mask = height_batch > 0.05  # ignore near-zero "bare ground" pixels for SILog stability
            loss = silog_loss(pred, height_batch, valid_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} — avg SILog loss: {avg_loss:.4f}")

        ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        print(f"  saved checkpoint to {ckpt_dir}")

    print("\nDone. Load the fine-tuned model back into DepthWizard with:\n")
    print("    from transformers import pipeline")
    print("    from PIL import Image")
    print("    import numpy as np")
    print("    from depthwizard.pipeline import DepthWizardPipeline")
    print(f"    hf_pipe = pipeline('depth-estimation', model='{args.output_dir}/epoch_{args.epochs}')")
    print("    def custom_depth_fn(rgb: np.ndarray) -> np.ndarray:")
    print("        return np.array(hf_pipe(Image.fromarray(rgb))['depth'], dtype=np.float32)")
    print("    pipeline = DepthWizardPipeline(backbone='auto',")
    print("        backbone_kwargs={'custom_model': custom_depth_fn})")
    print("    # DepthWizardPipeline.__init__ passes backbone_kwargs to get_backbone(),")
    print("    # which forwards custom_model= to TorchHubDepthBackbone — see depth_model.py")


if __name__ == "__main__":
    main()
