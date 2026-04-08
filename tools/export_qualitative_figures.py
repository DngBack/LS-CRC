#!/usr/bin/env python3
"""
Export qualitative figure panels: RGB (denorm) | GT mask | Prediction | LS-CRC acceptance (score >= tau*).

Requires the same --checkpoint-dir / backbone flags as evaluate.py. Calibrates tau* on --cal-root, then
saves PNGs from --test-root in order.

Example:
  mkdir -p figures/qual_cvc
  python tools/export_qualitative_figures.py \\
    --checkpoint-dir checkpoints_cvc_adapted \\
    --encoder-weights imagenet \\
    --cal-root data/CVC-ClinicDB \\
    --test-root data/CVC-ClinicDB \\
    --alpha 0.05 \\
    --num-images 6 \\
    --out-dir figures/qual_cvc \\
    --prefix lscrc_cvc_id
"""
import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from calibrate import calibrate_threshold
from models.backbone import get_backbone
from models.rejector import Rejector


def _denorm_rgb(tchw):
    """tchw: (3,H,W) tensor, ImageNet norm."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tchw.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tchw.device).view(3, 1, 1)
    x = tchw * std + mean
    return torch.clamp(x, 0, 1).cpu().numpy().transpose(1, 2, 0)


def main():
    p = argparse.ArgumentParser(description="Export qualitative LS-CRC panels to PNG.")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--backbone", type=str, default="unet", choices=["unet", "deeplabv3plus"])
    p.add_argument("--encoder-name", type=str, default="resnet50")
    p.add_argument("--encoder-weights", type=str, default="imagenet")
    p.add_argument("--cal-root", type=str, required=True)
    p.add_argument("--test-root", type=str, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--calibration-num-thresholds",
        type=int,
        default=500,
        help="Tau grid size (match evaluate.py for comparable tau*).",
    )
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--prefix", type=str, default="qual", help="Output filenames: {prefix}_{idx:03d}.png")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    from data.dataset import get_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_weights = None if args.encoder_weights.lower() in ("none", "null", "") else args.encoder_weights

    cal_loader = get_dataloader(
        args.cal_root, split="cal", batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = get_dataloader(
        args.test_root, split="test", batch_size=args.batch_size, num_workers=args.num_workers
    )

    backbone = get_backbone(
        model_name=args.backbone,
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
    ).to(device)
    rejector = Rejector(feature_channels=backbone.feature_channels).to(device)

    ckpt_b = os.path.join(args.checkpoint_dir, "backbone.pth")
    ckpt_r = os.path.join(args.checkpoint_dir, "rejector.pth")
    backbone.load_state_dict(torch.load(ckpt_b, map_location=device))
    rejector.load_state_dict(torch.load(ckpt_r, map_location=device))

    tau_star, _, _ = calibrate_threshold(
        backbone,
        rejector,
        cal_loader,
        device,
        args.alpha,
        method="lscrc",
        num_thresholds=args.calibration_num_thresholds,
    )
    print(f"Calibrated tau* = {tau_star:.4f} (alpha={args.alpha}, cal={args.cal_root})")

    os.makedirs(args.out_dir, exist_ok=True)
    backbone.eval()
    rejector.eval()

    saved = 0
    with torch.no_grad():
        for x, y, tags, fname in test_loader:
            x, y = x.to(device), y.to(device)
            _, prob, features = backbone(x)
            pred = (prob >= 0.5).float()
            scores = rejector(features, prob)
            accept = (scores >= tau_star).float()

            for i in range(x.size(0)):
                if saved >= args.num_images:
                    break
                img = _denorm_rgb(x[i])
                gt = y[i, 0].cpu().numpy()
                pr = pred[i, 0].cpu().numpy()
                acc = accept[i, 0].cpu().numpy()
                sc = scores[i, 0].cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                axes[0].imshow(img)
                axes[0].set_title("Image")
                axes[0].axis("off")
                axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("GT")
                axes[1].axis("off")
                axes[2].imshow(img)
                axes[2].imshow(pr, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
                axes[2].set_title("Pred (red)")
                axes[2].axis("off")
                axes[3].imshow(img)
                axes[3].imshow(acc, cmap="Greens", alpha=0.4, vmin=0, vmax=1)
                axes[3].imshow(sc, cmap="viridis", alpha=0.25, vmin=0, vmax=1)
                axes[3].set_title("Accept (green) + score")
                axes[3].axis("off")
                if isinstance(fname, (list, tuple)):
                    name = fname[i]
                elif isinstance(fname, torch.Tensor):
                    name = str(fname[i].item()) if fname.numel() > 0 else str(saved)
                else:
                    name = str(fname)
                fig.suptitle(f"{name} | tau={tau_star:.3f} | alpha={args.alpha}", fontsize=9)
                fig.tight_layout()
                out_path = os.path.join(args.out_dir, f"{args.prefix}_{saved:03d}.png")
                fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"Wrote {out_path}")
                saved += 1
            if saved >= args.num_images:
                break

    if saved == 0:
        print("Warning: no images exported (empty test loader?).")


if __name__ == "__main__":
    main()
