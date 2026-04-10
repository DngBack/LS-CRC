#!/usr/bin/env python3
"""
Export qualitative LS-CRC panels for papers: image, GT, pred, score heatmap, accept mask,
and a color-coded **state map** (TP / FP / FN / abstain-on-FG / abstain-on-BG) so errors are easy to see.

Requires the same --checkpoint-dir / backbone flags as evaluate.py. Calibrates tau* on --cal-root, then
saves PNGs from --test-root in order.

Example:
  mkdir -p figures/qual_cvc
  python tools/export_qualitative_figures.py \\
    --checkpoint-dir checkpoints_cvc_adapted \\
    --backbone deeplabv3plus \\
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


def _selective_state_rgb(y, pred, accept, blend_rgb=None, blend_alpha=0.22):
    """
    Per-pixel semantic coloring for selective prediction (paper-friendly).
    accept=1: committed region; accept=0: abstain.

    - Green: TP  (accept, pred=1, gt=1)
    - Red:   FP  (accept, pred=1, gt=0)
    - Blue:  FN  (accept, pred=0, gt=1) — accepted but missed foreground
    - Orange: abstain on foreground (gt=1, accept=0) — deferred polyp
    - Dark gray: abstain on background or TN under accept
    """
    y = (y > 0.5).astype(np.bool_)
    pred = (pred > 0.5).astype(np.bool_)
    acc = (accept > 0.5).astype(np.bool_)
    H, W = y.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    # TN under acceptance: accept, pred=0, gt=0
    m = acc & (~pred) & (~y)
    rgb[m] = (0.07, 0.07, 0.09)
    # Abstain on background
    m = (~acc) & (~y)
    rgb[m] = (0.11, 0.11, 0.13)
    # Abstain on foreground (important for LS-CRC story)
    m = (~acc) & y
    rgb[m] = (1.0, 0.55, 0.05)
    # Accepted FN / FP / TP
    m = acc & (~pred) & y
    rgb[m] = (0.2, 0.45, 1.0)
    m = acc & pred & (~y)
    rgb[m] = (0.95, 0.15, 0.2)
    m = acc & pred & y
    rgb[m] = (0.15, 0.82, 0.22)

    if blend_rgb is not None and blend_alpha > 0:
        bg = np.clip(blend_rgb.astype(np.float32), 0, 1)
        rgb = (1.0 - blend_alpha) * rgb + blend_alpha * bg
        rgb = np.clip(rgb, 0, 1)
    return rgb


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
    p.add_argument(
        "--blend-image-into-state",
        type=float,
        default=0.22,
        help="0=no blend; (0,1] mixes RGB input under the state map for spatial context (default 0.22).",
    )
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

                blend = max(0.0, min(1.0, args.blend_image_into_state))
                state_rgb = _selective_state_rgb(gt, pr, acc, blend_rgb=img, blend_alpha=blend)

                fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))
                axes[0].imshow(img)
                axes[0].set_title("Image")
                axes[0].axis("off")

                axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("GT mask")
                axes[1].axis("off")

                axes[2].imshow(pr, cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Pred mask")
                axes[2].axis("off")

                im_sc = axes[3].imshow(sc, cmap="magma", vmin=0, vmax=1)
                axes[3].set_title("Acceptance score")
                axes[3].axis("off")
                plt.colorbar(im_sc, ax=axes[3], fraction=0.046, pad=0.02)

                axes[4].imshow(acc, cmap="gray", vmin=0, vmax=1)
                axes[4].set_title(f"Accept mask (τ={tau_star:.2f})")
                axes[4].axis("off")

                axes[5].imshow(state_rgb)
                axes[5].set_title("Selective states")
                axes[5].axis("off")

                if isinstance(fname, (list, tuple)):
                    name = fname[i]
                elif isinstance(fname, torch.Tensor):
                    name = str(fname[i].item()) if fname.numel() > 0 else str(saved)
                else:
                    name = str(fname)
                leg = (
                    "Green=TP  Red=FP  Blue=FN(commit)  Orange=abstain|FG  "
                    "Gray=abstain|BG / TN"
                )
                fig.suptitle(f"{name} | τ={tau_star:.3f} | α={args.alpha}\n{leg}", fontsize=8)
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
