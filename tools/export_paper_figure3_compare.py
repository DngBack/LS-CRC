#!/usr/bin/env python3
"""
Figure 3: entropy vs LS-CRC at matched calibration budget α (separate τ per method).

Single row, six panels:
  Input | GT | Entropy accepted pred | Entropy committed errors | LS-CRC accepted pred | LS-CRC committed errors

Example:
  python tools/export_paper_figure3_compare.py \\
    --checkpoint-dir checkpoints_cvc_adapted \\
    --backbone deeplabv3plus --encoder-weights imagenet \\
    --cal-root data/CVC-ClinicDB --test-root data/CVC-ClinicDB \\
    --alpha 0.05 --filename 123.png \\
    -o figures/paper/figure3_entropy_vs_lscrc.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = Path(__file__).resolve().parent
for _p in (_ROOT, _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from calibrate import calibrate_threshold
from data.dataset import get_dataloader
from models.backbone import get_backbone
from models.rejector import Rejector
from paper_selective_helpers import accept_entropy, accept_lscrc, error_overlay_rgb, accepted_prediction_rgb


def _denorm_rgb(tchw):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tchw.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tchw.device).view(3, 1, 1)
    x = tchw * std + mean
    return torch.clamp(x, 0, 1).cpu().numpy().transpose(1, 2, 0)


def _loader_one_sample(test_root, filename, num_workers):
    base = get_dataloader(test_root, split="test", batch_size=1, num_workers=num_workers)
    ds = base.dataset
    if filename not in ds.filenames:
        raise FileNotFoundError(f"{filename!r} not in test split ({len(ds.filenames)} files).")
    idx = ds.filenames.index(filename)
    return DataLoader(Subset(ds, [idx]), batch_size=1, shuffle=False, num_workers=num_workers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_cvc_adapted")
    p.add_argument("--backbone", type=str, default="deeplabv3plus", choices=["unet", "deeplabv3plus"])
    p.add_argument("--encoder-name", type=str, default="resnet50")
    p.add_argument("--encoder-weights", type=str, default="imagenet")
    p.add_argument("--cal-root", type=str, required=True)
    p.add_argument("--test-root", type=str, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--calibration-num-thresholds", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--filename", type=str, required=True)
    p.add_argument("-o", "--out", type=str, default="figures/paper/figure3_entropy_vs_lscrc.png")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_weights = None if args.encoder_weights.lower() in ("none", "null", "") else args.encoder_weights

    cal_loader = get_dataloader(args.cal_root, split="cal", batch_size=4, num_workers=args.num_workers)
    ld = _loader_one_sample(args.test_root, args.filename, args.num_workers)

    backbone = get_backbone(
        model_name=args.backbone,
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
    ).to(device)
    rejector = Rejector(feature_channels=backbone.feature_channels).to(device)
    backbone.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "backbone.pth"), map_location=device))
    rejector.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "rejector.pth"), map_location=device))
    backbone.eval()
    rejector.eval()

    tau_e, _, _ = calibrate_threshold(
        backbone, None, cal_loader, device, args.alpha, method="entropy", num_thresholds=args.calibration_num_thresholds
    )
    tau_l, _, _ = calibrate_threshold(
        backbone,
        rejector,
        cal_loader,
        device,
        args.alpha,
        method="lscrc",
        num_thresholds=args.calibration_num_thresholds,
    )
    print(f"τ_entropy={tau_e:.4f}  τ_lscrc={tau_l:.4f}")

    x, y, _, _ = next(iter(ld))
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        _, prob, features = backbone(x)
        pred = (prob >= 0.5).float()
        A_e = accept_entropy(prob, tau_e)
        A_l, _ = accept_lscrc(rejector, features, prob, tau_l)

    img = _denorm_rgb(x[0])
    gt = y[0, 0].cpu().numpy()

    panels = [
        (img, None, "Input"),
        (gt, "gray", "GT"),
        (accepted_prediction_rgb(img, pred, A_e), None, "Accepted pred — Entropy"),
        (error_overlay_rgb(img, pred, y, A_e), None, "Committed errors — Entropy"),
        (accepted_prediction_rgb(img, pred, A_l), None, "Accepted pred — LS-CRC"),
        (error_overlay_rgb(img, pred, y, A_l), None, "Committed errors — LS-CRC"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(20, 3.4))
    for ax, (data, cmap, title) in zip(axes, panels):
        if cmap == "rgb" or cmap is None:
            ax.imshow(np.clip(data, 0, 1))
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"Figure 3 | α={args.alpha} | entropy vs LS-CRC | {args.filename}",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
