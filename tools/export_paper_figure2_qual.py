#!/usr/bin/env python3
"""
Figure 2: two-row qualitative figure (7 panels each).

Row panels: Image | GT | Pred | Entropy map (norm.) | LS-CRC score | Accepted pred (LS-CRC) | Committed error overlay (LS-CRC)

Example:
  python tools/export_paper_figure2_qual.py \\
    --checkpoint-dir checkpoints_cvc_adapted \\
    --backbone deeplabv3plus --encoder-weights imagenet \\
    --cal-root data/CVC-ClinicDB --test-root data/CVC-ClinicDB \\
    --alpha 0.05 \\
    --filename-row1 123.png --filename-row2 456.png \\
    -o figures/paper/figure2_qual_two_cases.png
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
from paper_selective_helpers import accept_lscrc, entropy_and_norm, error_overlay_rgb, accepted_prediction_rgb


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


def _run_row(device, backbone, rejector, tau_l, batch):
    x, y, _, _ = batch
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        _, prob, features = backbone(x)
        pred = (prob >= 0.5).float()
        _, norm_ent = entropy_and_norm(prob)
        A_l, scores = accept_lscrc(rejector, features, prob, tau_l)
    img = _denorm_rgb(x[0])
    gt = y[0, 0].cpu().numpy()
    pr = pred[0, 0].cpu().numpy()
    ent_map = norm_ent[0, 0].cpu().numpy()
    sc = scores[0, 0].cpu().numpy()

    panels = [
        (img, None, "rgb"),
        (gt, "gray", "GT"),
        (pr, "gray", "Prediction"),
        (ent_map, "magma", "Entropy (norm.)"),
        (sc, "viridis", "LS-CRC score"),
        (accepted_prediction_rgb(img, pred, A_l), None, "Accepted pred (LS-CRC)"),
        (error_overlay_rgb(img, pred, y, A_l), None, "Committed errors (LS-CRC)"),
    ]
    return panels


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
    p.add_argument("--filename-row1", type=str, required=True)
    p.add_argument("--filename-row2", type=str, required=True)
    p.add_argument("-o", "--out", type=str, default="figures/paper/figure2_qual_two_cases.png")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_weights = None if args.encoder_weights.lower() in ("none", "null", "") else args.encoder_weights

    cal_loader = get_dataloader(args.cal_root, split="cal", batch_size=4, num_workers=args.num_workers)
    ld1 = _loader_one_sample(args.test_root, args.filename_row1, args.num_workers)
    ld2 = _loader_one_sample(args.test_root, args.filename_row2, args.num_workers)

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

    row_a = _run_row(device, backbone, rejector, tau_l, next(iter(ld1)))
    row_b = _run_row(device, backbone, rejector, tau_l, next(iter(ld2)))

    fig, axes = plt.subplots(2, 7, figsize=(22, 6.2))
    for r, row in enumerate([row_a, row_b]):
        for c, (data, cmap, title) in enumerate(row):
            ax = axes[r, c]
            if cmap == "rgb" or cmap is None:
                ax.imshow(np.clip(data, 0, 1))
            else:
                im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
                if c in (3, 4):
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle(
        f"Figure 2 qualitative | α={args.alpha} | row1={args.filename_row1} | row2={args.filename_row2}",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
