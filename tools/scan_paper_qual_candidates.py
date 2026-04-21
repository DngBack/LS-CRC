#!/usr/bin/env python3
"""
Scan the test split and rank images for paper Figure 2 / Figure 3.

Computes per-image localized risk for entropy vs LS-CRC at a shared risk budget α (separate τ per method).
Writes a CSV you can open to pick filenames, then pass those to export_paper_figure2_qual.py / export_paper_figure3_compare.py.

Example:
  python tools/scan_paper_qual_candidates.py \\
    --checkpoint-dir checkpoints_cvc_adapted \\
    --backbone deeplabv3plus --encoder-weights imagenet \\
    --cal-root data/CVC-ClinicDB --test-root data/CVC-ClinicDB \\
    --alpha 0.05 --out-csv figures/paper/qual_candidates_cvc_id_a005.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = Path(__file__).resolve().parent
for _p in (_ROOT, _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from calibrate import calibrate_threshold
from data.dataset import get_dataloader
from models.backbone import get_backbone
from models.rejector import Rejector
from paper_selective_helpers import (
    accept_entropy,
    accept_lscrc,
    per_image_localized_risk_and_aux,
    spatial_weights,
)


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
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-csv", type=str, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_weights = None if args.encoder_weights.lower() in ("none", "null", "") else args.encoder_weights

    cal_loader = get_dataloader(args.cal_root, split="cal", batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = get_dataloader(args.test_root, split="test", batch_size=args.batch_size, num_workers=args.num_workers)

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
    print(f"Calibrated τ_entropy={tau_e:.4f}  τ_lscrc={tau_l:.4f}  α={args.alpha}")

    rows = []
    with torch.no_grad():
        for x, y, tags, fname in test_loader:
            x = x.to(device)
            y = y.to(device)
            _, prob, features = backbone(x)
            pred = (prob >= 0.5).float()
            A_e = accept_entropy(prob, tau_e)
            A_l, _ = accept_lscrc(rejector, features, prob, tau_l)
            w_u = spatial_weights(y)
            r_e, c_e, fg_e, ce_e = per_image_localized_risk_and_aux(y, pred, A_e, w_u)
            r_l, c_l, fg_l, ce_l = per_image_localized_risk_and_aux(y, pred, A_l, w_u)

            for i in range(x.size(0)):
                if isinstance(fname, (list, tuple)):
                    name = fname[i]
                elif isinstance(fname, torch.Tensor):
                    name = str(fname[i].item())
                else:
                    name = str(fname)
                sz = int(tags["size"][i].item()) if isinstance(tags["size"], torch.Tensor) else int(tags["size"][i])
                cp = int(tags["complexity"][i].item()) if isinstance(tags["complexity"], torch.Tensor) else int(tags["complexity"][i])
                rows.append(
                    {
                        "filename": name,
                        "risk_entropy": float(r_e[i].item()),
                        "risk_lscrc": float(r_l[i].item()),
                        "coverage_entropy": float(c_e[i].item()),
                        "coverage_lscrc": float(c_l[i].item()),
                        "fg_accept_frac_entropy": float(fg_e[i].item()),
                        "fg_accept_frac_lscrc": float(fg_l[i].item()),
                        "committed_err_frac_entropy": float(ce_e[i].item()),
                        "committed_err_frac_lscrc": float(ce_l[i].item()),
                        "size_tag": sz,
                        "complexity_tag": cp,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows; empty test loader?")
        return

    p80 = np.percentile(df["risk_entropy"].values, 80)
    df["score_fig2_boundary"] = (df["risk_entropy"] - df["risk_lscrc"]).clip(lower=0) * 3.0
    df["score_fig2_boundary"] += (df["fg_accept_frac_lscrc"] - df["fg_accept_frac_entropy"]).clip(lower=0) * 2.0
    df["score_fig2_boundary"] += (df["risk_entropy"] >= p80).astype(float) * 0.35
    df["score_fig2_boundary"] += df["complexity_tag"].astype(float) * 0.25
    cov_gap = (df["coverage_entropy"] - df["coverage_lscrc"]).abs()
    df["score_fig2_boundary"] -= (cov_gap > 0.18).astype(float) * 0.5

    df["score_fig2_small"] = (df["size_tag"] == 0).astype(float) * 0.4
    df["score_fig2_small"] += (df["risk_entropy"] - df["risk_lscrc"]).clip(lower=0) * 2.5
    df["score_fig2_small"] += (df["fg_accept_frac_lscrc"] - df["fg_accept_frac_entropy"]).clip(lower=0) * 2.0
    df["score_fig2_small"] -= (cov_gap > 0.22).astype(float) * 0.6

    df["score_fig3_compare"] = (df["committed_err_frac_entropy"] - df["committed_err_frac_lscrc"]).clip(lower=0) * 4.0
    df["score_fig3_compare"] += (df["coverage_lscrc"] >= df["coverage_entropy"] - 0.07).astype(float) * 0.25
    df["score_fig3_compare"] += (df["risk_entropy"] - df["risk_lscrc"]).clip(lower=0)

    df["rank_fig2_boundary"] = df["score_fig2_boundary"].rank(ascending=False, method="min").astype(int)
    df["rank_fig2_small"] = df["score_fig2_small"].rank(ascending=False, method="min").astype(int)
    df["rank_fig3"] = df["score_fig3_compare"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("filename")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}  ({len(df)} rows)")
    top_b = df.sort_values("score_fig2_boundary", ascending=False).head(5)
    print("Top 5 (fig2 boundary narrative):\n", top_b[["filename", "score_fig2_boundary", "risk_entropy", "risk_lscrc"]].to_string(index=False))
    top_s = df.sort_values("score_fig2_small", ascending=False).head(5)
    print("\nTop 5 (fig2 small-lesion narrative):\n", top_s[["filename", "score_fig2_small", "size_tag", "risk_entropy", "risk_lscrc"]].to_string(index=False))
    top3 = df.sort_values("score_fig3_compare", ascending=False).head(5)
    print("\nTop 5 (fig3 entropy vs LS-CRC):\n", top3[["filename", "score_fig3_compare", "committed_err_frac_entropy", "committed_err_frac_lscrc"]].to_string(index=False))


if __name__ == "__main__":
    main()
