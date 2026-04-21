#!/usr/bin/env python3
"""
Figure 1: two-panel risk–coverage figure for the paper.

Panel (a): CVC-ClinicDB in-domain — entropy vs LS-CRC at four calibrated α points (Option B).
Panel (b): Kvasir-SEG in-domain — operating-point comparison at α=0.05 (Option B fallback).

Defaults embed the sample numbers from the paper checklist.

Example:
  python tools/plot_figure1_risk_coverage_panels.py -o figures/paper/figure1_risk_coverage_panels.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _default_cvc_rows():
    return pd.DataFrame(
        {
            "alpha": [0.01, 0.05, 0.10, 0.15],
            "entropy_cov": [0.132, 0.824, 0.928, 0.973],
            "entropy_risk": [0.0005, 0.0040, 0.038, 0.082],
            "lscrc_cov": [0.302, 0.862, 0.946, 0.985],
            "lscrc_risk": [0.0004, 0.0034, 0.034, 0.080],
        }
    )


def _default_kvasir_rows():
    return pd.DataFrame(
        {
            "method": ["Entropy", "Max-Softmax", "Spatial-weighted CP", "LS-CRC (Ours)"],
            "coverage": [0.888, 0.886, 0.900, 0.922],
            "risk": [0.057, 0.057, 0.059, 0.054],
        }
    )


def plot_figure1(out_path: str, dpi: int, cvc: pd.DataFrame | None = None, kva: pd.DataFrame | None = None) -> None:
    cvc = cvc if cvc is not None else _default_cvc_rows()
    kva = kva if kva is not None else _default_kvasir_rows()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.plot(
        cvc["entropy_cov"],
        cvc["entropy_risk"],
        "s-",
        color="#c0392b",
        lw=2,
        ms=7,
        label="Entropy threshold",
    )
    ax.plot(
        cvc["lscrc_cov"],
        cvc["lscrc_risk"],
        "o-",
        color="#1e8449",
        lw=2,
        ms=8,
        label="LS-CRC (Ours)",
    )
    for a, cx, cy, lx, ly in zip(
        cvc["alpha"], cvc["entropy_cov"], cvc["entropy_risk"], cvc["lscrc_cov"], cvc["lscrc_risk"]
    ):
        ax.annotate(f"α={a:g}", (cx, cy), textcoords="offset points", xytext=(-2, -12), fontsize=7, color="#922b21")
        ax.annotate(f"α={a:g}", (lx, ly), textcoords="offset points", xytext=(4, 4), fontsize=7, color="#145a32")
    ax.set_xlabel("Coverage (mean pixel acceptance rate)")
    ax.set_ylabel("Expected localized risk (test)")
    ax.set_title("(a) CVC-ClinicDB in-domain")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(0.05, 1.02)
    y_hi = float(max(cvc["entropy_risk"].max(), cvc["lscrc_risk"].max()) * 1.15 + 0.01)
    ax.set_ylim(-0.002, min(y_hi, 0.12))

    ax = axes[1]
    colors = ["#c0392b", "#7d3c98", "#2874a6", "#1e8449"]
    markers = ["s", "D", "^", "o"]
    for i, row in kva.iterrows():
        ax.scatter(
            row["coverage"],
            row["risk"],
            c=colors[i],
            marker=markers[i],
            s=95,
            edgecolors="k",
            linewidths=0.6,
            zorder=3,
            label=row["method"],
        )
    ax.set_xlabel("Coverage (mean pixel acceptance rate)")
    ax.set_ylabel("Expected localized risk (test)")
    ax.set_title("(b) Kvasir-SEG in-domain @ α = 0.05")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_xlim(0.875, 0.935)
    ax.set_ylim(0.048, 0.064)

    fig.suptitle("Risk–coverage trade-offs (in-domain)", fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", type=str, default="figures/paper/figure1_risk_coverage_panels.png")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    plot_figure1(args.out, args.dpi)


if __name__ == "__main__":
    main()
