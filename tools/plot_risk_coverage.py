#!/usr/bin/env python3
"""
Plot risk--coverage sensitivity from multiple evaluate.py CSV exports (one file per alpha).

Example:
  python tools/plot_risk_coverage.py \\
    --dataset CVC-ClinicDB-ID \\
    --method "LS-CRC (Ours)" \\
    --alpha-csv 0.05 figures/sweep/eval_ckpt-cvcadapt_alpha0p05_scen4.csv \\
    --alpha-csv 0.10 figures/sweep/eval_ckpt-cvcadapt_alpha0p10_scen4.csv \\
    -o figures/paper/risk_coverage_vs_alpha_cvc_lscrc.png
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Plot Coverage & Expected Risk vs alpha from evaluate CSVs.")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Exact value in the Dataset column (e.g. CVC-ClinicDB-ID, Kvasir-SEG-ID).",
    )
    p.add_argument(
        "--method",
        type=str,
        default="LS-CRC (Ours)",
        help='Exact Method column string (default: LS-CRC (Ours)).',
    )
    p.add_argument(
        "--alpha-csv",
        nargs=2,
        metavar=("ALPHA", "CSV"),
        action="append",
        required=True,
        help="Repeat: alpha path/to.csv (alpha as float, e.g. 0.05).",
    )
    p.add_argument("-o", "--out", type=str, required=True, help="Output PNG path (dirs created if needed).")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    alphas = []
    coverages = []
    risks = []

    for alpha_str, csv_path in args.alpha_csv:
        alpha = float(alpha_str)
        if not os.path.isfile(csv_path):
            hint = ""
            sweep = "figures/sweep"
            if os.path.isdir(sweep):
                found = sorted(glob.glob(os.path.join(sweep, "*.csv")))
                if found:
                    hint = "\n  Files in figures/sweep/ (use real paths, not doc examples):\n    " + "\n    ".join(
                        found[:15]
                    )
                    if len(found) > 15:
                        hint += f"\n    ... and {len(found) - 15} more"
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                f"  Doc examples use fake timestamps; run: ls -1 figures/sweep/\n"
                f"  Then pass each real path after --alpha-csv.{hint}"
            )
        df = pd.read_csv(csv_path)
        if "Dataset" not in df.columns or "Method" not in df.columns:
            raise ValueError(f"{csv_path}: expected columns Dataset, Method (evaluate.py CSV).")
        sub = df[(df["Dataset"] == args.dataset) & (df["Method"] == args.method)]
        if sub.empty:
            raise ValueError(
                f"No row for Dataset={args.dataset!r} Method={args.method!r} in {csv_path}. "
                f"Available: {df[['Dataset', 'Method']].drop_duplicates().values.tolist()}"
            )
        row = sub.iloc[0]
        alphas.append(alpha)
        coverages.append(float(row["Coverage"]))
        risks.append(float(row["Expected Risk"]))

    # Sort by alpha for line plots
    order = sorted(range(len(alphas)), key=lambda i: alphas[i])
    alphas = [alphas[i] for i in order]
    coverages = [coverages[i] for i in order]
    risks = [risks[i] for i in order]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    axes[0].plot(alphas, coverages, "o-", color="tab:blue")
    axes[0].set_xlabel(r"Risk budget $\alpha$")
    axes[0].set_ylabel("Coverage")
    axes[0].set_title(f"{args.dataset} — {args.method}\nCoverage vs α")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alphas, risks, "o-", color="tab:red")
    axes[1].set_xlabel(r"Risk budget $\alpha$")
    axes[1].set_ylabel("Expected Risk")
    axes[1].set_title("Expected Risk vs α (on test)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Wrote {args.out}")

    # Second figure: risk–coverage scatter (each point labeled by alpha)
    fig2, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(coverages, risks, c=alphas, cmap="viridis", s=80, edgecolors="k", zorder=3)
    for a, c, r in zip(alphas, coverages, risks):
        ax.annotate(f"α={a:g}", (c, r), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Expected Risk")
    ax.set_title(f"{args.dataset} — {args.method}\nRisk–Coverage (colored by α)")
    ax.grid(True, alpha=0.3)
    p2 = os.path.splitext(args.out)[0] + "_scatter.png"
    fig2.tight_layout()
    fig2.savefig(p2, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
