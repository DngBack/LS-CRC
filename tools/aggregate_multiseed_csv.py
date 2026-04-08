#!/usr/bin/env python3
"""
Aggregate several evaluate.py CSV exports (same columns, same alpha/scenarios).
Computes mean and std over runs for numeric columns, grouped by Dataset + Method.

Example:
  python tools/aggregate_multiseed_csv.py \\
    "figures/multiseed/sweep_*__alpha0p05__scen4.csv" \\
    -o figures/multiseed/summary_alpha0p05.csv
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pattern", help="Glob path to CSV files (quoted if it contains *).")
    p.add_argument("-o", "--output", required=True, help="Output CSV path.")
    args = p.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched: {args.pattern!r}")

    dfs = [pd.read_csv(path) for path in paths]
    all_df = pd.concat(dfs, ignore_index=True)
    keys = ["Dataset", "Method"]
    for k in keys:
        if k not in all_df.columns:
            raise SystemExit(f"Expected column {k!r} in CSV (evaluate.py export).")

    num_cols = [c for c in all_df.columns if c not in keys]
    g = all_df.groupby(keys, sort=True)
    counts = g.size().reset_index(name="num_runs")
    agg = g[num_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    summary = counts.merge(agg.reset_index(), on=keys, how="left")
    summary.to_csv(args.output, index=False)
    print(f"Wrote {args.output} from {len(paths)} file(s), {len(all_df)} total rows.")


if __name__ == "__main__":
    main()
