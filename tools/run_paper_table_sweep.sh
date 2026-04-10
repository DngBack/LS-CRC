#!/usr/bin/env bash
# Full paper-style evaluate sweep: 4 datasets × all methods (incl. new baselines) × several α.
# Usage (from repo root):
#   source .venv/bin/activate
#   chmod +x tools/run_paper_table_sweep.sh
#   BASE="$(date +%Y%m%d_%H%M%S)__ckpt-cvc_adapted" CKPT_DIR=checkpoints_cvc_adapted ./tools/run_paper_table_sweep.sh
#
# Override backbone if checkpoint was trained with U-Net:
#   BACKBONE=unet ./tools/run_paper_table_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CKPT_DIR="${CKPT_DIR:-checkpoints_cvc_adapted}"
BACKBONE="${BACKBONE:-deeplabv3plus}"
GRID="${GRID:-1000}"
# Paper main table often uses α=0.05; include full set used in docs.
ALPHAS="${ALPHAS:-0.01 0.05 0.10 0.15}"

if [[ -z "${BASE:-}" ]]; then
  BASE="$(date +%Y%m%d_%H%M%S)__ckpt-cvc_adapted"
  echo "BASE not set; using BASE=$BASE"
fi

mkdir -p figures/sweep

for ALPHA in $ALPHAS; do
  A_FILE=$(echo "$ALPHA" | sed 's/\./p/g')
  CSV="figures/sweep/${BASE}__alpha${A_FILE}__grid${GRID}__scen4.csv"
  echo "=== $CSV ==="
  python evaluate.py \
    --checkpoint-dir "$CKPT_DIR" \
    --backbone "$BACKBONE" \
    --encoder-weights imagenet \
    --alpha "$ALPHA" \
    --calibration-num-thresholds "$GRID" \
    --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
    --scenario 'CVC-ClinicDB-ID,data/CVC-ClinicDB,data/CVC-ClinicDB' \
    --scenario 'Cross-Kvasir-cal-CVC-test,data/Kvasir-SEG,data/CVC-ClinicDB' \
    --scenario 'Cross-CVC-cal-Kvasir-test,data/CVC-ClinicDB,data/Kvasir-SEG' \
    --results-csv "$CSV"
done

echo "Done. BASE=$BASE — use this prefix in tools/plot_risk_coverage.py (see docs/guide_figures_commands.md)."
echo "Note: plot script expects __alpha0p05__ in the filename; include 0.05 in ALPHAS if you need those figures."
