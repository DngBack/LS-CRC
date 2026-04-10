#!/usr/bin/env bash
# Vẽ Coverage / Expected Risk vs α + scatter từ 4 CSV sweep (cùng BASE, grid1000).
#
# Dùng:
#   cd /path/to/LS-CRC && source .venv/bin/activate
#   BASE=20260409_144715__ckpt-cvc_adapted ./tools/render_paper_figures_from_sweep.sh
#
# Tuỳ chọn:
#   METHOD="LS-CRC (Ours)"   (mặc định)
#   OUT_PREFIX=""            — thêm vào đầu tên file PNG (tránh đè khi vẽ method khác)
#     Ví dụ: OUT_PREFIX=entropy_ METHOD="Entropy Threshold" BASE=... ./tools/render_paper_figures_from_sweep.sh
#   SWEEP_DIR=figures/sweep
#   OUT_DIR=figures/paper

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

BASE="${BASE:?Set BASE=TIMESTAMP__ckpt-cvc_adapted (prefix trước __alpha trong tên CSV)}"
SWEEP_DIR="${SWEEP_DIR:-figures/sweep}"
OUT_DIR="${OUT_DIR:-figures/paper}"
METHOD="${METHOD:-LS-CRC (Ours)}"
OUT_PREFIX="${OUT_PREFIX:-}"
GRID_TAG="${GRID_TAG:-grid1000}"

P="${SWEEP_DIR}/${BASE}"
mkdir -p "$OUT_DIR"

_alpha_csv() {
  local a="$1"
  local f
  f=$(echo "$a" | sed 's/\./p/g')
  echo "--alpha-csv" "$a" "${P}__alpha${f}__${GRID_TAG}__scen4.csv"
}

plot_one() {
  local dataset="$1"
  local out_base="$2"
  python tools/plot_risk_coverage.py \
    --dataset "$dataset" \
    --method "$METHOD" \
    $(_alpha_csv 0.01) \
    $(_alpha_csv 0.05) \
    $(_alpha_csv 0.10) \
    $(_alpha_csv 0.15) \
    -o "${OUT_DIR}/${out_base}.png" \
    --dpi 300
}

plot_one "CVC-ClinicDB-ID" "${OUT_PREFIX}lscrc_cvc_id_risk_coverage_vs_alpha"
plot_one "Kvasir-SEG-ID" "${OUT_PREFIX}lscrc_kvasir_id_risk_coverage_vs_alpha"
plot_one "Cross-Kvasir-cal-CVC-test" "${OUT_PREFIX}lscrc_cross_kvasir_cal_cvc_test_risk_coverage_vs_alpha"
plot_one "Cross-CVC-cal-Kvasir-test" "${OUT_PREFIX}lscrc_cross_cvc_cal_kvasir_test_risk_coverage_vs_alpha"

echo "Wrote PNG + _scatter.png under $OUT_DIR/ (OUT_PREFIX=${OUT_PREFIX:-<empty>}, METHOD=$METHOD)."
