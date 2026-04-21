#!/usr/bin/env bash
# Render paper Figure 1 (embedded table numbers) and optionally scan + print export commands for Fig 2–3.
#
# Usage:
#   cd /path/to/LS-CRC && source .venv/bin/activate  # if you use a venv
#   ./tools/render_neurips_paper_figures.sh
#
# Optional env:
#   CKPT=checkpoints_cvc_adapted
#   CAL_ROOT=data/CVC-ClinicDB TEST_ROOT=data/CVC-ClinicDB
#   RUN_SCAN=1   # run candidate scan when data + weights exist
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p figures/paper

python tools/plot_figure1_risk_coverage_panels.py \
  -o figures/paper/figure1_risk_coverage_panels.png --dpi 300

CKPT="${CKPT:-checkpoints_cvc_adapted}"
CAL_ROOT="${CAL_ROOT:-data/CVC-ClinicDB}"
TEST_ROOT="${TEST_ROOT:-data/CVC-ClinicDB}"
CSV_OUT="figures/paper/qual_candidates_cvc_id_a005.csv"
RUN_SCAN="${RUN_SCAN:-}"

if [[ "$RUN_SCAN" == "1" ]]; then
  if [[ -f "$CKPT/backbone.pth" && -f "$CKPT/rejector.pth" && -d "$CAL_ROOT" && -d "$TEST_ROOT" ]]; then
    echo "Running qual candidate scan -> $CSV_OUT"
    python tools/scan_paper_qual_candidates.py \
      --checkpoint-dir "$CKPT" \
      --backbone deeplabv3plus \
      --encoder-weights imagenet \
      --cal-root "$CAL_ROOT" \
      --test-root "$TEST_ROOT" \
      --alpha 0.05 \
      --calibration-num-thresholds 1000 \
      --out-csv "$CSV_OUT"
    python - <<PY
import pandas as pd
ckpt = r"""$CKPT"""
cal = r"""$CAL_ROOT"""
tst = r"""$TEST_ROOT"""
df = pd.read_csv(r"""$CSV_OUT""")
b = df.sort_values("score_fig2_boundary", ascending=False).iloc[0]["filename"]
small_sorted = df.sort_values("score_fig2_small", ascending=False)
s = small_sorted.iloc[0]["filename"]
if s == b and len(small_sorted) > 1:
    s = small_sorted.iloc[1]["filename"]
t = df.sort_values("score_fig3_compare", ascending=False).iloc[0]["filename"]
print("\n--- Suggested commands (edit filenames if you prefer another row) ---\n")
print(f"ROW1={b!r} ROW2={s!r} FIG3={t!r}")
print(
    f"""
python tools/export_paper_figure2_qual.py \\
  --checkpoint-dir {ckpt} \\
  --backbone deeplabv3plus --encoder-weights imagenet \\
  --cal-root {cal} \\
  --test-root {tst} \\
  --alpha 0.05 \\
  --filename-row1 {b} --filename-row2 {s} \\
  -o figures/paper/figure2_qual_two_cases.png

python tools/export_paper_figure3_compare.py \\
  --checkpoint-dir {ckpt} \\
  --backbone deeplabv3plus --encoder-weights imagenet \\
  --cal-root {cal} \\
  --test-root {tst} \\
  --alpha 0.05 --filename {t} \\
  -o figures/paper/figure3_entropy_vs_lscrc.png
"""
)
PY
  else
    echo "Skip scan: need weights under $CKPT and directories $CAL_ROOT / $TEST_ROOT"
  fi
else
  echo "Tip: RUN_SCAN=1 with data+weights present to write $CSV_OUT and print Fig 2–3 commands."
fi

echo "Done. Figure 1 -> figures/paper/figure1_risk_coverage_panels.png"
