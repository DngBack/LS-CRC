#!/usr/bin/env bash
# LS-CRC: train Kvasir (DeepLab) → domain-adapt rejector/joint on CVC → optional full α sweep.
#
# Chạy từ thư mục gốc repo:
#   cd /path/to/LS-CRC && source .venv/bin/activate   # hoặc conda env
#   ./tools/run_end_to_end_deeplab_cvc_adapt.sh
#
# Biến môi trường (tuỳ chọn):
#   SKIP_DOWNLOAD=1      — bỏ qua download_data.py
#   SKIP_TRAIN_KVASIR=1  — nhảy thẳng adapt (đã có CKPT_KVASIR_DIR)
#   SKIP_ADAPT_CVC=1     — bỏ bước CVC
#   RUN_SWEEP=1          — sau khi train xong, chạy sweep α (4 file CSV)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BACKBONE="${BACKBONE:-deeplabv3plus}"
ENC_W="${ENCODER_WEIGHTS:-imagenet}"
CKPT_KVASIR_DIR="${CKPT_KVASIR_DIR:-checkpoints_kvasir_deeplab}"
CKPT_CVC_DIR="${CKPT_CVC_DIR:-checkpoints_cvc_adapted}"
SEED="${SEED:-42}"

# Epochs mặc định giống doc; có thể giảm khi debug (ví dụ EPOCHS_BACKBONE=5)
EPOCHS_BACKBONE="${EPOCHS_BACKBONE:-100}"
EPOCHS_REJECTOR="${EPOCHS_REJECTOR:-50}"
EPOCHS_JOINT="${EPOCHS_JOINT:-50}"
EPOCHS_REJECTOR_CVC="${EPOCHS_REJECTOR_CVC:-40}"
EPOCHS_JOINT_CVC="${EPOCHS_JOINT_CVC:-30}"

echo "=== Repo: $REPO_ROOT ==="

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  echo "=== Download / splits (Kvasir + CVC) ==="
  python download_data.py
else
  echo "=== SKIP_DOWNLOAD=1 — giả định data/Kvasir-SEG và data/CVC-ClinicDB đã có ==="
fi

if [[ "${SKIP_TRAIN_KVASIR:-0}" != "1" ]]; then
  echo "=== 1/3 Train backbone + rejector + joint trên Kvasir-SEG ==="
  python train.py \
    --data-root data/Kvasir-SEG \
    --backbone "$BACKBONE" \
    --encoder-weights "$ENC_W" \
    --seed "$SEED" \
    --epochs-backbone "$EPOCHS_BACKBONE" \
    --epochs-rejector "$EPOCHS_REJECTOR" \
    --epochs-joint "$EPOCHS_JOINT" \
    --checkpoint-dir "$CKPT_KVASIR_DIR"
  echo "Đã lưu: $CKPT_KVASIR_DIR/backbone.pth, rejector.pth"
else
  echo "=== SKIP_TRAIN_KVASIR=1 — dùng checkpoint có sẵn trong $CKPT_KVASIR_DIR ==="
fi

if [[ "${SKIP_ADAPT_CVC:-0}" != "1" ]]; then
  echo "=== 2/3 Adapt rejector + joint trên CVC (backbone frozen) ==="
  python train.py \
    --data-root data/CVC-ClinicDB \
    --backbone "$BACKBONE" \
    --encoder-weights "$ENC_W" \
    --seed "$SEED" \
    --resume-backbone "$CKPT_KVASIR_DIR/backbone.pth" \
    --resume-rejector "$CKPT_KVASIR_DIR/rejector.pth" \
    --epochs-backbone 0 \
    --epochs-rejector "$EPOCHS_REJECTOR_CVC" \
    --epochs-joint "$EPOCHS_JOINT_CVC" \
    --lr-rejector 5e-5 \
    --lr-joint 2e-5 \
    --checkpoint-dir "$CKPT_CVC_DIR"
  echo "Đã lưu: $CKPT_CVC_DIR/backbone.pth, rejector.pth"
else
  echo "=== SKIP_ADAPT_CVC=1 ==="
fi

echo "=== Sanity: Dice Plain (Kvasir in-domain) — Dice phải >> 0 (thường ~0.8+) ==="
SAN="figures/sweep/_sanity_dice.csv"
mkdir -p figures/sweep
python evaluate.py \
  --checkpoint-dir "$CKPT_CVC_DIR" \
  --backbone "$BACKBONE" \
  --encoder-weights "$ENC_W" \
  --alpha 0.05 \
  --calibration-num-thresholds 500 \
  --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
  --results-csv "$SAN"
echo "Xem cột Dice (Plain) trong: $SAN"

if [[ "${RUN_SWEEP:-0}" == "1" ]]; then
  echo "=== 3/3 Sweep α (paper tables) ==="
  export CKPT_DIR="$CKPT_CVC_DIR"
  export BACKBONE
  BASE="$(date +%Y%m%d_%H%M%S)__ckpt-cvc_adapted"
  export BASE
  ./tools/run_paper_table_sweep.sh
else
  echo "Hoàn tất train. Để sweep đầy đủ 4 α → CSV: RUN_SWEEP=1 $0"
  echo "Hoặc: BASE=... CKPT_DIR=$CKPT_CVC_DIR BACKBONE=$BACKBONE ./tools/run_paper_table_sweep.sh"
fi

echo "Xong."
