# Lệnh đầy đủ để tạo figure cho paper (LS-CRC)

Giả định: thư mục gốc repo, đã `source .venv/bin/activate`.

**Quy ước tên sweep (khớp `guide_improvements_v1.2.md`):**

- `figures/sweep/<BASE>__alpha0p01__scen4.csv` … `alpha0p15__scen4.csv`
- `BASE` = phần tên **trước** `__alpha` (ví dụ `20260407_094441__ckpt-cvc_adapted`).

Sau khi tải dữ liệu / train xong checkpoint, làm theo **A → B → C** (hoặc chỉ **B** nếu đã có CSV).

---

## A. (Tuỳ chọn) Tạo lại 4 CSV sweep — một `BASE` duy nhất

Đặt tên checkpoint và tag; chạy xong sẽ có 4 file CSV + biến `BASE` dùng cho mục B.

```bash
cd /path/to/LS-CRC
source .venv/bin/activate

mkdir -p figures/sweep figures/paper
STAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR=checkpoints_cvc_adapted
CKPT_TAG=cvc_adapted
BASE="${STAMP}__ckpt-${CKPT_TAG}"

for ALPHA in 0.01 0.05 0.10 0.15; do
  A_FILE=$(echo "$ALPHA" | sed 's/\./p/g')
  CSV="figures/sweep/${BASE}__alpha${A_FILE}__scen4.csv"
  echo "=== $CSV ==="
  python evaluate.py \
    --checkpoint-dir "$CKPT_DIR" \
    --encoder-weights imagenet \
    --alpha "$ALPHA" \
    --calibration-num-thresholds 500 \
    --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
    --scenario 'CVC-ClinicDB-ID,data/CVC-ClinicDB,data/CVC-ClinicDB' \
    --scenario 'Cross-Kvasir-cal-CVC-test,data/Kvasir-SEG,data/CVC-ClinicDB' \
    --scenario 'Cross-CVC-cal-Kvasir-test,data/CVC-ClinicDB,data/Kvasir-SEG' \
    --results-csv "$CSV"
done

echo "BASE=$BASE"
echo "Dùng BASE này trong mục B (copy giá trị BASE=...)."
```

**Lưu ý:** Nếu đã có sweep cũ, chỉ cần:

```bash
ls -1 figures/sweep/
```

Chọn một nhóm 4 file `__alpha0p01__` … `__alpha0p15__` cùng prefix, rồi đặt:

```bash
BASE=20260407_094441__ckpt-cvc_adapted   # SỬA cho đúng prefix của bạn
```

---

## B. Đồ thị Coverage / Risk vs α + scatter (LS-CRC)

Đặt `BASE` (và `SWEEP=figures/sweep`) rồi chạy **từng khối** (mỗi khối = một scenario trong paper).

```bash
cd /path/to/LS-CRC
source .venv/bin/activate

SWEEP=figures/sweep
BASE=20260407_094441__ckpt-cvc_adapted    # SỬA
P="${SWEEP}/${BASE}"
mkdir -p figures/paper
```

### B.1 — CVC in-domain

```bash
python tools/plot_risk_coverage.py \
  --dataset CVC-ClinicDB-ID \
  --method "LS-CRC (Ours)" \
  --alpha-csv 0.01 "${P}__alpha0p01__scen4.csv" \
  --alpha-csv 0.05 "${P}__alpha0p05__scen4.csv" \
  --alpha-csv 0.10 "${P}__alpha0p10__scen4.csv" \
  --alpha-csv 0.15 "${P}__alpha0p15__scen4.csv" \
  -o figures/paper/lscrc_cvc_id_risk_coverage.png \
  --dpi 300
```

Tạo thêm: `figures/paper/lscrc_cvc_id_risk_coverage_scatter.png` (script tự ghi cùng prefix `_scatter`).

### B.2 — Kvasir in-domain

```bash
python tools/plot_risk_coverage.py \
  --dataset Kvasir-SEG-ID \
  --method "LS-CRC (Ours)" \
  --alpha-csv 0.01 "${P}__alpha0p01__scen4.csv" \
  --alpha-csv 0.05 "${P}__alpha0p05__scen4.csv" \
  --alpha-csv 0.10 "${P}__alpha0p10__scen4.csv" \
  --alpha-csv 0.15 "${P}__alpha0p15__scen4.csv" \
  -o figures/paper/lscrc_kvasir_id_risk_coverage.png \
  --dpi 300
```

### B.3 — Cross: cal Kvasir → test CVC

```bash
python tools/plot_risk_coverage.py \
  --dataset Cross-Kvasir-cal-CVC-test \
  --method "LS-CRC (Ours)" \
  --alpha-csv 0.01 "${P}__alpha0p01__scen4.csv" \
  --alpha-csv 0.05 "${P}__alpha0p05__scen4.csv" \
  --alpha-csv 0.10 "${P}__alpha0p10__scen4.csv" \
  --alpha-csv 0.15 "${P}__alpha0p15__scen4.csv" \
  -o figures/paper/lscrc_cross_kvasir_cal_cvc_test_risk_coverage.png \
  --dpi 300
```

### B.4 — Cross: cal CVC → test Kvasir

```bash
python tools/plot_risk_coverage.py \
  --dataset Cross-CVC-cal-Kvasir-test \
  --method "LS-CRC (Ours)" \
  --alpha-csv 0.01 "${P}__alpha0p01__scen4.csv" \
  --alpha-csv 0.05 "${P}__alpha0p05__scen4.csv" \
  --alpha-csv 0.10 "${P}__alpha0p10__scen4.csv" \
  --alpha-csv 0.15 "${P}__alpha0p15__scen4.csv" \
  -o figures/paper/lscrc_cross_cvc_cal_kvasir_test_risk_coverage.png \
  --dpi 300
```

### B.5 — Baseline (Entropy) — cùng CSV, đổi `--method`

Chỉ cần nếu paper muốn đường con baseline trên figure (hoặc vẽ riêng file PNG).

```bash
python tools/plot_risk_coverage.py \
  --dataset CVC-ClinicDB-ID \
  --method "Entropy Threshold" \
  --alpha-csv 0.01 "${P}__alpha0p01__scen4.csv" \
  --alpha-csv 0.05 "${P}__alpha0p05__scen4.csv" \
  --alpha-csv 0.10 "${P}__alpha0p10__scen4.csv" \
  --alpha-csv 0.15 "${P}__alpha0p15__scen4.csv" \
  -o figures/paper/entropy_cvc_id_risk_coverage.png \
  --dpi 300
```

```bash
python tools/plot_risk_coverage.py \
  --dataset CVC-ClinicDB-ID \
  --method "Max-Softmax Threshold" \
  --alpha-csv 0.01 "${P}__alpha0p01__scen4.csv" \
  --alpha-csv 0.05 "${P}__alpha0p05__scen4.csv" \
  --alpha-csv 0.10 "${P}__alpha0p10__scen4.csv" \
  --alpha-csv 0.15 "${P}__alpha0p15__scen4.csv" \
  -o figures/paper/maxsoftmax_cvc_id_risk_coverage.png \
  --dpi 300
```

**Nếu sweep của bạn thiếu một α:** xóa các dòng `--alpha-csv` tương ứng (script yêu cầu ít nhất một cặp).

---

## C. Ảnh qualitative (panel 4 cột: Image | GT | Pred | Accept)

Đặt `CKPT` trùng checkpoint đã dùng lúc eval. `--calibration-num-thresholds` nên trùng `evaluate.py`.

### C.1 — CVC in-domain (cal CVC, test CVC)

```bash
mkdir -p figures/qual_cvc_id
python tools/export_qualitative_figures.py \
  --checkpoint-dir checkpoints_cvc_adapted \
  --encoder-weights imagenet \
  --cal-root data/CVC-ClinicDB \
  --test-root data/CVC-ClinicDB \
  --alpha 0.05 \
  --calibration-num-thresholds 500 \
  --num-images 8 \
  --out-dir figures/qual_cvc_id \
  --prefix lscrc_cvc_id \
  --dpi 200
```

### C.2 — Kvasir in-domain

```bash
mkdir -p figures/qual_kvasir_id
python tools/export_qualitative_figures.py \
  --checkpoint-dir checkpoints_cvc_adapted \
  --encoder-weights imagenet \
  --cal-root data/Kvasir-SEG \
  --test-root data/Kvasir-SEG \
  --alpha 0.05 \
  --calibration-num-thresholds 500 \
  --num-images 8 \
  --out-dir figures/qual_kvasir_id \
  --prefix lscrc_kvasir_id \
  --dpi 200
```

### C.3 — Cross: cal Kvasir → test CVC (minh họa OOD / shift)

```bash
mkdir -p figures/qual_cross_kvasir_cal_cvc
python tools/export_qualitative_figures.py \
  --checkpoint-dir checkpoints_cvc_adapted \
  --encoder-weights imagenet \
  --cal-root data/Kvasir-SEG \
  --test-root data/CVC-ClinicDB \
  --alpha 0.05 \
  --calibration-num-thresholds 500 \
  --num-images 8 \
  --out-dir figures/qual_cross_kvasir_cal_cvc \
  --prefix lscrc_cal_kvasir_test_cvc \
  --dpi 200
```

### C.4 — Cross: cal CVC → test Kvasir

```bash
mkdir -p figures/qual_cross_cvc_cal_kvasir
python tools/export_qualitative_figures.py \
  --checkpoint-dir checkpoints_cvc_adapted \
  --encoder-weights imagenet \
  --cal-root data/CVC-ClinicDB \
  --test-root data/Kvasir-SEG \
  --alpha 0.05 \
  --calibration-num-thresholds 500 \
  --num-images 8 \
  --out-dir figures/qual_cross_cvc_cal_kvasir \
  --prefix lscrc_cal_cvc_test_kvasir \
  --dpi 200
```

### C.5 — DeepLab (nếu checkpoint train bằng DeepLab)

```bash
mkdir -p figures/qual_deeplab_cvc
python tools/export_qualitative_figures.py \
  --checkpoint-dir checkpoints_deeplab \
  --backbone deeplabv3plus \
  --encoder-weights imagenet \
  --cal-root data/CVC-ClinicDB \
  --test-root data/CVC-ClinicDB \
  --alpha 0.05 \
  --calibration-num-thresholds 500 \
  --num-images 6 \
  --out-dir figures/qual_deeplab_cvc \
  --prefix deeplab_lscrc_cvc_id \
  --dpi 200
```

---

## D. Một khối: chỉ vẽ lại figure từ sweep đã có (thay `BASE`)

```bash
cd /path/to/LS-CRC && source .venv/bin/activate
BASE=20260407_094441__ckpt-cvc_adapted
P="figures/sweep/${BASE}"
mkdir -p figures/paper

for DATASET_OUT in \
  "CVC-ClinicDB-ID:lscrc_cvc_id" \
  "Kvasir-SEG-ID:lscrc_kvasir_id" \
  "Cross-Kvasir-cal-CVC-test:lscrc_cross_kvasir_cal_cvc" \
  "Cross-CVC-cal-Kvasir-test:lscrc_cross_cvc_cal_kvasir"
do
  DS="${DATASET_OUT%%:*}"
  TAG="${DATASET_OUT##*:}"
  python tools/plot_risk_coverage.py \
    --dataset "$DS" \
    --method "LS-CRC (Ours)" \
    --alpha-csv 0.01 "${P}__alpha0p01__scen4.csv" \
    --alpha-csv 0.05 "${P}__alpha0p05__scen4.csv" \
    --alpha-csv 0.10 "${P}__alpha0p10__scen4.csv" \
    --alpha-csv 0.15 "${P}__alpha0p15__scen4.csv" \
    -o "figures/paper/${TAG}_risk_coverage.png" \
    --dpi 300
done
```

---

## File output tóm tắt

| Loại | Đường dẫn mẫu |
|------|----------------|
| Line + risk vs α | `figures/paper/<name>_risk_coverage.png` |
| Scatter risk–coverage | cùng tên + `_scatter.png` |
| Qualitative | `figures/qual_<...>/<prefix>_000.png`, … |

---

## Lưu ý

- **zsh/bash:** không dùng ký tự `<` trong tên file/path (shell hiểu redirect).  
- **CSV sweep chỉ có `grid1000` trong tên** (ví dụ `...alpha0p05__grid1000__scen4.csv`): vẫn dùng được; chỉ cần đổi **tên file** trong `--alpha-csv` cho khớp `ls figures/sweep/`.  
- Checkpoint multi-seed: đổi `CKPT_DIR` / `evaluate` sweep rồi vẽ lại mục B; qualitative đổi `--checkpoint-dir checkpoints_cvc_adapted_seed0` v.v.
