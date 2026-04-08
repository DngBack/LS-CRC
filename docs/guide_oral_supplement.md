# Hướng dẫn bổ sung thí nghiệm (seed / CI, calibration, stress test, baseline, figure)

Giả định repo gốc: `LS-CRC`, đã `source .venv/bin/activate` và cài `requirements.txt`.

**Những gì đã được thêm vào code để hỗ trợ guide này**

- `train.py --seed`: cố định RNG + thứ tự shuffle batch train.
- `download_data.py --split-seed`: tạo split train/val/cal/test khác nhau khi regenerate.
- `evaluate.py --calibration-num-thresholds` (mặc định **500**): lưới τ mịn hơn (trước đây 100 điểm trong `calibrate.py`).
- `tools/aggregate_multiseed_csv.py`: gom nhiều CSV eval → mean/std theo `Dataset` × `Method`.
- `tools/export_qualitative_figures.py --calibration-num-thresholds`: khớp τ* với `evaluate.py`.

---

## Phần A — Calibration “sạch” hơn (chạy trước khi so sánh với paper cũ)

1. **Dùng lưới dày** (mặc định 500). Nếu vẫn thấy hai α cho cùng τ* hoặc nhảy Plain sớm, thử **1000**:

```bash
python evaluate.py \
  --checkpoint-dir checkpoints_cvc_adapted \
  --encoder-weights imagenet \
  --alpha 0.05 \
  --calibration-num-thresholds 1000 \
  --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
  --scenario 'CVC-ClinicDB-ID,data/CVC-ClinicDB,data/CVC-ClinicDB' \
  --scenario 'Cross-Kvasir-cal-CVC-test,data/Kvasir-SEG,data/CVC-ClinicDB' \
  --scenario 'Cross-CVC-cal-Kvasir-test,data/CVC-ClinicDB,data/Kvasir-SEG' \
  --results-csv figures/sweep/$(date +%Y%m%d_%H%M%S)__ckpt-cvc_adapted__alpha0p05__grid1000__scen4.csv
```

2. **Trong paper / slide**: nêu rõ *finite-sample target* `α − 1/n`, lưới τ trên `[0.01, 0.99]`, và **số điểm lưới**; thừa nhận khi risk–coverage không đơn điệu theo τ thì greedy trên lưới có thể lệch (đây là limitation chung, không chỉ repo).

---

## Phần B — Nhiều seed (ước lượng độ bền, CI / mean ± std)

**Cách 1 (khuyến nghị — rẻ hơn): chỉ đổi seed huấn luyện, giữ split cố định**

Mỗi seed = một full train trên Kvasir + (tuỳ pipeline) adapt CVC. Lưu checkpoint riêng.

```bash
SEEDS="0 1 2 3 4"
for S in $SEEDS; do
  python train.py \
    --data-root data/Kvasir-SEG \
    --seed "$S" \
    --checkpoint-dir "checkpoints_kvasir_seed${S}"

  python train.py \
    --data-root data/CVC-ClinicDB \
    --resume-backbone "checkpoints_kvasir_seed${S}/backbone.pth" \
    --resume-rejector "checkpoints_kvasir_seed${S}/rejector.pth" \
    --epochs-backbone 0 \
    --epochs-rejector 40 \
    --epochs-joint 30 \
    --lr-rejector 5e-5 \
    --lr-joint 2e-5 \
    --encoder-weights imagenet \
    --backbone unet \
    --seed "$S" \
    --checkpoint-dir "checkpoints_cvc_adapted_seed${S}"
done
```

Sau đó với **mỗi** checkpoint, chạy eval **cùng α** (ví dụ 0.05), tên file có `seed${S}`:

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
ALPHA=0.05
A=$(echo "$ALPHA" | tr . p)
for S in $SEEDS; do
  python evaluate.py \
    --checkpoint-dir "checkpoints_cvc_adapted_seed${S}" \
    --encoder-weights imagenet \
    --alpha "$ALPHA" \
    --calibration-num-thresholds 500 \
    --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
    --scenario 'CVC-ClinicDB-ID,data/CVC-ClinicDB,data/CVC-ClinicDB' \
    --scenario 'Cross-Kvasir-cal-CVC-test,data/Kvasir-SEG,data/CVC-ClinicDB' \
    --scenario 'Cross-CVC-cal-Kvasir-test,data/CVC-ClinicDB,data/Kvasir-SEG' \
    --results-csv "figures/multiseed/${STAMP}_seed${S}__alpha${A}__scen4.csv"
done
```

**Gom mean ± std** (cùng pattern tên file — sửa glob cho khớp thư mục bạn):

```bash
mkdir -p figures/multiseed
python tools/aggregate_multiseed_csv.py \
  "figures/multiseed/${STAMP}_seed*__alpha${A}__scen4.csv" \
  -o "figures/multiseed/${STAMP}__summary_alpha${A}.csv"
```

Bảng paper: dùng cột `Coverage_mean`, `Coverage_std`, `Expected Risk_mean`, `Expected Risk_std` (chú ý: với **1 run**, std trống/NaN là bình thường).

**Cách 2 (nặng hơn): đổi cả split (`--split-seed`)**

Chỉ khi bạn muốn báo cáo **nhạy với random split**. Mỗi `split-seed` cần **bộ `*.txt` riêng** — cách sạch nhất là **sao chép dataset** sang thư mục khác hoặc script tạo `data/Kvasir-SEG_seed7/` (tránh ghi đè `cal.txt` của run khác).

Ví dụ (một dataset, hiểu ý tưởng):

```bash
python download_data.py --regenerate-splits data/Kvasir-SEG --split-seed 7 \
  --train 500 --val 80 --cal 150
# sau đó train + eval như trên (checkpoint dir khác)
```

Lặp vài `split-seed` → nhiều bảng → có thể aggregate thủ công hoặc thống nhất tên CSV rồi dùng `aggregate_multiseed_csv.py`.

**Lưu ý GPU:** `torch` vẫn có thể khác nhau nhẹ giữa lần chạy dù đã `--seed`; nếu cần tối đa ổn định, thử `--num-workers 0` khi train (chậm hơn).

---

## Phần C — Stress test (cal nhỏ, domain)

**Cal nhỏ**: giảm `n_cal` khi regenerate (giữ train/val/test hợp lý).

```bash
python download_data.py --regenerate-splits data/Kvasir-SEG --train 500 --val 80 --cal 80
python download_data.py --regenerate-splits data/CVC-ClinicDB --cal 30
```

Sau đó train (hoặc chỉ eval nếu chỉ đổi cal) và so sánh **cùng α** với bản “cal đầy đủ”. Mục tiêu: một đoạn *“under small calibration set, LS-CRC is more stable than …”* hoặc thừa nhận degradation.

**Domain / cross** (bạn đã có 4 scenario): nhấn mạnh trong text hàng **Cross-Kvasir-cal-CVC-test** và **Cross-CVC-cal-Kvasir-test** là stress; có thể thêm một dòng **DeepLab** (`guide_improvements_v1.2.md` §3) như stress kiến trúc.

---

## Phần D — “Baseline literature” (ý nghĩa + hướng làm trong repo)

**Baseline** = cùng task, cùng metric risk/coverage, cùng protocol cal trên `split=cal`.

Repo hiện có: Plain, Entropy, Max-Softmax (đều qua `calibrate_threshold` nếu `requires_cal`).

**Để “đủ oral” hơn**, thường thêm **một** trong các dòng sau (chọn 1 phù hợp paper, implement ngoài scope file này nếu cần):

1. **Post-hoc calibration trên cal** (temperature scaling trên logits **trước** khi entropy / CRC) — vẫn là heuristic nhưng literature quen thuộc; cần lưu `T` từ cal rồi áp vào `evaluate_model` / `evaluate_risk`.
2. **Baseline từ bài conformal / selective segmentation** (nếu có): tái hiện cùng backbone hoặc trích score rồi plug vào cùng `calibrate_threshold` nếu score đơn điệu với risk trên cal.

Trong *Related work* + *Discussion*: nói rõ Entropy/Max-softmax là **score-based selective** được **CRC trên cal**; baseline mạnh hơn là **phần mở rộng** chứ không thay thế claim chính về rejector học được.

---

## Phần E — Ablation (train lại — khớp `train.py`)

| Ablation | Lệnh (đổi `--checkpoint-dir`) |
|----------|--------------------------------|
| Không smooth | `--lambda-smooth 0` |
| Không surrogate joint | `--lambda-surrogate 0` |
| Không CVC-adapt | So sánh `checkpoints_kvasir_only` với `checkpoints_cvc_adapted` (cùng eval) |
| DeepLab | `--backbone deeplabv3plus` end-to-end hoặc chỉ eval nếu đã train |

Sau mỗi ablation: một lần `evaluate.py` **giống hệt** bản full (cùng α, `--calibration-num-thresholds`, 4 scenario).

---

## Phần F — Figure qualitative + text

**Export panel** (GT | pred | acceptance):

```bash
mkdir -p figures/qual_cvc
python tools/export_qualitative_figures.py \
  --checkpoint-dir checkpoints_cvc_adapted \
  --encoder-weights imagenet \
  --cal-root data/CVC-ClinicDB \
  --test-root data/CVC-ClinicDB \
  --alpha 0.05 \
  --calibration-num-thresholds 500 \
  --num-images 8 \
  --out-dir figures/qual_cvc \
  --prefix lscrc_cvc_id
```

**Text gợi ý (1 đoạn figure caption):** *“Under risk budget α on the calibration set, LS-CRC yields spatially coherent accept regions; failures occur when … (tóm tắt 1 case).”*

---

## Thứ tự chạy gợi ý (tóm tắt)

1. **Eval lại với `--calibration-num-thresholds` 500–1000** + sweep α (so với kết quả cũ 100 điểm).
2. **5 seed** train → 5 eval @ α=0.05 → `aggregate_multiseed_csv.py`.
3. **1 stress** (cal nhỏ) @ α=0.05.
4. **1–2 ablation** (λ hoặc no-adapt).
5. **Qualitative PNG** + cập nhật bảng chính / limitation trong manuscript.

---

*Phiên bản tài liệu này khớp với `calibrate.py`, `evaluate.py`, `train.py`, `download_data.py` sau khi bổ sung các flag trên.*
