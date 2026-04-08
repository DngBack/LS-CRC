# Hướng dẫn chạy các cải tiến (LS-CRC v1.2)

Giả định bạn đang ở thư mục gốc repo, đã `source .venv/bin/activate` (hoặc tương đương) và đã `pip install -r requirements.txt`.

---

## 0. Chuẩn bị dữ liệu & split

Sau khi tải hoặc khi số ảnh trên đĩa không khớp `train/val/cal/test.txt`:

```bash
python download_data.py
# hoặc chỉ tạo lại split từ thư mục images/ hiện có:
python download_data.py --regenerate-splits data/CVC-ClinicDB
python download_data.py --regenerate-splits data/Kvasir-SEG --train 500 --val 80 --cal 150
```

---

## 1. Huấn luyện dài hơn & chỉnh siêu tham số (cùng miền Kvasir)

```bash
python train.py \
  --data-root data/Kvasir-SEG \
  --epochs-backbone 100 \
  --epochs-rejector 50 \
  --epochs-joint 50 \
  --lambda-smooth 0.5 \
  --lambda-surrogate 1.0 \
  --lr-backbone 1e-4 \
  --lr-rejector 1e-4 \
  --lr-joint 5e-5 \
  --encoder-weights imagenet \
  --checkpoint-dir checkpoints
```

**Grid nhỏ (ví dụ chỉ đổi λ):**

```bash
for s in 0.25 0.5 1.0; do
  python train.py --lambda-smooth "$s" --checkpoint-dir "checkpoints_ls_${s}"
done
```

Sau mỗi lần train, chạy `evaluate.py` trỏ `--checkpoint-dir` tương ứng để so sánh CSV.

---

## 2. Fine-tune rejector / joint trên CVC (thích ứng miền)

Giữ backbone đã học trên Kvasir, **chỉ** huấn luyện rejector + joint trên `data/CVC-ClinicDB` (giảm hiện tượng coverage ~0 khi cal trên CVC):

```bash
python train.py \
  --data-root data/CVC-ClinicDB \
  --resume-backbone checkpoints/backbone.pth \
  --resume-rejector checkpoints/rejector.pth \
  --epochs-backbone 0 \
  --epochs-rejector 40 \
  --epochs-joint 30 \
  --lr-rejector 5e-5 \
  --lr-joint 2e-5 \
  --encoder-weights imagenet \
  --backbone unet \
  --checkpoint-dir checkpoints_cvc_adapted
```

Sau đó đánh giá:

```bash
python evaluate.py \
  --checkpoint-dir checkpoints_cvc_adapted \
  --encoder-weights imagenet \
  --scenario 'CVC-ClinicDB-ID,data/CVC-ClinicDB,data/CVC-ClinicDB' \
  --results-csv eval_after_cvc_adapt.csv
```

**Lưu ý:** `--backbone`, `--encoder-name`, `--encoder-weights` phải **trùng** lúc train Kvasir và lúc resume / evaluate.

---

## 3. Kiến trúc khác (DeepLabV3+)

```bash
python train.py --backbone deeplabv3plus --encoder-weights imagenet --checkpoint-dir checkpoints_deeplab

python evaluate.py --backbone deeplabv3plus --encoder-weights imagenet --checkpoint-dir checkpoints_deeplab \
  --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
  --results-csv eval_deeplab.csv
```

---

## 4. Bảng thí nghiệm đầy đủ (multi-scenario)

```bash
python evaluate.py \
  --checkpoint-dir checkpoints \
  --encoder-weights imagenet \
  --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
  --scenario 'CVC-ClinicDB-ID,data/CVC-ClinicDB,data/CVC-ClinicDB' \
  --scenario 'Cross-Kvasir-cal-CVC-test,data/Kvasir-SEG,data/CVC-ClinicDB' \
  --results-csv experiments_full.csv
```

---

## 5. Sweep `α` — quy ước tên file (tránh đè) + lệnh mẫu

Mặc định `evaluate.py` dùng `--alpha 0.05`. Mỗi lần đổi α cần **một CSV riêng** với tên **duy nhất** (checkpoint + α + số scenario + thời gian).

**Gợi ý cấu trúc tên:**

`figures/sweep/<STAMP>__ckpt-<TÊN_NGẮN>__alpha<0p05>__scen4.csv`

- `STAMP`: `$(date +%Y%m%d_%H%M%S)` — tránh trùng khi chạy lại nhiều lần trong ngày.  
- `TÊN_NGẮN`: bạn tự đặt, ví dụ `cvc_adapted`, `kvasir_only`, `ls0p25`.  
- `0p05`: thay dấu `.` bằng `p` trong α (0.05 → `0p05`) để tên file an toàn.  
- `scen4`: 4 dòng `--scenario` (đổi thành `scen3` nếu bạn ít hơn).

**Một lần chạy (α = 0.10):**

```bash
mkdir -p figures/sweep
STAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR=checkpoints_cvc_adapted
CKPT_TAG=cvc_adapted
ALPHA=0.10
A_FILE=$(echo "$ALPHA" | sed 's/\./p/g')

python evaluate.py \
  --checkpoint-dir "$CKPT_DIR" \
  --encoder-weights imagenet \
  --alpha "$ALPHA" \
  --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
  --scenario 'CVC-ClinicDB-ID,data/CVC-ClinicDB,data/CVC-ClinicDB' \
  --scenario 'Cross-Kvasir-cal-CVC-test,data/Kvasir-SEG,data/CVC-ClinicDB' \
  --scenario 'Cross-CVC-cal-Kvasir-test,data/CVC-ClinicDB,data/Kvasir-SEG' \
  --results-csv "figures/sweep/${STAMP}__ckpt-${CKPT_TAG}__alpha${A_FILE}__scen4.csv"
```

**Sweep nhiều α (vòng lặp — mỗi α một file, không đè):**

```bash
mkdir -p figures/sweep
STAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR=checkpoints_cvc_adapted
CKPT_TAG=cvc_adapted

for ALPHA in 0.01 0.05 0.10 0.15; do
  A_FILE=$(echo "$ALPHA" | sed 's/\./p/g')
  CSV="figures/sweep/${STAMP}__ckpt-${CKPT_TAG}__alpha${A_FILE}__scen4.csv"
  echo "=== Writing $CSV ==="
  python evaluate.py \
    --checkpoint-dir "$CKPT_DIR" \
    --encoder-weights imagenet \
    --alpha "$ALPHA" \
    --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
    --scenario 'CVC-ClinicDB-ID,data/CVC-ClinicDB,data/CVC-ClinicDB' \
    --scenario 'Cross-Kvasir-cal-CVC-test,data/Kvasir-SEG,data/CVC-ClinicDB' \
    --scenario 'Cross-CVC-cal-Kvasir-test,data/CVC-ClinicDB,data/Kvasir-SEG' \
    --results-csv "$CSV"
done
```

Đổi `CKPT_DIR` / `CKPT_TAG` khi so sánh checkpoint khác; mỗi lần so sánh **checkpoint khác** nên dùng `CKPT_TAG` khác (hoặc `STAMP` mới) để tên vẫn duy nhất.

### 5.1 Vẽ đồ thị risk–coverage ngay sau sweep (copy-paste)

Chạy **ngay trong cùng terminal** sau khối sweep ở trên (biến `STAMP`, `CKPT_TAG` vẫn còn). **Không cần** sửa đường dẫn từng file.

```bash
mkdir -p figures/paper
SWEEP_DIR=figures/sweep
P="${SWEEP_DIR}/${STAMP}__ckpt-${CKPT_TAG}"

python tools/plot_risk_coverage.py \
  --dataset CVC-ClinicDB-ID \
  --method "LS-CRC (Ours)" \
  --alpha-csv 0.01 "${P}__alpha0p01__scen4.csv" \
  --alpha-csv 0.05 "${P}__alpha0p05__scen4.csv" \
  --alpha-csv 0.10 "${P}__alpha0p10__scen4.csv" \
  --alpha-csv 0.15 "${P}__alpha0p15__scen4.csv" \
  -o figures/paper/risk_coverage_vs_alpha__cvc__lscrc.png
```

Nếu sweep của bạn **không** có α = 0.01 (hoặc thiếu 0.15), xóa các dòng `--alpha-csv` tương ứng cho khớp vòng `for ALPHA in ...`.

### 5.2 Vẽ đồ thị sau khi mở terminal mới (chỉ sửa một dòng `BASE`)

```bash
cd ~/Desktop/LS-CRC
source .venv/bin/activate

ls -1 figures/sweep/

SWEEP_DIR=figures/sweep
BASE=20250322_153045__ckpt-cvc_adapted
P="${SWEEP_DIR}/${BASE}"
```

Sửa **một dòng** `BASE=...`: copy từ `ls` phần tên file **trước** `__alpha` (ví dụ file là `20250322_153045__ckpt-cvc_adapted__alpha0p05__scen4.csv` thì `BASE=20250322_153045__ckpt-cvc_adapted`).

```bash
mkdir -p figures/paper
python tools/plot_risk_coverage.py \
  --dataset CVC-ClinicDB-ID \
  --method "LS-CRC (Ours)" \
  --alpha-csv 0.01 "${P}__alpha0p01__scen4.csv" \
  --alpha-csv 0.05 "${P}__alpha0p05__scen4.csv" \
  --alpha-csv 0.10 "${P}__alpha0p10__scen4.csv" \
  --alpha-csv 0.15 "${P}__alpha0p15__scen4.csv" \
  -o figures/paper/risk_coverage_vs_alpha__cvc__lscrc.png
```

### 5.3 Một khối: sweep 4 α + plot (copy-paste toàn bộ)

Chạy từ thư mục gốc repo (sửa `cd` nếu cần). Cuối khối sẽ có CSV trong `figures/sweep/` và PNG trong `figures/paper/`.

```bash
cd ~/Desktop/LS-CRC
source .venv/bin/activate

mkdir -p figures/sweep figures/paper
STAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR=checkpoints_cvc_adapted
CKPT_TAG=cvc_adapted

for ALPHA in 0.01 0.05 0.10 0.15; do
  A_FILE=$(echo "$ALPHA" | sed 's/\./p/g')
  CSV="figures/sweep/${STAMP}__ckpt-${CKPT_TAG}__alpha${A_FILE}__scen4.csv"
  echo "=== $CSV ==="
  python evaluate.py \
    --checkpoint-dir "$CKPT_DIR" \
    --encoder-weights imagenet \
    --alpha "$ALPHA" \
    --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
    --scenario 'CVC-ClinicDB-ID,data/CVC-ClinicDB,data/CVC-ClinicDB' \
    --scenario 'Cross-Kvasir-cal-CVC-test,data/Kvasir-SEG,data/CVC-ClinicDB' \
    --scenario 'Cross-CVC-cal-Kvasir-test,data/CVC-ClinicDB,data/Kvasir-SEG' \
    --results-csv "$CSV"
done

P="figures/sweep/${STAMP}__ckpt-${CKPT_TAG}"
python tools/plot_risk_coverage.py \
  --dataset CVC-ClinicDB-ID \
  --method "LS-CRC (Ours)" \
  --alpha-csv 0.01 "${P}__alpha0p01__scen4.csv" \
  --alpha-csv 0.05 "${P}__alpha0p05__scen4.csv" \
  --alpha-csv 0.10 "${P}__alpha0p10__scen4.csv" \
  --alpha-csv 0.15 "${P}__alpha0p15__scen4.csv" \
  -o figures/paper/risk_coverage_vs_alpha__cvc__lscrc.png

echo "Done. CSV: figures/sweep/${STAMP}__ckpt-${CKPT_TAG}__alpha*.csv"
echo "Figures: figures/paper/risk_coverage_vs_alpha__cvc__lscrc.png (+ _scatter.png)"
```

---

## 5b. Figure cho paper — script có sẵn trong repo

**Không bắt buộc** nếu paper chỉ cần **bảng**; **nên có** 1–2 figure minh họa trade-off và qualitative.

| Loại | Script | Ý nghĩa |
|------|--------|---------|
| **Đồ thị α → Coverage / Risk** + scatter risk–coverage | `tools/plot_risk_coverage.py` | Đọc các CSV sweep ở trên; cần đúng chuỗi `Dataset` và `Method` như trong CSV. |
| **Ảnh qualitative** (RGB, GT, pred, vùng accept LS-CRC) | `tools/export_qualitative_figures.py` | Hiệu chỉnh τ trên `--cal-root`, xuất PNG từ `--test-root`. |

### 5b.1 Đồ thị từ nhiều CSV (nhắc nhanh)

Ưu tiên dùng **§5.1** (cùng shell sau sweep) hoặc **§5.2** (một biến `BASE`) — copy-paste, không cần dán từng path.

**Cảnh báo zsh:** Không dùng ký tự `<` trong đường dẫn (shell hiểu là redirect).

- Xuất **hai** file: `...png` và `..._scatter.png`.  
- Đổi `--dataset` (ví dụ `Kvasir-SEG-ID`, `Cross-Kvasir-cal-CVC-test`) và tên `-o ...png` nếu vẽ scenario khác.  
- Path sai → `plot_risk_coverage.py` sẽ gợi ý liệt kê `figures/sweep/`.

### 5b.2 Ảnh qualitative (LS-CRC)

```bash
cd ~/Desktop/LS-CRC
source .venv/bin/activate
mkdir -p figures/qual_cvc_in

python tools/export_qualitative_figures.py \
  --checkpoint-dir checkpoints_cvc_adapted \
  --encoder-weights imagenet \
  --cal-root data/CVC-ClinicDB \
  --test-root data/CVC-ClinicDB \
  --alpha 0.05 \
  --num-images 6 \
  --out-dir figures/qual_cvc_in \
  --prefix lscrc_cvc_cal-on-cvc_test-on-cvc
```

- Panel: **Image | GT | Pred (đỏ) | Accept (xanh) + heatmap score**.  
- Đổi `cal-root` / `test-root` cho cross-domain (ví dụ cal Kvasir, test CVC) nếu cần minh họa OOD.  
- Thêm `--backbone deeplabv3plus` nếu checkpoint train bằng DeepLab.

### 5b.3 Chưa có script tự động (tuỳ chọn sau)

| Loại | Ghi chú |
|------|---------|
| **Histogram / boxplot** risk theo ảnh | Cần xuất per-image risk từ `evaluate.py` hoặc script riêng — hiện bảng chỉ có mean trên tập. |

---

## 6. Cải tiến cần chỉnh **code** (chưa có lệnh một dòng)

Các mục sau được mô tả trong [report_experiments_eval.md](report_experiments_eval.md); triển khai trong repo tương lai:

| Mục | Gợi ý file / việc làm |
|-----|------------------------|
| Temperature scaling trên score rejector | Thêm bước hiệu chỉnh trên tập `cal` trước `calibrate_threshold`, hoặc wrapper trong `calibrate.py`. |
| Lưới τ mịn / tìm nhị phân | `calibrate.py`: thay `np.linspace(0.01, 0.99, 100)` bằng lưới dày hơn hoặc binary search trên risk. |
| Cảnh báo calibration failsafe / coverage suy biến | `evaluate.py` hoặc `calibrate.py`: nếu `tau` gần 0.999 hoặc mean coverage &lt; 1%, in WARNING. |
| Huấn luyện đa miền (một `DataLoader` gộp Kvasir+CVC) | Mở rộng `train.py` / `dataset.py` (hai root hoặc `ConcatDataset`). |

---

## 7. `evaluate.py` có vẻ “treo” rồi `KeyboardInterrupt`

Lần đầu chạy, Python phải nạp **PyTorch + torchvision** (và chuỗi import phụ như `sympy`) — có thể **20–60 giây** tùy máy. Nếu bạn bấm **Ctrl+C** trong lúc đó, traceback sẽ dừng giữa chừng ở `import torchvision` / `sympy` nhưng **không phải lỗi code**.

- Đợi hết đợt import; lần sau thường nhanh hơn (OS cache).  
- Code đã lùi import `data.dataset`/`torchvision` đến lúc thật sự tạo DataLoader (sau khi load backbone), để bạn thấy log `Loading models...` sớm hơn.

---

## 8. Đẩy PR sau khi commit v1.2

```bash
git checkout -b release/v1.2
git add -A
git status   # kiểm tra không commit nhầm .pth (đã ignore)
git commit -m "release: v1.2 — CLI train/eval, splits, docs, resume checkpoints"
git push -u origin release/v1.2
```

Trên GitHub: **Compare & pull request** → base `main`, compare `release/v1.2`.  
Hoặc CLI: `gh pr create --base main --head release/v1.2 --title "release: LS-CRC v1.2" --body-file docs/report_v1.2.md`

