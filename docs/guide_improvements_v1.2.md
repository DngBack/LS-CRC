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

## 5. Cải tiến cần chỉnh **code** (chưa có lệnh một dòng)

Các mục sau được mô tả trong [report_experiments_eval.md](report_experiments_eval.md); triển khai trong repo tương lai:

| Mục | Gợi ý file / việc làm |
|-----|------------------------|
| Temperature scaling trên score rejector | Thêm bước hiệu chỉnh trên tập `cal` trước `calibrate_threshold`, hoặc wrapper trong `calibrate.py`. |
| Lưới τ mịn / tìm nhị phân | `calibrate.py`: thay `np.linspace(0.01, 0.99, 100)` bằng lưới dày hơn hoặc binary search trên risk. |
| Cảnh báo calibration failsafe / coverage suy biến | `evaluate.py` hoặc `calibrate.py`: nếu `tau` gần 0.999 hoặc mean coverage &lt; 1%, in WARNING. |
| Huấn luyện đa miền (một `DataLoader` gộp Kvasir+CVC) | Mở rộng `train.py` / `dataset.py` (hai root hoặc `ConcatDataset`). |

---

## 6. Đẩy PR sau khi commit v1.2

```bash
git checkout -b release/v1.2
git add -A
git status   # kiểm tra không commit nhầm .pth (đã ignore)
git commit -m "release: v1.2 — CLI train/eval, splits, docs, resume checkpoints"
git push -u origin release/v1.2
```

Trên GitHub: **Compare & pull request** → base `main`, compare `release/v1.2`.  
Hoặc CLI: `gh pr create --base main --head release/v1.2 --title "release: LS-CRC v1.2" --body-file docs/report_v1.2.md`
