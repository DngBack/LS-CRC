# Báo cáo thực nghiệm LS-CRC — pipeline đánh giá & figure (10/04/2026)

Tài liệu tổng hợp **các lệnh đã chạy**, **toàn bộ artifact (output) theo thư mục**, **ý nghĩa từng loại file**, và **phân tích số liệu** cho lần chạy chính dùng checkpoint **`checkpoints_cvc_adapted`** (DeepLabV3+), sweep timestamp **`20260409_144715__ckpt-cvc_adapted`**, lưới calibration **1000** bước, bốn scenario (Kvasir-ID, CVC-ID, cross Kvasir→CVC, cross CVC→Kvasir).

---

## 1. Bản đồ output: thư mục, ý nghĩa, dùng để làm gì

### 1.1 `checkpoints_cvc_adapted/`

| File | Ý nghĩa | Dùng cho |
|------|---------|----------|
| `backbone.pth` | Trọng số segmentation (DeepLabV3+), thường bắt đầu từ pretrain Kvasir rồi giữ cố định khi adapt CVC | `train.py` (lưu), `evaluate.py`, `export_qualitative_figures.py` |
| `rejector.pth` | Trọng số rejector (acceptance map) sau adapt trên CVC | Cùng các script trên |

**Lưu ý:** Mọi lệnh eval/figure phải khớp `--backbone deeplabv3plus` và `--encoder-weights imagenet` với lúc train.

---

### 1.2 `figures/sweep/`

**Quy ước tên (khuyến nghị):**

`figures/sweep/<STAMP>__ckpt-<tag>__alpha<0p05>__grid1000__scen4.csv`

- `STAMP`: thời gian hoặc nhãn run (ví dụ `20260409_144715`).
- `alpha0p05`: ngân sách rủi ro α (dấu `.` → `p`).
- `grid1000`: độ mịn lưới τ khi calibration (khớp `--calibration-num-thresholds 1000`).
- `scen4`: bốn dòng `--scenario` trong `evaluate.py`.

**Nội dung CSV:** mỗi file tương ứng **một α**. Cột gồm:

| Cột | Ý nghĩa |
|-----|---------|
| `Dataset` | Nhãn scenario (ví dụ `Kvasir-SEG-ID`, `CVC-ClinicDB-ID`, …) |
| `Method` | Plain / Entropy / Max-Softmax / Standard CRC (Bates) / Spatial-weighted CP / LS-CRC |
| `Dice` | Dice trung bình test (cùng backbone → giống nhau giữa các method selective) |
| `Coverage` | Tỷ lệ pixel được chấp nhận (trung bình theo ảnh) |
| `Expected Risk` | Rủi ro selective có trọng số không gian (boundary-weighted) trên test |
| `Risk Std`, `Worst 10%`, `CVaR_0.9`, `Worst Group` | Thống kê đuôi và subgroup (theo tag trong dataloader) |

**Dùng để:** bảng chính paper (chọn một α, thường **0.05**); so sánh method; vẽ đồ thị risk–coverage theo α (nối 4 file α khác nhau).

**Các nhóm file có thể có trên đĩa (lịch sử run):**

- **`20260409_144715__ckpt-cvc_adapted__alpha*{0p01,0p05,0p10,0p15}__grid1000__scen4.csv`** — run “paper” gần nhất (grid 1000).
- Các prefix cũ (`20260407_*`, `20260322_*`, …) — sweep trước đó; chỉ dùng khi so sánh lịch sử hoặc tái phân tích.
- **`_sanity.csv`** — thường là một lần `evaluate` nhanh để kiểm tra Dice (có thể ghi đè khi chạy lại).

---

### 1.3 `figures/paper/`

| Nội dung | Ý nghĩa | Dùng cho |
|----------|---------|----------|
| `table_main_alpha0p05_all_methods_scen4.csv` | Bản sao (hoặc tương đương) của sweep **α=0.05** để mở nhanh khi viết bảng | Table 1 / so sánh baseline trong Word/LaTeX |
| `lscrc_*_risk_coverage_vs_alpha.png` (+ `_scatter.png`) | Đồ thị Coverage và Expected Risk theo α cho từng `Dataset`, method **LS-CRC** (mặc định) | Figure risk–coverage trong paper |
| `risk_coverage_vs_alpha__cvc__lscrc.png` (symlink) | Trỏ tới figure CVC-ID để khớp snippet trong `docs/paper_assets/` | `\includegraphics` trong LaTeX |

**Tạo / cập nhật figure:**  
`BASE=20260409_144715__ckpt-cvc_adapted ./tools/render_paper_figures_from_sweep.sh`  
Vẽ baseline khác không đè file: `METHOD="Entropy Threshold" OUT_PREFIX=entropy_ BASE=... ./tools/render_paper_figures_from_sweep.sh`.

*Nếu sau khi clone máy khác chưa có PNG trong `figures/paper/`, chạy lại script trên (cần `matplotlib`, `pandas`).*

---

### 1.4 `figures/qual_cvc_in/`

| Pattern tên | Ý nghĩa |
|-------------|---------|
| `lscrc_cvc_id_20260409_*.png` | Bản qualitative **cũ** (layout 4 panel, nếu đã export trước khi nâng cấp script) |
| `lscrc_cvc_id_states_v2_*.png` | Bản **mới**: 6 panel — ảnh, GT, pred, score heatmap, accept mask, **bản đồ trạng thái màu** (TP/FP/FN/abstain) |

**Dùng cho:** figure qualitative / minh hoạ LS-CRC trên CVC in-domain (`cal` & `test` cùng CVC).

**Lệnh tạo (bản states v2):**

```bash
python tools/export_qualitative_figures.py \
  --checkpoint-dir checkpoints_cvc_adapted \
  --backbone deeplabv3plus \
  --encoder-weights imagenet \
  --cal-root data/CVC-ClinicDB \
  --test-root data/CVC-ClinicDB \
  --alpha 0.05 \
  --calibration-num-thresholds 1000 \
  --num-images 6 \
  --out-dir figures/qual_cvc_in \
  --prefix lscrc_cvc_id_states_v2 \
  --dpi 200
```

---

### 1.5 Script hỗ trợ (`tools/`)

| Script | Vai trò |
|--------|---------|
| `run_paper_table_sweep.sh` | Vòng lặp α ∈ {0.01, 0.05, 0.10, 0.15} → ghi 4 CSV vào `figures/sweep/` |
| `render_paper_figures_from_sweep.sh` | Từ 4 CSV cùng `BASE`, vẽ risk–coverage + scatter (tuỳ `METHOD`, `OUT_PREFIX`) |
| `run_end_to_end_deeplab_cvc_adapt.sh` | Train Kvasir → adapt CVC → sanity → tuỳ chọn sweep |
| `plot_risk_coverage.py` | Được gọi bởi `render_*`; có thể gọi tay với từng CSV |
| `export_qualitative_figures.py` | Xuất panel qualitative (đã có mã màu trạng thái) |
| `aggregate_multiseed_csv.py` | Gộp nhiều CSV cùng cấu trúc (multi-seed) → mean/std |

---

## 2. Tổng hợp phân tích — run `20260409_144715`, α = 0.05 (bảng chính)

Nguồn: `figures/sweep/20260409_144715__ckpt-cvc_adapted__alpha0p05__grid1000__scen4.csv` (và bản copy trong `figures/paper/` nếu có).

### 2.1 Chất lượng phân đoạn (backbone)

- **Kvasir in-domain:** Dice ≈ **0.844**
- **CVC in-domain (và cross Kvasir→CVC test):** Dice ≈ **0.893**

→ Backbone + pipeline CVC-adapted cho số Dice **đủ tốt** để báo cáo; các method selective **không đổi Dice** (cùng pred nhị phân), chỉ thay acceptance.

### 2.2 CVC-ClinicDB-ID (cal & test trên CVC), α = 0.05

| Method | Coverage | Expected Risk |
|--------|----------|----------------|
| Plain | 1.0 | 0.133 |
| Entropy | 0.824 | 0.0044 |
| Max-Softmax | 0.685 | 0.00077 |
| Standard CRC | 0.0 | 0.0 |
| Spatial-w. CP | 0.743 | 0.00070 |
| **LS-CRC** | **0.761** | **0.00086** |

**Nhận xét:**

- **Entropy** giữ coverage cao nhưng risk thấp hơn Plain rất nhiều; đây là baseline uncertainty mạnh trên CVC-ID với α này.
- **Max-Softmax** và **Spatial-weighted CP** có coverage thấp hơn Entropy, risk rất thấp (gần sàn lưới τ).
- **LS-CRC** đạt risk cực thấp, coverage **cao hơn Max-Softmax** (~0.76 vs ~0.68), tức trade-off khác entropy (entropy vẫn coverage cao nhất trong bảng này). Câu chuyện paper có thể nhấn **định hướng abstain theo không gian** và metric đuôi (Worst 10%, CVaR — xem trực tiếp trong CSV).
- **Standard CRC (global image gate)** tại α=0.05 vẫn **0 coverage** trong log này → rule global quá bảo thủ với calibration set / grid; nên trình bày như baseline “spatially blunt” hoặc điều chỉnh grid/score global nếu muốn baseline công bằng hơn (ngoài phạm vi báo cáo này).

### 2.3 Kvasir-SEG-ID, α = 0.05

- **LS-CRC** coverage ~**0.92**, risk ~**0.059** (gần mức budget đã hiệu chỉnh trên cal với finite-sample correction).
- **Entropy / Max-Softmax** coverage ~0.89, risk ~0.057 — tương đương LS-CRC về risk, LS-CRC **coverage cao hơn một chút** trên Kvasir-ID trong file này.

### 2.4 Cross-domain

- **Cross-Kvasir-cal-CVC-test:** LS-CRC coverage ~**0.93**, risk ~**0.032**; entropy/max ~0.91, risk ~0.028–0.029. So sánh chi tiết nên kèm **Worst Group / CVaR** từ CSV.
- **Cross-CVC-cal-Kvasir-test:** LS-CRC coverage ~**0.71**, risk ~**0.023**; spatial-weighted CP coverage ~0.72, risk ~0.018 — cần diễn giải cẩn thận theo mục tiêu paper (risk vs coverage vs tail).

### 2.5 α khác (0.01, 0.10, 0.15)

Dùng bốn file `alpha0p01` … `alpha0p15` cùng prefix `20260409_144715` để:

- kiểm tra **đường cong** coverage–risk (đã render vào `figures/paper/` nếu đã chạy `render_paper_figures_from_sweep.sh`);
- giải thích **bão hòa lưới τ** nếu hai α cho cùng điểm (hiện tượng đã được nhắc trong tài liệu nội bộ trước đó).

---

## 3. Thay đổi mã nguồn liên quan lần pipeline này (để truy vết)

- **`evaluate.py` / `calibrate.py`:** thêm baseline **Standard CRC (Bates et al.)** và **Spatial-weighted CP**; cùng lưới τ với các method khác.
- **`models/rejector.py`:** nội suy `features` lên cùng kích thước `prob` trước `torch.cat` (sửa lỗi DeepLab feature map nhỏ hơn mask).
- **`tools/export_qualitative_figures.py`:** layout 6 panel + bản đồ trạng thái màu (TP/FP/FN/abstain).
- **Script:** `run_paper_table_sweep.sh`, `render_paper_figures_from_sweep.sh` (có `OUT_PREFIX`), `run_end_to_end_deeplab_cvc_adapt.sh`.

---

## 4. Checklist “đủ cho paper”

- [x] Sweep 4 α, 4 scenario → CSV trong `figures/sweep/` (prefix `20260409_144715` + `grid1000`).
- [x] Bảng α=0.05 → copy hoặc tham chiếu `figures/paper/table_main_alpha0p05_all_methods_scen4.csv`.
- [ ] Figure risk–coverage → chạy `render_paper_figures_from_sweep.sh` nếu chưa có PNG trên máy hiện tại.
- [x] Qualitative CVC → `figures/qual_cvc_in/lscrc_cvc_id_states_v2_*.png` (khuyến nghị).
- [ ] Chèn số vào `docs/paper_assets/*.tex` / Overleaf theo CSV mới nhất.

---

## 5. Lệnh tham chiếu nhanh (một dòng)

```bash
# Sweep (đã chạy)
BASE="$(date +%Y%m%d_%H%M%S)__ckpt-cvc_adapted" CKPT_DIR=checkpoints_cvc_adapted ./tools/run_paper_table_sweep.sh

# Figure từ sweep đã có
BASE=20260409_144715__ckpt-cvc_adapted ./tools/render_paper_figures_from_sweep.sh

# Qualitative (states v2)
python tools/export_qualitative_figures.py --checkpoint-dir checkpoints_cvc_adapted \
  --backbone deeplabv3plus --encoder-weights imagenet \
  --cal-root data/CVC-ClinicDB --test-root data/CVC-ClinicDB \
  --alpha 0.05 --calibration-num-thresholds 1000 --num-images 6 \
  --out-dir figures/qual_cvc_in --prefix lscrc_cvc_id_states_v2 --dpi 200
```

---

*Báo cáo này mô tả trạng thái repo và phân tích định tính/định lượng dựa trên file sweep đã có; nếu bạn đổi checkpoint hoặc seed train, hãy cập nhật `BASE`, đường dẫn CSV và đoạn phân tích mục 2 cho đúng run mới.*
