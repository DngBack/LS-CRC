# Experiments & Results

Tài liệu này tổng hợp và phân tích kết quả từ một lần chạy sweep ngân sách rủi ro **α** (checkpoint **`checkpoints_cvc_adapted`**, timestamp file sweep **`20260322_124208__ckpt-cvc_adapted__*`**). Có thể đưa trực tiếp hoặc rút gọn vào paper (mục *Experimental Setup* và *Results*).

---

## 1. Experimental setup (Experiments)

### 1.1 Mục tiêu

Đánh giá **Localized Selective Conformal Risk Control (LS-CRC)** so với phân đoạn thường (**Plain**) và các ngưỡng heuristic (**Entropy**, **Max-softmax**) trên:

- hai tập **in-domain** (cal/test cùng dataset);
- hai tập **cross-calibration** (cal trên dataset A, test trên dataset B).

Đồng thời đo **nhạy cảm** của các phương pháp có hiệu chỉnh theo **α** (ngân sách rủi ro mục tiêu trong `calibrate.py`).

### 1.2 Dữ liệu và chia tập

- **Kvasir-SEG**: train/val/cal/test theo `download_data.py` / `*.txt` trên đĩa.
- **CVC-ClinicDB**: tương tự; split đã chỉnh để khớp số ảnh thực tế (regenerate splits nếu cần).

### 1.3 Scenario đánh giá (4 bảng mỗi lần chạy `evaluate.py`)

| Nhãn (`Dataset` trong CSV) | Cal | Test | Ý nghĩa |
|---------------------------|-----|------|---------|
| `Kvasir-SEG-ID` | Kvasir | Kvasir | In-domain Kvasir |
| `CVC-ClinicDB-ID` | CVC | CVC | In-domain CVC |
| `Cross-Kvasir-cal-CVC-test` | Kvasir | CVC | Cal nguồn Kvasir, đo OOD trên CVC |
| `Cross-CVC-cal-Kvasir-test` | CVC | Kvasir | Cal nguồn CVC, đo OOD trên Kvasir |

### 1.4 Mô hình và huấn luyện

- **Backbone**: U-Net, encoder **ResNet-50**, trọng số encoder **ImageNet** (`segmentation_models_pytorch`).
- **Pipeline**: ba giai đoạn (backbone → rejector → joint), có validation và checkpoint tốt nhất theo val (theo `train.py` v1.2).
- **Thích ứng miền**: checkpoint **`checkpoints_cvc_adapted`** — fine-tune rejector/joint trên **CVC** với backbone Kvasir đã huấn luyện (xem `guide_improvements_v1.2.md`).

### 1.5 Hiệu chỉnh và metric

- **α**: lần lượt **0.01, 0.05, 0.10, 0.15** trong sweep; mỗi α một file CSV riêng.
- **Calibrate**: `calibrate.py` — lưới τ ∈ [0.01, 0.99] (100 điểm), mục tiêu risk trên tập cal ≤ α − 1/n.
- **Dice**: từ mặt nạ nhị phân `p ≥ 0.5` — **cố định** theo tập test, không đổi giữa các phương pháp selective (cùng xác suất backbone).
- **Coverage**: tỷ lệ pixel được chấp nhận theo từng phương pháp.
- **Expected Risk**: rủi ro có trọng số không gian (localized) trên vùng được chấp nhận, trung bình trên ảnh test.

### 1.6 Figure phụ trợ

- Đồ thị **Coverage vs α** và **Expected Risk vs α** (và scatter risk–coverage): `tools/plot_risk_coverage.py` trên các CSV sweep, lọc `Dataset = CVC-ClinicDB-ID`, `Method = LS-CRC (Ours)` (và có thể lặp cho dataset khác).

---

## 2. Results — tổng hợp số liệu (LS-CRC và baseline)

Bảng dưới chỉ trích **Expected Risk** và **Coverage** cho **LS-CRC (Ours)**; Plain luôn Coverage = 1 và Risk = risk phân đoạn đầy đủ trên tập đó (Kvasir ~0.118, CVC ~0.110).

### 2.1 In-domain CVC (`CVC-ClinicDB-ID`)

| α | τ* (LS-CRC) | Coverage | Expected Risk | Ghi chú |
|---|-------------|----------|----------------|---------|
| 0.01 | 0.999 | 0.034 | ~0 | Gần như không chấp nhận → risk trung bình suy biến (~0) |
| 0.05 | 0.525 | 0.878 | 0.0127 | Hành vi selective **ổn định**, risk thấp hơn α |
| 0.10 | 0.525 | 0.878 | 0.0127 | Cùng τ* với α=0.05 (lưới τ rời rạc → **bão hòa**) |
| 0.15 | 0.010 | 1.000 | 0.1097 | τ rơi xuống sàn lưới → **chấp nhận hết** → giống Plain |

**Entropy / Max-softmax (CVC-ID, α=0.05):** Coverage cao (~0.92–0.93), Expected Risk ~0.025–0.045 — risk cao hơn LS-CRC; tại α=0.01 nhiều baseline cũng suy biến (coverage rất thấp, risk ~0).

### 2.2 In-domain Kvasir (`Kvasir-SEG-ID`)

| α | τ* (LS-CRC) | Coverage | Expected Risk |
|---|-------------|----------|----------------|
| 0.01 | 0.980 | 0.426 | 0.0048 |
| 0.05 | 0.525 | 0.805 | 0.0284 |
| 0.10 | 0.010 | 1.000 | 0.1179 |
| 0.15 | 0.010 | 1.000 | 0.1179 |

Tại **α ≥ 0.10**, LS-CRC với lưới hiện tại chọn **τ = 0.01** → coverage 1 → **không còn selective** (risk trùng Plain).

### 2.3 Cross: cal Kvasir → test CVC (`Cross-Kvasir-cal-CVC-test`)

| α | LS-CRC Coverage | LS-CRC Risk | So sánh nhanh |
|---|-----------------|-------------|----------------|
| 0.01 | 0.606 | ~0 | Baseline entropy/max trên CVC test cũng risk ~0 với coverage thấp/trung |
| 0.05 | 0.878 | 0.0127 | **Giống hệt** hàng CVC-ID cùng α (cùng τ*=0.525) |
| 0.10 | 1.000 | 0.1097 | Mất selective (τ=0.01) |
| 0.15 | 1.000 | 0.1097 | Mất selective |

Ở **α = 0.05**, entropy trên cùng scenario có Expected Risk **~0.068** và Coverage **~0.97** — **vượt** ngân sách 0.05 trên test; LS-CRC giữ risk **~0.013** với coverage **~0.88**.

### 2.4 Cross: cal CVC → test Kvasir (`Cross-CVC-cal-Kvasir-test`)

| α | LS-CRC Coverage | LS-CRC Risk |
|---|-----------------|-------------|
| 0.01 | 0.020 | ~0 |
| 0.05 | 0.805 | 0.0284 |
| 0.10 | 0.805 | 0.0284 |
| 0.15 | 1.000 | 0.1179 |

Tại α=0.05 và 0.10, LS-CRC giữ **cùng** τ*=0.525 và **cùng** coverage/risk (lại hiện tượng bão hòa lưới τ giữa hai mức α).

---

## 3. Phân tích (Discussion gắn với Results)

### 3.1 α quá nhỏ (0.01)

- Ngân sách **α − 1/n** trên tập cal rất chặt → thuật toán đẩy **τ* rất cao** (0.98–0.999) → coverage cực thấp trên CVC/Kvasir, risk trung bình gần 0 (ít vùng accepted để tính risk có ý nghĩa).
- **Ý nghĩa paper:** cần nói rõ đây là **vùng vận hành cực đoan**; không nên chỉ báo cáo α=0.01 làm điểm duy nhất cho trade-off coverage–risk.

### 3.2 α = 0.05 (điểm khuyến nghị cho bảng chính)

- **CVC-ID:** LS-CRC đạt coverage ~**0.88**, risk ~**0.013** — dưới α, entropy/max-softmax risk cao hơn hoặc coverage khác.
- **Cross (cal Kvasir, test CVC):** LS-CRC vẫn kiểm soát risk tốt; entropy **không** khóa được risk ≤ 0.05 trên test ở log này.
- Phù hợp làm **α mặc định** trong bảng chính và figure risk–coverage (kèm α=0.10 nếu muốn thấy bão hòa).

### 3.3 α lớn (0.10–0.15) và sàn lưới τ

- Với **α ≥ 0.10**, nhiều scenario chọn **τ* = 0.01** (điểm đầu lưới) → **chấp nhận gần như toàn bộ pixel** → LS-CRC về mặt metric aggregate **trùng Plain** (coverage 1, risk = risk phân đoạn đầy đủ).
- **Nguyên nhân kỹ thuật:** `calibrate_threshold` chỉ thử 100 giá trị τ cách đều; với α lớn, mọi τ thấp đều thỏa risk trên cal → chọn τ **nhỏ nhất trong lưới** để tối đa coverage → 0.01.
- **Cải tiến có thể đề cập trong paper (future work):** lưới mịn hơn, τ tối thiểu > 0, hoặc ràng buộc coverage tối đa / điều chỉnh monotonicity.

### 3.4 Bão hòa giữa α = 0.05 và α = 0.10 (CVC-ID, LS-CRC)

- Cùng **τ* = 0.525** và **cùng** coverage/risk — do với lưới cố định, mức risk tại τ=0.525 đã **đủ thấp** so với cả target 0.05 và 0.10; τ nhỏ hơn trong lưới có thể làm risk cal vượt target hoặc không được chọn theo logic hiện tại.
- Figure **Coverage vs α** sẽ thấy **đoạn ngang** — không phải lỗi, mà phản ánh **discretization** của hiệu chỉnh.

### 3.5 Dice không đổi giữa các method

- Đúng thiết kế hiện tại: mọi phương pháp dùng chung **prob** từ backbone; selective chỉ thay **A_u** cho risk/coverage.
- Trong paper nên nhấn mạnh: so sánh chính là **selective risk / coverage / tail metrics**, không phải cải thiện Dice trừ khi có bước chỉ sửa dự đoán trên vùng accepted.

### 3.6 Vai trò CVC-adapted checkpoint

- So với báo cáo trước adapt (CVC-ID LS-CRC coverage ~0), checkpoint **cvc_adapted** cho phép LS-CRC **hoạt động có nghĩa** trên CVC tại α=0.05 (τ ~0.5 thay vì ~0.99).
- Phần Experiments nên nêu rõ: **backbone** chủ yếu từ Kvasir, **rejector** tinh chỉnh thêm trên CVC.

---

## 4. Gợi ý cách trình bày trong paper

1. **Experiments:** tóm tắt mục 1 (dataset, scenarios, backbone, calibrate α, metrics) trong 1–2 đoạn + một bảng scenario.
2. **Results:**  
   - Bảng chính: **α = 0.05**, cả 4 scenario, 4 methods (Plain, Entropy, Max-softmax, LS-CRC): Dice, Coverage, Expected Risk, (CVaR / Worst-group nếu journal cho phép).  
   - Bảng hoặc figure phụ: sweep α (hoặc chỉ 0.01 / 0.05 / 0.15) cho **LS-CRC** trên CVC-ID và một cross scenario — giải thích sàn τ và suy biến α=0.01.  
3. **Figure:** dùng `risk_coverage_vs_alpha__cvc__lscrc.png` và `_scatter.png`; thêm 1–2 ảnh qualitative (`export_qualitative_figures.py`).

**Bảng LaTeX + Markdown + snippet figure sẵn để chèn:** thư mục [paper_assets/](paper_assets/) — xem [paper_assets/PAPER_INSERT.md](paper_assets/PAPER_INSERT.md).

---

## 5. Tệp sinh ra (tham chiếu run log)

- CSV: `figures/sweep/20260322_124208__ckpt-cvc_adapted__alpha0p01__scen4.csv` … `alpha0p15__scen4.csv`
- Figure: `figures/paper/risk_coverage_vs_alpha__cvc__lscrc.png`, `..._scatter.png`

Khi chạy lại sweep với `STAMP` mới, cập nhật tên file trong mục 5 cho khớp bản reproducibility.
