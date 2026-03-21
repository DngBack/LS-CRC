# Báo cáo đánh giá thí nghiệm LS-CRC (v1.2)

**Phiên bản tài liệu:** 1.2 (xem [report_v1.2.md](report_v1.2.md), hướng dẫn chạy cải tiến: [guide_improvements_v1.2.md](guide_improvements_v1.2.md)).

**Ngữ cảnh số liệu:** log `evaluate.py` đa scenario, checkpoint cục bộ, encoder ResNet-50 ImageNet — ba scenario: Kvasir in-domain, CVC in-domain, cross (cal Kvasir → test CVC).

---

## 1. Bối cảnh và cách đọc bảng

- **Dice / IoU:** chất lượng phân đoạn (mặt nạ nhị phân từ `prob ≥ 0.5`); với các phương pháp selective, Dice **không đổi** giữa Plain và các baseline vì cùng một bản đồ xác suất — khác biệt nằm ở **Coverage** (tỷ lệ pixel được “chấp nhận”) và **Expected Risk** (rủi ro có trọng số trên vùng được chấp nhận).
- **Ngân sách rủi ro:** `alpha = 0.05` trong `calibrate.py` (có hiệu chỉnh hữu hạn mẫu `α - 1/n` trên tập cal).
- **LS-CRC:** chấp nhận pixel khi điểm rejector `s_φ ≥ τ*`; `τ*` chọn trên tập **cal** để rủi ro hiện tại ≤ ngân sách, ưu tiên **coverage lớn nhất** trong lưới ngưỡng thử.

---

## 2. Tóm tắt kết quả quan sát

### 2.1 Kvasir-SEG (in-domain)

| Phương pháp | Coverage | Expected Risk | Nhận xét ngắn |
|-------------|----------|---------------|----------------|
| Plain | 1.0 | ~0.138 | Không từ chối; rủi ro cao. |
| Entropy | ~0.656 | ~0.038 | Gần ngân sách; coverage hợp lý. |
| Max-softmax | ~0.938 | ~0.084 | Rất “dễ chấp nhận”; rủi ro vượt mục tiêu 0.05. |
| **LS-CRC** | ~0.424 | ~0.033 | Rủi ro dưới 0.05; coverage trung bình — hành vi **nhất quán** với mục tiêu selective risk. |

**Đánh giá:** Trên miền huấn luyện (Kvasir), pipeline hiện tại cho thấy LS-CRC kiểm soát rủi ro tốt hơn Plain và tương thích với ngân sách; so với entropy, LS-CRC thay đổi trade-off coverage–risk theo hướng phụ thuộc mạng rejector đã học.

### 2.2 CVC-ClinicDB (in-domain)

| Phương pháp | Coverage | Expected Risk (LS-CRC) | Nhận xét |
|-------------|----------|-------------------------|----------|
| Plain / Entropy / Max-softmax | Giống pattern Kvasir (Dice ~0.79) | — | Baseline ổn định. |
| **LS-CRC** | **~0.003** | **0** (suy biến) | **Bất thường:** gần như không pixel nào được chấp nhận. |

**Đánh giá:** Dòng LS-CRC trên CVC-ID **không phản ánh** một selective predictor hữu ích: coverage ~0 làm các chỉ số rủi ro trung bình mất ý nghĩa (gần như không có vùng được chấp nhận để tính rủi ro có định thức). Ngưỡng hiệu chỉnh `τ* ≈ 0.99` cho thấy trên tập **cal của CVC**, chỉ mức ngưỡng cực cao mới đủ để đáp ứng `risk ≤ target` — thường gặp khi **rejector + backbone huấn luyện trên Kvasir** nhưng phân phối đặc trưng/score trên CVC lệch mạnh so với giả định lúc train.

### 2.3 Cross: cal Kvasir → test CVC

- **LS-CRC:** Coverage ~0.48, Expected Risk ~0.066 — **cao hơn** ngân sách 0.05 một mức nhỏ (sai số hiệu chỉnh / domain shift).
- So với CVC-ID, cross-scenario cho LS-CRC **có coverage và risk không suy biến**, phù hợp với kỳ vọng “đo ngoài miền” nhưng vẫn cần hiệu chỉnh chặt hơn nếu muốn khóa đúng α trên từng tập test.

**Điểm đáng chú ý:** Cùng test CVC nhưng **cal trên Kvasir** lại cho LS-CRC khả dụng hơn **cal trên CVC** — dấu hiệu của **mismatch** giữa phân phối score trên cal CVC và cách `calibrate_threshold` tìm `τ*` (ví dụ rủi ro trên cal CVC rất “khó” dưới ngưỡng α trừ khi `τ` cực lớn). Điều này nhấn mạnh nhu cầu **hiệu chỉnh / thích ứng miền** thay vì chỉ một bộ trọng số cố định.

---

## 3. Kết luận ngắn

1. **In-domain Kvasir:** Kết quả LS-CRC **hợp lý** với mục tiêu kiểm soát rủi ro có chọn lọc.
2. **In-domain CVC với rejector chưa thích ứng:** LS-CRC **suy biến** (coverage ~0) — cần xem là **lỗi thực nghiệm / pipeline** chứ không phải “CVC không dùng được LS-CRC”.
3. **Cross-domain:** LS-CRC cho trade-off có thể đọc được; risk hơi vượt α, phù hợp hướng cải tiến bên dưới.

---

## 4. Định hướng cải tiến (ưu tiên)

### 4.1 Sửa / làm rõ pipeline trên CVC (ưu tiên cao)

- **Fine-tune rejector (và có thể joint một ít epoch)** trên **CVC train** hoặc trên tổng hợp Kvasir+CVC để phân phối score gần với miền đánh giá.
- **Temperature scaling / hiệu chỉnh logits** của rejector trên tập cal của **đúng** miền test (Platt hoặc scaling một chiều trên score trung bình theo batch).
- **Tăng cỡ tập cal** hoặc **lặp lại cal** trên nhiều fold để giảm phương sai `τ*`.

### 4.2 Hiệu chỉnh thuật toán ngưỡng

- Kiểm tra **đơn điệu** risk/coverage theo `τ` trên từng dataset; nếu không đơn điệu, lưới `np.linspace(0.01, 0.99, 100)` có thể chọn nhầm — nên dùng **tìm nhị phân** hoặc lưới mịn hơn quanh vùng khả thi.
- Khi **không có** `τ` nào đạt `risk ≤ target`, `calibrate.py` trả về failsafe `0.999` — nên **log cảnh báo** và tách báo cáo metric (“calibration failed”) thay vì in một dòng số 0 dễ hiểu nhầm.

### 4.3 Đánh giá và báo cáo

- Bổ sung **điều kiện tối thiểu coverage** (ví dụ báo đỏ nếu mean coverage < 1%) để phát hiện suy biến như bảng CVC-ID.
- Ghi rõ trong báo cáo: **Dice cố định** giữa các method là **cố ý** (cùng backbone); so sánh chính là **risk–coverage**.

### 4.4 Huấn luyện backbone

- Tiếp tục **đủ epoch** và (nếu cần) **dữ liệu đa miền** để Dice trên CVC tăng — giảm entropy/score nhiễu giúp các baseline entropy/max-softmax và rejector ổn định hơn.

---

## 5. Lệnh tái lập (tham chiếu)

```bash
python evaluate.py \
  --checkpoint-dir checkpoints \
  --encoder-weights imagenet \
  --scenario 'Kvasir-SEG-ID,data/Kvasir-SEG,data/Kvasir-SEG' \
  --scenario 'CVC-ClinicDB-ID,data/CVC-ClinicDB,data/CVC-ClinicDB' \
  --scenario 'Cross-Kvasir-cal-CVC-test,data/Kvasir-SEG,data/CVC-ClinicDB' \
  --results-csv experiments_full.csv
```

Sau khi chỉnh code hoặc huấn luyện lại, nên chạy lại cùng lệnh và cập nhật bảng trong báo cáo phiên bản tiếp theo.
