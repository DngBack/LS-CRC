# Tóm tắt paper: LS-CRC

Nguồn: `main.md` — *Learning Structured Abstention for Localized Conformal Risk Control in Segmentation* (NeurIPS 2026 submission).

---

## Paper là gì?

**Tiêu đề:** *Learning Structured Abstention for Localized Conformal Risk Control in Segmentation*.

**Vấn đề:** Conformal Risk Control (CRC) chuẩn chỉ đảm bảo **rủi ro kỳ vọng toàn cục** (marginal), không xử lý được việc lỗi tập trung ở **biên mỏng, cấu trúc nhỏ, ảnh khó** — điều quan trọng trong phân đoạn y tế (ví dụ polyp).

**Ý tưởng chính — LS-CRC:** Kết hợp **rejector học theo pixel** (bản đồ chấp nhận có cấu trúc không gian) với **calibration conformal tách tập (split CRC)**. Rủi ro được định nghĩa là **localized selective miss risk**: false negative có trọng số (đặc biệt nhấn mạnh **vùng biên**), chỉ tính trên pixel foreground **được chấp nhận**; mẫu số cố định để giữ tính đơn điệu cho calibration.

**Đóng góp lý thuyết:**

1. **Theorem 1 (Marginal):** Giữ đảm bảo marginal CRC — rủi ro kỳ vọng ≤ α + O(1/n).
2. **Theorem 2 (Subgroup):** Bound độ lệch rủi ro theo subgroup.
3. **Theorem 3 (Tail):** Cải thiện CVaR / tail risk khi rejector gần “chính sách tối ưu”.

---

## Thực nghiệm (tóm tắt)

| Hạng mục | Chi tiết |
|----------|----------|
| **Dữ liệu** | Kvasir-SEG; CVC-ClinicDB (in-domain sau adapt + cross-domain) |
| **Mô hình** | DeepLabV3+ (ResNet-50), rejector nông trên feature + xác suất + entropy + margin |
| **Baseline** | Plain segmentation; entropy / max-softmax threshold; Standard CRC (cổng cấp ảnh); spatial-weighted conformal prediction |
| **Điểm chính** | α = 0.05; thêm đánh giá nhiều α và chuyển miền |

---

## Kết quả chính

### In-domain, α = 0.05

| Bối cảnh | Điểm nổi bật |
|----------|----------------|
| **Kvasir-SEG** | LS-CRC **coverage cao nhất** trong các phương pháp selective (**0.920** vs ~0.90 baseline tốt nhất khác), risk ~0.059 (trong ngân sách). Entropy / max-softmax **tốt hơn** về Worst 10% và CVaR — paper giải thích rejector chưa “adapt” sâu cho domain này. |
| **CVC-ClinicDB (adapted)** | LS-CRC **coverage cao nhất** trong các phương pháp có cấu trúc không gian (**0.761** vs 0.743 Spatial CP, 0.685 Max-Softmax), đồng thời **worst-10% image risk thấp nhất** (**&lt;0.001**). |

**Standard CRC** tại α = 0.05: **coverage = 0** (cổng cấp ảnh không phù hợp ngân sách risk theo pixel chặt).

### Cross-domain (α = 0.05)

- **Kvasir → CVC:** LS-CRC **coverage cao nhất** (0.931) và **worst-10% tốt nhất** (0.031 vs ~0.041 các baseline khác).
- **CVC → Kvasir:** Max-Softmax có risk thấp nhất nhưng coverage rất thấp; LS-CRC nằm giữa (coverage ~0.706, risk 0.023).

### Nhiều mức α trên CVC (in-domain)

- **α = 0.01:** LS-CRC coverage **~2.7×** so với entropy (0.359 vs 0.132), risk ~0.
- **α = 0.10:** LS-CRC tốt hơn entropy về coverage, risk, worst-10%, worst-group.
- Một số điểm **CVaR** entropy có thể hơn (paper gán cho sự mượt của mẫu abstention dựa entropy).

### Lưu ý trong bản thảo

Bảng multi-seed và ablation được ghi là **projected**; bản camera-ready dự kiến thay bằng chạy nhiều seed thật.

---

## Kết luận một dòng

LS-CRC dùng **abstention có cấu trúc không gian** + **CRC** để chi tiêu cùng ngân sách rủi ro hiệu quả hơn ngưỡng uncertainty vô hướng; thực nghiệm polyp cho thấy **frontier risk–coverage tốt**, đặc biệt trên **domain đã adapt (CVC)** và **chuyển Kvasir→CVC**, trong khi trên Kvasir “thuần” các chỉ số tail có thể chưa vượt entropy nếu rejector chưa khớp domain.
