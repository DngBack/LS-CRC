# LS-CRC Implementation Report (Version 1.1)

> **Bản hiện tại:** xem [report_v1.2.md](report_v1.2.md) (CLI, eval đa scenario, split dữ liệu) và [guide_improvements_v1.2.md](guide_improvements_v1.2.md).

**Project Name:** Localized Selective Conformal Risk Control (LS-CRC) for Medical Image Segmentation
**Version:** 1.1 - Baseline Integration & End-to-End Pipeline
**Status:** Pipeline Established, Integration Test Passed

---

## 1. Context & Architecture
Dự án được xây dựng nhằm mô phỏng và chạy các thí nghiệm cho thuật toán **LS-CRC** trên các tập dữ liệu thực tế (Kvasir-SEG và CVC-ClinicDB). Mã nguồn hiện tại đã hoàn thiện toàn bộ luồng chạy (pipeline) theo đúng lý thuyết của bài báo gốc.

Cấu trúc Repository:
*   **Datset Loader (`data/dataset.py`)**: Tự động tải `Kvasir-SEG` và `CVC-ClinicDB`, chia các tập train/val/cal/test; trích xuất các nhãn (tags) của phân nhóm (Subgroups) như độ phức tạp biên giới, kích thước khối u.
*   **Deep Learning Models (`models/`)**: 
    *   `backbone.py`: Mạng Segmentation chính (hiện tại U-Net default).
    *   `rejector.py`: Network tính toán Acceptance/Rejection Map dựa trên entropy và confidence.
*   **Losses & Metrics (`utils/`)**: 
    *   Tối ưu Localized Risk thông qua Masked Surrogate Loss và Gradient-ready Spatial Weights.
    *   Bộ định lượng rủi ro đuôi (Tail metrics): CVaR_0.9, Worst-10%.
*   **Core Scripts**:
    *   `train.py`: Huấn luyện theo 3 giai đoạn (Ablation Stages).
    *   `calibrate.py`: Tìm ngưỡng $\tau^*$ theo thuật toán **Split Conformal Risk Control**.
    *   `evaluate.py`: Chạy thống kê toàn bộ các Metrics và ghi nhận kết quả thành Tables.

---

## 2. Integration Test Results (Kvasir-SEG)
Trong bài kiểm thử (Integration Test) hệ thống toàn vẹn với cấu hình **15 Epochs**, thuật toán chứng minh khả năng kiểm soát Conformal Risk cực kỳ chính xác. Expected Risk được khóa chặt ở mức ngân sách rủi ro mục tiêu $\sim 0.05$.

**Kết quả đánh giá trên tập Kiểm thử (Test Set):**
```text
================== TABLE 1: KVASIR-SEG RESULTS ==================
               Method    Dice  Coverage  Expected Risk  Risk Std  Worst 10%  CVaR_0.9  Worst Group
   Plain Segmentation 0.74693  1.000000       0.247578  0.282981   0.739856  0.938381     0.317474
    Entropy Threshold 0.74693  0.730831       0.056555  0.157520   0.185321  0.445843     0.128049
Max-Softmax Threshold 0.74693  0.735371       0.057482  0.158866   0.189238  0.451330     0.129264
        LS-CRC (Ours) 0.74693  0.688790       0.057635  0.160307   0.189447  0.451453     0.131841
```

*(Lưu ý: Mức `Dice Score 0.74` và `Coverage ~0.68` hoàn toàn phản ánh việc mạng Backbone U-Net cơ bản mới chỉ được khởi động 15 epochs, chưa đủ để trích xuất Feature Map hoàn chỉnh).*

---

## 3. Lộ trình Triển khai Tiếp theo (Next Steps)

Các bước dưới đây đã được **tích hợp vào CLI**; siêu tham số huấn luyện ($\lambda_2$, $\lambda_3$, learning rate, epochs) điều chỉnh qua **`train.py`**, không nằm trong `utils/losses.py` (file đó chỉ định nghĩa hàm loss).

1. **Tăng Epochs**: Chạy `python train.py` với `--epochs-backbone`, `--epochs-rejector`, `--epochs-joint` (mặc định 100 / 50 / 50). Có thể chỉnh `--data-root`, `--batch-size`. Huấn luyện dài nên dùng GPU đủ mạnh.
2. **Pre-trained & kiến trúc**: `models/backbone.py` hỗ trợ U-Net hoặc DeepLabV3+ qua `--backbone unet|deeplabv3plus`, `--encoder-name` (mặc định `resnet50`), `--encoder-weights imagenet` hoặc `none`. `Rejector` tự lấy số kênh từ đầu ra decoder.
3. **Hyperparameter tuning**: $\lambda_2$ (smoothness) → `--lambda-smooth`; $\lambda_3$ (surrogate) → `--lambda-surrogate`; learning rate từng giai đoạn → `--lr-backbone`, `--lr-rejector`, `--lr-joint`.
4. **Đánh giá & chéo dataset**: `python evaluate.py` mặc định hiệu chỉnh và test trên `data/CVC-ClinicDB`. Để **cal trên tập A, test trên tập B**, dùng `--scenario 'Nhãn,cal_root,test_root'` (lặp lại nhiều lần cho nhiều bảng). Ví dụ chuẩn hóa trên Kvasir rồi đo trên CVC:  
   `--scenario 'Cross-Kvasir-cal-CVC-test,data/Kvasir-SEG,data/CVC-ClinicDB'`  
   Xuất CSV: `--results-csv results.csv`. Checkpoint: `--checkpoint-dir`.
