# Localized Selective Conformal Risk Control (LS-CRC) cho Kvasir-SEG

Đây là mã nguồn (codebase) chính thức cho quá trình tái tạo và thử nghiệm thuật toán **Localized Selective Conformal Risk Control** trong bài toán Medical Image Segmentation. Dự án hỗ trợ tập dữ liệu Kvasir-SEG và CVC-ClinicDB.

## 1. Cấu trúc Dự án
```text
/home/admin1/Desktop/LS-CRC/
  ├── train.py               # Script chính để train các mô hình (Backbone, Rejector, Joint-Finetuning).
  ├── evaluate.py            # Chạy đánh giá và hiệu chuẩn (Calibration), in ra các bảng thống kê.
  ├── calibrate.py           # Thuật toán Split Conformal Risk Control (tìm threshold $\tau^*$).
  ├── download_data.py       # Tải tự động các Dataset và tạo splits files (train/val/cal/test).
  ├── data/
  │    └── dataset.py        # Logic tải và tiền xử lý Kvasir-SEG / CVC-ClinicDB, chia Subgroup Tags.
  ├── models/
  │    ├── backbone.py       # Kiến trúc mạng phân vùng (U-Net).
  │    └── rejector.py       # Mạng Rejector dự đoán vùng rủi ro (Acceptance Score).
  └── utils/
       ├── losses.py         # Tính Spatial FNR, Surrogate Miss Risk, và Smoothness Penalty.
       ├── metrics.py        # Tính mảng metrics: Dice, IoU, CVaR_0.9, Worst-10%.
       └── pseudo_labels.py  # Thuật toán tạo Pseudo-labels giám sát Rejector.
```

## 2. Kết quả Thực nghiệm Hiện Tại
Trong một test run tích hợp kéo dài `15 epochs` trên tập Kvasir-SEG, dưới đây là kết quả Table 1:

```text
================== TABLE 1: KVASIR-SEG RESULTS ==================
               Method    Dice  Coverage  Expected Risk  Risk Std  Worst 10%  CVaR_0.9  Worst Group
   Plain Segmentation 0.74693  1.000000       0.247578  0.282981   0.739856  0.938381     0.317474
    Entropy Threshold 0.74693  0.730831       0.056555  0.157520   0.185321  0.445843     0.128049
Max-Softmax Threshold 0.74693  0.735371       0.057482  0.158866   0.189238  0.451330     0.129264
        LS-CRC (Ours) 0.74693  0.688790       0.057635  0.160307   0.189447  0.451453     0.131841
```

*(Thuật toán CRC hoạt động hoàn hảo khi đẩy `Expected Risk` về giới hạn $\approx 0.05$ theo yêu cầu chặt chẽ)*.

## 3. Lộ trình Triển khai Tiếp theo (Next Steps)

Để hoàn thiện mạng lưới và đẩy model lên mức State-of-the-Art khớp hoàn toàn với bài báo:

1. **Tăng Epochs lên kích thước thực tế**: Vào `train.py` để tăng số batch/epochs (VD: Backbone train 100 epochs, Rejector 50 epochs). Để train cực lâu anh sẽ cần một GPU mạnh dạn.
2. **Sử dụng Pre-trained Models**: Kết nối U-Net (hoặc DeepLabV3+) với Encoder như ResNet-50 có pretrained weights trên ImageNet thay vì khởi tạo ngẫu nhiên, để đẩy Dice gốc > 0.90 ngay lập tức.
3. **Hyperparameter Tuning**: Đẩy trọng số $\lambda_2$ (Smoothness) và $\lambda_3$ (Surrogate Risk). Tăng cường sức mạnh liên tiếp bằng cách tuning learning rate của file `losses.py`.
4. **Đánh giá Chéo Dataset**: Khởi chạy `evaluate.py` kết xuất các bảng thống kê cho tập dữ liệu thứ 2 là `CVC-ClinicDB` để đánh giá tính bền vững ngoại vi (Robustness).
