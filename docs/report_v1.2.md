# LS-CRC — Báo cáo phiên bản 1.2

**Phiên bản:** 1.2 — CLI, đánh giá đa scenario, split dữ liệu an toàn, tài liệu thí nghiệm  
**Trạng thái:** Sẵn sàng cho PR / merge; checkpoint huấn luyện **không** đưa vào git (xem `.gitignore`).

---

## Tóm tắt thay đổi so với 1.1

| Hạng mục | Mô tả |
|----------|--------|
| **Huấn luyện** | `train.py`: argparse (epochs, LR, λ, backbone, encoder), validation + checkpoint tốt nhất theo val, `--epochs-* 0` để bỏ qua stage, **`--resume-backbone` / `--resume-rejector`** để fine-tune miền khác. |
| **Mô hình** | `models/backbone.py`: U-Net / DeepLabV3+, encoder tùy chọn, `decoder(features)` tương thích SMP 0.5+, `feature_channels` suy ra tự động. |
| **Đánh giá** | `evaluate.py`: nhiều `--scenario`, `--results-csv`, `--checkpoint-dir`, đồng bộ kiến trúc với train. |
| **Dữ liệu** | `download_data.py`: `generate_splits` khớp `len(dataset)`, giữ tối thiểu ~10% cho test, **`--regenerate-splits`**. `dataset.py`: bỏ dòng split trỏ file thiếu. |
| **Tài liệu** | `report_v1.1.md` cập nhật Next Steps theo CLI. `report_experiments_eval.md`: phân tích kết quả thực nghiệm. **`guide_improvements_v1.2.md`**: hướng dẫn chạy các hướng cải tiến. |

---

## Tài liệu liên quan

- [report_v1.1.md](report_v1.1.md) — báo cáo nền & kiến trúc (v1.1).  
- [report_experiments_eval.md](report_experiments_eval.md) — đánh giá số liệu & định hướng phân tích.  
- [guide_improvements_v1.2.md](guide_improvements_v1.2.md) — **lệnh cụ thể** cho cải tiến tiếp theo.

---

## Pull Request (gợi ý)

**Title:** `release: LS-CRC v1.2 — training CLI, multi-scenario eval, data split fixes`

**Body (rút gọn):**

- Thêm CLI đầy đủ cho `train.py` / `evaluate.py`, resume checkpoint cho fine-tune đa miền.  
- Sửa tạo split HF theo đúng số mẫu + regenerate từ đĩa; dataset bỏ qua file thiếu.  
- Báo cáo thí nghiệm + hướng dẫn cải tiến v1.2.  
- `checkpoints/*.pth` và `experiments_full.csv` chuyển sang `.gitignore`.
