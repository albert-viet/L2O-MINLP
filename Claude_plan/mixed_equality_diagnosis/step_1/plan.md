# step_1 — Baseline đối chứng

Kế hoạch tổng: [../plan.md](../plan.md). Cần có hạ tầng từ step_0.

## Yêu cầu
- Giữ nguyên seed, `K`, split, optimizer, số epoch, penalty và projection setting giữa các biến thể; chỉ thay đúng một yếu tố.
- Chạy 4 biến thể (xem step_0) trên mixed dataset hiện tại (`p~U(1,8)`), mỗi biến thể 3-5 seed.
- Báo cáo cùng một bộ metrics cho mọi biến thể.

## Mục tiêu cần đạt
Tách ảnh hưởng của equality, integrality và tương tác equality–integrality; xác định biến thể nào bắt đầu xuất hiện degradation đáng kể.

## To-do
- [x] **Chạy biến thể 1 (không equality).**
  *Mục tiêu:* có kết quả đủ các seed, lưu CSV trong `data/`.
- [x] **Chạy biến thể 2 (equality chỉ continuous).**
  *Mục tiêu:* như trên, cùng cấu hình với biến thể 1.
- [x] **Chạy biến thể 3 (mixed equality, integer relaxed).**
  *Mục tiêu:* như trên.
- [x] **Chạy biến thể 4 (mixed equality + RC).**
  *Mục tiêu:* như trên; kết quả khớp xu hướng của `eq_results/infea_feas_case` ở mức trung bình.
- [x] **Lưu cùng format:** objective, inequality violation, equality violation, feasibility rate, `F(ε)`.
  *Mục tiêu:* một file tổng hợp duy nhất đọc được bằng một lệnh pandas, không thiếu cột nào ở bất kỳ biến thể nào.
- [x] **Lập bảng so sánh bốn biến thể** (mean ± std) và hình so sánh.
  *Mục tiêu:* bảng và hình trong `data/`, `figures/` cho thấy chênh lệch giữa các biến thể lớn hơn độ nhiễu giữa seed, hoặc nêu rõ là không.
- [x] **Kết luận biến thể nào gây degradation.**
  *Mục tiêu:* một đoạn ngắn nêu biến thể đầu tiên degradation đáng kể và bằng chứng số liệu.

## Điều kiện hoàn thành
Tick hết các to-do trên, có bảng so sánh trực tiếp bốn case.

## Kết quả
Xem [report.md](report.md). Dữ liệu trong `data/runs/`, bảng `data/comparison_table.md`, hình `figures/variant_comparison.png`.
