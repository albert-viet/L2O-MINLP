# step_5 — Geometry của mixed equality

Kế hoạch tổng: [../plan.md](../plan.md). Dùng lại log `x̄`, `x̂` từ step_3.

## Yêu cầu
Phân tích cả lý thuyết và thực nghiệm giả thuyết: mixed equality `h(x_r,z) = aᵀx_r − c·bᵀz = 0` tạo họ feasible slices rời rạc `F_h = ∪_z H_z` theo integer assignment; đo tác động trực tiếp của integer rounding lên equality residual.

## Mục tiêu cần đạt
Có bằng chứng định lượng cho hoặc chống lại giả thuyết quantization-induced equality violation.

## To-do
- [ ] **Viết derivation** khoảng dịch của equality hyperplane khi `z` đổi một đơn vị.
  *Mục tiêu:* công thức tường minh cho độ dịch của `h` theo hệ số `c` và `b`, áp dụng cho MIRB đang dùng.
- [ ] **Tính `Δh = h(x̂) − h(x̄)`** cho mọi instance từ log step_3.
  *Mục tiêu:* có `Δh` từng instance, kiểm tra một vài instance bằng tay khớp derivation.
- [ ] **Vẽ phân phối `Δh`.**
  *Mục tiêu:* hình trong `figures/`.
- [ ] **So sánh `|h(x̄)|` và `|h(x̂)|`.**
  *Mục tiêu:* tỷ lệ instance mà rounding làm `|h|` tăng.
- [ ] **Đánh dấu các sample có integer flip.**
  *Mục tiêu:* mỗi sample có cờ flip/không flip.
- [ ] **Đo tương quan giữa integer flip và bước nhảy của equality violation.**
  *Mục tiêu:* hệ số tương quan (hoặc so sánh nhóm flip và không flip) kèm kết luận ủng hộ hay bác bỏ giả thuyết.

## Điều kiện hoàn thành
Tick hết các to-do trên.
