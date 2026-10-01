# step_3 — Instrument pipeline theo từng stage

Kế hoạch tổng: [../plan.md](../plan.md). Dùng hook ghi log từ step_0, áp dụng cho biến thể 4 (mixed equality + RC).

## Yêu cầu
Ghi lại `h(x̄)`, `h(x̂)`, `h(x_proj^(1..T))` cùng vi phạm bất đẳng thức, số integer flip, khoảng cách latent tới ngưỡng làm tròn và `‖∇_{x̄_r} V_eq‖`, `‖∇_{x̄_z} V_eq‖`.

## Mục tiêu cần đạt
Chỉ ra stage đầu tiên làm equality violation tăng hoặc không thể giảm tiếp: trước rounding, sau rounding hay trong projection.

## To-do
- [ ] **Log `h(x̄)` tại output latent.**
  *Mục tiêu:* có giá trị cho mọi test instance và seed.
- [ ] **Log `h(x̂)` ngay sau RC correction.**
  *Mục tiêu:* như trên, cùng instance với `h(x̄)` để so sánh từng cặp.
- [ ] **Log `h(x_proj^(t))` tại từng vòng projection.**
  *Mục tiêu:* chuỗi đầy đủ `t=1..T` cho mỗi instance.
- [ ] **Log vi phạm bất đẳng thức song song** với `h` ở mọi stage.
  *Mục tiêu:* biết lúc `h` giảm thì bất đẳng thức tăng hay không.
- [ ] **Log số tọa độ integer thay đổi ở mỗi vòng projection.**
  *Mục tiêu:* xác định projection có làm đổi `z` hay không.
- [ ] **Log khoảng cách của latent integer variables tới ngưỡng làm tròn.**
  *Mục tiêu:* có phân phối khoảng cách để dùng cho step_5 và step_6.
- [ ] **Log `‖∇_{x̄_r} V_eq‖` và `‖∇_{x̄_z} V_eq‖`.**
  *Mục tiêu:* so được độ lớn gradient phần continuous và phần integer.
- [ ] **Vẽ trajectory của `h` qua toàn pipeline.**
  *Mục tiêu:* hình trong `figures/` cho thấy rõ stage nào làm `h` tăng hoặc kẹt.
- [ ] **Kết luận stage đầu tiên gây vấn đề.**
  *Mục tiêu:* một đoạn ngắn kèm số liệu.

## Điều kiện hoàn thành
Tick hết các to-do trên; dữ liệu log lưu trong `data/` để step_5 dùng lại.
