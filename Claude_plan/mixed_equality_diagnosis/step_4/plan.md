# step_4 — Conditional feasibility với `ẑ` cố định

Kế hoạch tổng: [../plan.md](../plan.md). Đây là milestone bắt buộc trước khi thay đổi thuật toán (step_final).

## Yêu cầu
- Giữ cố định `ẑ` do network dự đoán (RC), dùng exact solver (SCIP/Pyomo) cho bài toán continuous có điều kiện:
  tìm `x_r` thỏa `g(x_r, ẑ; ξ) ≤ 0` và `h(x_r, ẑ; ξ) = 0`.
- Phân loại từng instance theo kết quả conditional feasibility.

## Mục tiêu cần đạt
Tách integer-selection failure khỏi continuous-repair failure: mỗi instance được xếp vào "`z` sai" hoặc "`z` đúng/sửa được nhưng `x_r` hỏng".

## To-do
- [ ] **Trích `ẑ` từ RC** cho từng test instance.
  *Mục tiêu:* có `ẑ` cho mọi test instance và seed, lưu trong `data/`.
- [ ] **Tạo bài toán continuous có điều kiện** (fix biến integer trong Pyomo).
  *Mục tiêu:* model hợp lệ, giải thử được trên một vài instance.
- [ ] **Giải bằng SCIP/Pyomo và ghi tỷ lệ conditional-feasible.**
  *Mục tiêu:* mỗi instance có trạng thái feasible/infeasible/timeout rõ ràng; phân biệt infeasible với timeout.
- [ ] **Kiểm tra điều kiện giải tích** `Kp/2 ≤ Σz ≤ K√p/c`.
  *Mục tiêu:* trước hết xác nhận từ code rằng ràng buộc linear (`b`, `q`) không làm đổi điều kiện; sau đó đối chiếu với kết quả SCIP và ghi số instance mâu thuẫn (nếu có).
- [ ] **Với instance conditional-feasible:** đo khoảng cách giữa `x_r` của network và `x_r*` đã sửa.
  *Mục tiêu:* phân phối khoảng cách và vi phạm `h` tương ứng.
- [ ] **Với instance conditional-infeasible:** ghi integer assignment và equality target.
  *Mục tiêu:* bảng liệt kê `ẑ`, `Σẑ`, `p`, nêu được ràng buộc nào bị vi phạm.
- [ ] **Phân loại instance và kết luận.**
  *Mục tiêu:* tỷ lệ "`z` sai" so với "`z` sửa được nhưng `x_r` hỏng", kèm hình trong `figures/`.

## Điều kiện hoàn thành
Tick hết các to-do trên; phân loại wrong `z` vs repairable `z` có số liệu.
