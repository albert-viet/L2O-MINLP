# step_7 — Oracle equality repair

Kế hoạch tổng: [../plan.md](../plan.md). Dùng `ẑ` và `x̂_r` đã trích ở step_4.

## Yêu cầu
Thử hai oracle với `z` cố định:
1. chiếu closed-form chỉ trên equality: `x_r⁺ = x_r − (aᵀx_r − c·bᵀz)/‖a‖² · a`;
2. repair có cả bất đẳng thức: `min ‖x_r − x̂_r‖²` s.t. `g(x_r, ẑ) ≤ 0`, `h(x_r, ẑ) = 0`.

## Mục tiêu cần đạt
Xác định bottleneck nằm ở cơ chế gradient projection hay ở integer selection:
- fixed-`z` repair thường thành công ⇒ projection là bottleneck;
- fixed-`z` repair thường infeasible ⇒ integer selection là bottleneck chính.

## To-do
- [ ] **Cài đặt chiếu closed-form** với `z` cố định.
  *Mục tiêu:* kiểm tra trên instance đơn giản rằng sau chiếu `h = 0` (tới sai số số học).
- [ ] **Đo equality residual trước/sau oracle projection.**
  *Mục tiêu:* phân phối `|h|` trước và sau cho mọi test instance.
- [ ] **Đo vi phạm bất đẳng thức sau equality-only projection.**
  *Mục tiêu:* biết chiếu equality có làm hỏng bất đẳng thức hay không.
- [ ] **Cài đặt bài toán repair đầy đủ** (SCIP/Ipopt).
  *Mục tiêu:* giải được; phân biệt rõ infeasible với timeout.
- [ ] **Đo tỷ lệ oracle repair feasible.**
  *Mục tiêu:* tỷ lệ theo seed, đối chiếu với tỷ lệ conditional-feasible của step_4.
- [ ] **So sánh hai oracle với gradient projection của Tang.**
  *Mục tiêu:* bảng `|h|`, `V_ineq`, objective cho ba phương pháp trên cùng instance.
- [ ] **Kết luận bottleneck.**
  *Mục tiêu:* một đoạn ngắn theo hai trường hợp ở phần mục tiêu, kèm số liệu.

## Điều kiện hoàn thành
Tick hết các to-do trên.
