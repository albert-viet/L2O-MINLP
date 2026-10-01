# step_0 — Hạ tầng chung

Kế hoạch tổng: [../plan.md](../plan.md). Step sau: step_1 (chỉ bắt đầu khi được người dùng yêu cầu).

## Yêu cầu
- Một bộ harness duy nhất chạy được 4 biến thể bài toán với cùng seed, split, optimizer, số epoch, penalty và projection setting.
- Cùng một bộ metrics cho mọi biến thể: `|h(x)|`, `V_ineq(x)`, `f(x)`, feasibility rate, `F(ε)` với `ε ∈ {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}`.
- Quy mô: MIRB `K=3`, `steepness=50`, `p~U(1,8)`, `a~U(0.5,4.5)`, `c≈0.9428` (p*=4.5), test 200+ instance, 3-5 seed.

## Mục tiêu cần đạt
Chạy được cả 4 biến thể bằng một lệnh, với cùng cấu hình, ra cùng định dạng metrics, và có hook ghi `x̄`, `x̂` và từng vòng projection để các step sau dùng lại.

### Bốn biến thể
1. MIRB gốc, không equality.
2. MIRB + equality chỉ chứa biến continuous.
3. MIRB + mixed equality, nới lỏng toàn bộ biến integer thành continuous.
4. MIRB + mixed equality + integer correction RC (như hiện tại).

## To-do
- [x] **Rà soát code hiện có** (`run/rosenbrock_eq.py`, `run_rb_eq.py`, `src/postprocess/project.py`).
  *Mục tiêu:* có danh sách đầy đủ các giá trị đang hardcode (train/val/test size, hsize, penalty, device) và các điểm cần tham số hóa.
- [x] **Problem factory cho 4 biến thể**, tái dùng `src/problem/{math_solver,neuromancer}/rosenbrock_eq.py` và `rosenbrock.py` gốc.
  *Mục tiêu:* gọi factory với tên biến thể trả về cặp model exact (Pyomo) và loss (NeuroMANCER) đúng; kiểm tra bằng cách giải một instance ở mỗi biến thể và xác nhận số ràng buộc đúng.
- [x] **Module metrics dùng chung.**
  *Mục tiêu:* một hàm nhận nghiệm và tham số, trả về `|h|`, `V_ineq`, objective, feasibility và `F(ε)`; kết quả khớp với `cal_violation()` của solver trên ít nhất vài instance.
- [x] **Hook ghi log pipeline** (`x̄`, `x̂`, từng vòng projection), dùng `record_history` có sẵn.
  *Mục tiêu:* một lần chạy đánh giá lưu được ba loại tensor này ra `data/` mà không làm đổi kết quả so với khi tắt hook.
- [x] **Cơ chế chạy nhiều seed và gom bảng** (mean, std).
  *Mục tiêu:* chạy 3 seed ra một bảng tổng hợp; cùng seed chạy lại cho kết quả giống nhau.
- [x] **Smoke test:** 1 biến thể, 1 seed, ít epoch.
  *Mục tiêu:* chạy xong không lỗi, metrics trong khoảng hợp lý, file đầu ra đúng định dạng.

## Điều kiện hoàn thành
Tick hết các to-do trên. Lưu dữ liệu vào `data/`, hình vào `figures/`.
