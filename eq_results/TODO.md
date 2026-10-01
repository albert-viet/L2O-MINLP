# To-do list: thêm ràng buộc đẳng thức trộn biến nguyên–liên tục vào MIRB

Tất cả nằm trong commit `9347b5e`, và bản phân tích nằm ở `4f33794`.

## A. Thiết kế ràng buộc
- [x] Chọn dạng ràng buộc đẳng thức **toàn cục trộn biến**: `Σ x₂ᵢ = eq_coef · Σ x₂ᵢ₊₁`. Đây là tổng biến liên tục bằng `eq_coef` lần tổng biến nguyên, không khớp theo từng khối. Lý do: ràng buộc khớp từng khối bị đánh giá là gần như không học được.
- [x] Phân tích giải tích khả thi bằng Cauchy–Schwarz kết hợp với ràng buộc inner và outer có sẵn. Kết quả là ngưỡng `p* = (2/eq_coef)²`. Với `p > p*` bài toán vô nghiệm theo cấu trúc, bất kể `a`.
- [x] Thêm tham số `eq_coef` để chỉnh ngưỡng. Case 1 dùng `eq_coef=1.0` (p*=4). Case 2 dùng `eq_coef=2/√p*≈0.9428` (p*=4.5), chọn có chủ đích để miền lấy mẫu `p~U(1,8)` có cả vùng khả thi và vùng vô nghiệm.

## B. Mô hình solver chính xác (Pyomo/SCIP)
- [x] Tạo [src/problem/math_solver/rosenbrock_eq.py](../src/problem/math_solver/rosenbrock_eq.py). Lớp `rosenbrock_eq` kế thừa `rosenbrock` và thêm đẳng thức bằng `m.cons.add(...)`.
- [x] Kiểm chứng rằng `abc_solver._constraint_violation` và `penalty()` xử lý đẳng thức đúng mà không cần sửa. Exact-SCIP cho vi phạm 0 trên mọi instance giải được.
- [x] Chẩn đoán "20/30 unsolved": chạy lại để xác nhận `termination_condition == infeasible`, không phải timeout.

## C. Mô hình khả vi (NeuroMANCER)
- [x] Tạo [src/problem/neuromancer/rosenbrock_eq.py](../src/problem/neuromancer/rosenbrock_eq.py), lớp `penaltyLoss_eq` kế thừa `penaltyLoss`.
- [x] `cal_eq_violation`: phạt bậc hai `(Σx₂ᵢ − eq_coef·Σx₂ᵢ₊₁)²`.
- [x] Loss = obj + `penalty`·viol_ineq + `penalty_eq`·viol_eq. Trọng số `penalty_eq` được tách riêng khỏi `penalty`.
- [x] Override `cal_constr_viol` để trả tổng vi phạm thô, dùng cho `gradientProjection`.
- [x] Thêm `cal_violation_breakdown`, trả về vi phạm tách theo inner, outer, linear, equality để log và vẽ hình.
- [x] Đăng ký `msRosenbrockEq` và `nmRosenbrockEq` trong `__init__.py` của `src/problem/`, `math_solver/` và `neuromancer/`.

## D. Pipeline thực nghiệm
- [x] Tạo [run/rosenbrock_eq.py](../run/rosenbrock_eq.py) gồm:
  - `exact()`: giải SCIP.
  - `rndCls()`: huấn luyện RC (Gumbel), đánh giá có và không projection.
  - `evaluate_eq()`: tính vi phạm tách theo từng loại.
  - `make_figures()`: vẽ hình.
  - `analyze_regions()`: tách kết quả theo vùng `p<p*` và `p≥p*`.
  - `write_readme()`: sinh README tự động.
- [x] Tạo script chạy [run_rb_eq.py](../run_rb_eq.py) với các cờ `--size`, `--penalty`, `--penalty_eq`, `--project`, `--eq_p_threshold` và `--out_dir`.
- [x] Chạy ở cấu hình nhỏ để kiểm chứng (K=3, train 2000, val 200, test 30), không chạy benchmark đầy đủ của paper.

## E. Sửa gradient projection
- [x] Mở rộng [src/postprocess/project.py](../src/postprocess/project.py) với `record_history=True`, để ghi đường cong hội tụ.
- [x] Thêm `normalize_by_group=True`. Cờ này tách vi phạm thành nhóm bất đẳng thức và nhóm đẳng thức, chuẩn hóa gradient mỗi nhóm về norm 1, rồi cộng lại.

## F. Hình ảnh và báo cáo
- [x] Vẽ 5 hình trong `figures/` cho mỗi case:
  - `violation_breakdown`
  - `equality_violation_hist`
  - `objective_vs_eqviolation`
  - `projection_convergence`
  - `region_split_violation`
- [x] Tạo [run/rosenbrock_eq_plots.py](../run/rosenbrock_eq_plots.py) để tái tạo 3 hình theo phong cách paper trong `result/`:
  - `trajectory`
  - `method_diagram`
  - `method_comparison`, quét `penalty_eq` ∈ {0.3, 1, 3, 10, 30, 100}
- [x] Chạy 3 case, kết quả lưu ở `eq_results/`:
  - `infeasible_region_case`
  - `infea_feas_case`
  - `modified_gradproj`
- [x] Viết README cho từng case và bảng so sánh D1/D2/D3 (case chuẩn hóa so với case chưa chuẩn hóa).
- [x] Viết [analyze.md](../analyze.md), tổng hợp 8 hiện tượng cùng chuỗi nhân quả.
- [x] Cập nhật [CLAUDE.md](../CLAUDE.md).

## G. Kết luận rút ra từ các việc trên
- [x] Equality hợp lệ ở solver chính xác. Phần học (learned) chưa đạt feasibility: vi phạm dư ~0.085, projection làm tệ hơn, và 0% feasibility ở 35/36 lần đánh giá.

## H. Chưa làm (để tránh hiểu nhầm là đã xong)
- [ ] Chưa có run đối chứng không có equality ở cùng cấu hình.
- [ ] Chưa chạy K lớn hơn.
- [ ] Chưa làm bản LT (Learnable Thresholding) cho bài toán có equality.
- [ ] Chưa thử exact penalty, augmented Lagrangian, projection Newton, hoặc `step_size` riêng cho từng nhóm.
