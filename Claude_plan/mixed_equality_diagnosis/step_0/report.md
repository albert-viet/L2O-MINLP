# Báo cáo step_0 — Hạ tầng chung

## Kết quả đạt được
- 6/6 to-do trong [plan.md](plan.md) đã tick, mỗi to-do có kiểm chứng riêng (script và log trong `data/`).
- **Harness 4 biến thể** chạy bằng một lệnh: `python run_diag_eq.py --variants V1 V2 V3 V4 --seeds 0 1 2 --project --save_hooks --out_dir <dir>` (cần `export PATH=/home/viethq/miniconda3/envs/l2o-minlp/bin:$PATH`).
  - V1 không equality; V2 `Σx₂ᵢ = c·K·p/2` (chỉ biến continuous, cùng ngưỡng p*); V3 mixed equality với integer relaxed; V4 mixed equality + RC.
  - Factory: số ràng buộc đúng (4/5/5/5), số biến integer đúng (3/3/0/3), SCIP giải được cả 4 ở `p=3.2` với vi phạm 0 (`data/check_factory.log`).
- **Metrics dùng chung** (`run/diag_eq/metrics.py`): `|h|`, `V_ineq`, objective, feasibility, `F(ε)`. Khớp `cal_violation()` của Pyomo với sai khác ≤ 3.2e-16 trên 20 instance, V4 và V2, có và không projection (`data/check_metrics.log`).
- **Hook:** lưu `x̄`, `x̂`, trạng thái latent và rounded ở mỗi vòng projection. Bật và tắt hook cho CSV giống hệt (`data/check_hooks.log`).
- **Nhiều seed:** 3 seed × 4 biến thể ra bảng mean ± std; chạy lại seed 0 cho kết quả giống hệt (`data/multiseed_run1/`, `data/multiseed_run2/`).
- **Projection theo batch:** khớp projection từng instance, sai khác ≤ 2.4e-7 (`data/check_projection.log`), nên nhanh hơn nhiều mà không đổi kết quả.
- **Smoke test** V4, 1 seed, 10 epoch, 200 test, có projection: chạy xong không lỗi, đủ file CSV, `.pt`, bảng tổng hợp (`data/smoke/`).

## Kết quả chưa đạt hoặc cần lưu ý
- Chưa chạy thí nghiệm thật (đủ epoch, 3-5 seed, 200+ test): đó là việc của step_1. Số liệu trong smoke test và multiseed (6-10 epoch) chỉ để kiểm tra pipeline, không dùng để kết luận.
- Early stopping chưa được kích hoạt ở cấu hình ngắn nên `train_iters` chỉ phản ánh số epoch đặt ra; cần xem lại khi chạy đủ 200 epoch ở step_1.
- Feasibility ở smoke test bằng 0 hoặc gần 0 do model chưa train đủ, không phải lỗi harness.
- `data/smoke/V4_seed0_hooks.pt` nặng ~13 MB; cân nhắc không commit file này.

## Thay đổi code
- Sửa: `src/problem/math_solver/rosenbrock_eq.py`, `src/problem/neuromancer/rosenbrock_eq.py` (thêm `eq_mode`, `relax_int`, mặc định giữ nguyên hành vi cũ), `src/postprocess/project.py` (`record_states`), `src/problem/neuromancer/trainer.py` (`total_iters`), `run/utils.py` (tham số `epochs`, trả về trainer).
- Mới: `run/diag_eq/{variants,metrics,pipeline}.py`, `run_diag_eq.py`.
