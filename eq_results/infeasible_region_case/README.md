# MIRB + equality constraint — demo/validation results

Equality constraint added to the Mixed-Integer Rosenbrock Problem (MIRB):

    sum_i x_2i  =  sum_i x_2i+1     (i = 0..K-1)

i.e. the total of all continuous block-variables equals the total of all integer block-variables. This mixes both variable types in a single global (aggregate) constraint, as opposed to a per-block match.

## Configuration

- num_blocks (K / --size): 3
- steepness: 50
- penalty (inequality): 50.0
- penalty_eq (equality): 100.0
- project (gradient projection enabled): True
- hsize / hlayers_sol / hlayers_rnd: 8 / 5 / 4
- train/val/test size: 2000 / 200 / 30

This is a small-scale demo/validation run (not the paper's full benchmark sweep) meant to check that the new equality constraint is well-posed for both the exact SCIP solver and the differentiable neuromancer surrogate.

## Results (mean over test set)

| Source | Inner | Outer | Linear | Equality | Obj Val (mean) |
|---|---|---|---|---|---|
| Exact-SCIP | 0 | 0 | 0 | 0 | 2333.57 |
| Learned-NoProject | 4.78836 | 0 | 1.57128 | 0.0840522 | 29.7259 |
| Learned-WithProject | 3.94814 | 0 | 0.423678 | 0.214598 | 693.851 |

## Figures (see figures/)

- `violation_breakdown.png` — mean violation per constraint type, grouped by source.
- `equality_violation_hist.png` — distribution of the equality-violation across test instances, with vs. without projection.
- `objective_vs_eqviolation.png` — objective value vs. equality violation per test instance.
- `projection_convergence.png` — gradient-projection convergence curve (1 representative instance), if `--project` was used.

## Key findings

1. **The equality constraint is well-posed and exactly satisfiable when the
   instance is feasible.** Exact-SCIP achieves 0 violation on every
   constraint (including the new equality) on the instances it manages to
   solve — this confirms the Pyomo formula `sum(x_2i) == sum(x_2i+1)` is
   correct and that `abc_solver._constraint_violation`/`penalty()` handle it
   correctly out of the box, with no changes needed there.

2. **`Number of unsolved instances: 20` out of 30 for Exact-SCIP is NOT a
   timeout — it is genuine structural infeasibility**, confirmed by a
   diagnostic re-run showing SCIP's `termination_condition == infeasible`
   (not `maxTimeLimit`) for the failing instances. Analytically: the
   equality forces `sum(x_2i) = sum(x_2i+1) >= K*p/2` (from the *inner*
   constraint), while the *outer* constraint `sum(x_2i^2) <= K*p` bounds
   `sum(x_2i) <= K*sqrt(p)` (Cauchy–Schwarz, tight when all x_2i are equal).
   These two bounds are simultaneously satisfiable only when `sqrt(p) >=
   p/2`, i.e. **p <= 4** — so for `p` in `(4, 8]` (about 4/7 ≈ 57% of the
   `Uniform(1,8)` sampling range used here), the problem is infeasible *by
   construction*, independent of `a`. This matches the observed ~67%
   infeasibility rate. **This is a direct, empirical illustration of the
   Root-Cause-Analysis warning from the earlier A3 review: an added
   equality constraint can silently conflict with existing constraints
   across large parts of the parametric family**, and should be checked
   analytically (as done here) whenever a new equality is introduced.

3. **The learned surrogate satisfies the (aggregate) equality reasonably
   well even without projection** (mean violation 0.084, vs. 4.79/1.57 for
   the still-imperfect inner/linear inequalities) — consistent with the
   design rationale for choosing a *global* Σ-equality (learnable by the
   continuous branch adjusting its own sum) rather than a per-block
   equality (which the earlier analysis flagged as effectively unlearnable).

4. **Gradient projection did not uniformly help here — it increased the
   mean equality violation (0.084 → 0.215) and the mean objective (29.7 →
   693.9), while improving the linear/inner terms.** `projection_convergence.png`
   shows the total violation oscillating for the full 1000 iterations
   without ever reaching the `1e-6` stopping threshold. This empirically
   confirms Countermeasure #4 from the earlier A3 analysis: the equality's
   squared-violation gradient has a different scale than the inequalities'
   `relu`-violation gradients, and the shared fixed `step_size=0.01` is not
   well-tuned for this mixed objective — a per-term step size or
   normalization (not implemented here) would likely be needed to make
   projection reliably helpful once an equality term is present.

## Hình bổ sung theo phong cách paper gốc (`result/`)

Ba hình dưới đây tái tạo đúng LOẠI/cách biểu diễn của các hình gốc trong `img/`, áp dụng cho case này (`eq_coef=1.0`, `p*=4`):

- `result/trajectory.png` — cùng kiểu `img/example.png`/`example2.png`: quỹ đạo nghiệm relaxed (tím) và rounded (xanh đậm) qua các bước huấn luyện, vẽ trên lát cắt 2D của khối 0 (`x₀,x₁`), chồng lên contour mục tiêu và các đường biên ràng buộc (inner/outer/equality), có đánh dấu nghiệm tối ưu Exact-SCIP. Các khối còn lại được giữ cố định ở giá trị học được cuối cùng để có thể vẽ lát cắt 2D — đây là một PHÉP CHIẾU minh họa, không phải hình chiếu ràng buộc chính xác cho toàn bộ không gian 6 chiều.
- `result/method_diagram.png` — cùng kiểu `img/method_RC.png`: sơ đồ luồng RC (Gumbel rounding) với số liệu thật từ một test instance cụ thể (input, nghiệm relaxed, hidden state, nghiệm mixed-integer, công thức loss). Không có phiên bản LT vì thực nghiệm này chỉ triển khai RC (Gumbel), không triển khai Learnable-Thresholding cho bài toán có ràng buộc đẳng thức.
- `result/method_comparison.png` — cùng kiểu `img/cq_s100_penalty.png`/`img/rb_s100_penalty.png`: %feasibility và objective value quét theo `penalty_eq` ∈ {0.3,1,3,10,30,100} (thang log), so sánh RC (không projection) vs RC-P (có projection) vs Exact-SCIP (đường tham chiếu, không phụ thuộc penalty). **Đây là bản rút gọn** so với bản gốc của paper: chỉ 6 mức penalty (thay vì 9), chỉ 12 test instance đầu tiên (thay vì 30, để việc đánh giá có projection — vốn chậm — hoàn thành trong vài phút), và chỉ có RC/RC-P (không có LT/LT-P). Kết quả thật: trên toàn bộ 6×2=12 lần đánh giá, **11/12 cho 0% feasibility** (0/12 instance thỏa mãn TẤT CẢ ràng buộc trong dung sai `1e-6`); duy nhất 1 điểm ngoại lệ đạt 41.7% (`penalty_eq=10`, có projection) — không có xu hướng đơn điệu rõ ràng theo `penalty_eq`, khác hẳn hành vi tăng dần đều đặn quan sát được ở `img/cq_s100_penalty.png`/`img/rb_s100_penalty.png` cho bài toán CHỈ CÓ bất đẳng thức. Đây là bằng chứng định lượng bổ sung cho kết luận đã nêu ở trên: thêm ràng buộc đẳng thức phá vỡ hành vi "tăng feasibility đơn điệu theo penalty weight" vốn là đặc trưng của framework gốc.
