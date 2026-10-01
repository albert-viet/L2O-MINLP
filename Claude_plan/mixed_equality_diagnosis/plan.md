# Kế hoạch thực hiện: Mixed Equality Diagnosis

Kế hoạch gốc (mục tiêu, giả thuyết, metrics): [../mixed_equality_diagnosis_plan.md](../mixed_equality_diagnosis_plan.md).
Tài liệu này là kế hoạch **thực hiện**: chia nhỏ từng bước thành đầu việc, kèm đầu vào, đầu ra và điều kiện hoàn thành.

## Quy ước chung

### Quy mô thí nghiệm
- Bài toán: MIRB, `K=3` (`--size 3`), `steepness=50`, `p~U(1,8)`, `a~U(0.5,4.5)`.
- Tăng số test instance lên **200+** (cũ: 30) và chạy **3-5 seed** cho mỗi cấu hình, vì kết quả cũ trong `eq_results/` chỉ có 1 seed và 30 instance nên nhiễu lớn.
- Hệ số equality: `eq_coef = 2/√p*` với `p* = 4.5` (`c≈0.9428`), giữ như case `infea_feas_case`.
- Huấn luyện chạy trên `cuda` (hardcode trong `run/*.py`).

### Cấu trúc thư mục kết quả
Mỗi step có `data/` (csv, txt, log) và `figures/`. Báo cáo cuối: `report.md` (theo rule 4-5 trong CLAUDE.md).

### Phụ thuộc giữa các step
`step_0 → step_1 → step_2 → step_3 → step_4 → (step_5, step_6, step_7) → step_final`

Ánh xạ với kế hoạch gốc: step_0 là hạ tầng chung (mới). step_1…step_7 tương ứng Bước 1…7. step_final gồm Bước 8 và báo cáo.

---

## step_0 — Hạ tầng chung

### Mục tiêu
Có một bộ harness duy nhất để chạy mọi biến thể với cùng seed, split, optimizer, epoch, penalty và projection setting, và xuất cùng một bộ metrics.

### Đầu việc
- [ ] Đọc lại `run/rosenbrock_eq.py` và `run_rb_eq.py`, liệt kê các giá trị đang hardcode (train/val/test size, hsize, penalty).
- [ ] Problem factory cho 4 biến thể (mục "Bốn biến thể" bên dưới), tái dùng `src/problem/{math_solver,neuromancer}/rosenbrock_eq.py` và `rosenbrock.py` gốc.
- [ ] Module metrics dùng chung: `|h(x)|`, `V_ineq(x)`, `f(x)`, feasibility rate, đường cong `F(ε)` với `ε ∈ {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}`.
- [ ] Hook ghi `x̄` (latent), `x̂` (sau rounding) và từng vòng lặp projection (dùng `record_history` có sẵn trong `src/postprocess/project.py`).
- [ ] Cơ chế chạy nhiều seed và gom kết quả thành một bảng (mean, std).
- [ ] Chạy thử smoke test: 1 biến thể, 1 seed, ít epoch, kiểm tra metrics đầu ra hợp lý.

### Bốn biến thể
1. MIRB gốc, không equality.
2. MIRB + equality chỉ chứa biến continuous.
3. MIRB + mixed equality, nới lỏng toàn bộ biến integer thành continuous.
4. MIRB + mixed equality + integer correction RC (như hiện tại).

### Điều kiện hoàn thành
Chạy được cả 4 biến thể bằng một lệnh với cùng cấu hình và ra cùng định dạng metrics.

---

## step_1 — Baseline đối chứng

### Mục tiêu
Tách ảnh hưởng của equality, integrality và tương tác equality–integrality.

### Đầu việc
- [ ] Chạy 4 biến thể trên mixed dataset hiện tại (`p~U(1,8)`), mỗi biến thể 3-5 seed.
- [ ] Lưu cùng format: objective, inequality violation, equality violation, feasibility rate, `F(ε)`.
- [ ] Lập bảng so sánh trực tiếp 4 biến thể (mean ± std).
- [ ] Xác định biến thể nào bắt đầu xuất hiện degradation đáng kể.

### Điều kiện hoàn thành
Có bảng so sánh bốn case và kết luận biến thể nào gây degradation.

---

## step_2 — Tách feasible và infeasible region

### Mục tiêu
Loại nhiễu do các instance vốn vô nghiệm theo cấu trúc (`p ≥ p*`).

### Đầu việc
- [ ] Sinh dataset feasible-only (`p < p*`).
- [ ] Train và đánh giá 4 biến thể trên feasible-only dataset.
- [ ] Đánh giá riêng trên `p < p*` và `p ≥ p*` cho model huấn luyện trên mixed dataset.
- [ ] Dùng SCIP xác nhận ground truth feasible/infeasible cho từng test instance.
- [ ] Tính objective gap so với nghiệm exact ở các instance SCIP giải được.

### Điều kiện hoàn thành
Trả lời rõ: model thất bại ngay cả trên instance chắc chắn feasible, hay chỉ thất bại mạnh khi dataset chứa vùng infeasible.

---

## step_3 — Instrument pipeline theo từng stage

### Mục tiêu
Xác định equality violation phát sinh chủ yếu trước rounding, sau rounding hay trong projection.

### Đầu việc
- [ ] Log `h(x̄)` tại output latent.
- [ ] Log `h(x̂)` ngay sau RC correction.
- [ ] Log `h(x_proj^(t))` tại từng vòng projection.
- [ ] Log vi phạm bất đẳng thức song song.
- [ ] Log số tọa độ integer thay đổi mỗi vòng.
- [ ] Log khoảng cách của latent integer variables tới ngưỡng làm tròn.
- [ ] Log `‖∇_{x̄_r} V_eq‖` và `‖∇_{x̄_z} V_eq‖`.
- [ ] Vẽ trajectory của `h` qua toàn pipeline.

### Điều kiện hoàn thành
Chỉ ra được stage đầu tiên làm violation tăng hoặc không thể giảm tiếp.

---

## step_4 — Conditional feasibility với `ẑ` cố định

### Mục tiêu
Tách integer-selection failure khỏi continuous-repair failure. Đây là milestone bắt buộc trước khi thay đổi thuật toán.

### Đầu việc
- [ ] Trích `ẑ` từ RC cho từng test instance.
- [ ] Tạo bài toán continuous có điều kiện với `ẑ` cố định (fix biến integer trong Pyomo).
- [ ] Giải bằng SCIP, ghi tỷ lệ conditional-feasible.
- [ ] Kiểm tra điều kiện giải tích `Kp/2 ≤ Σz ≤ K√p/c` so với kết quả SCIP (trước đó cần xác nhận các ràng buộc linear trong code không làm đổi điều kiện này).
- [ ] Với instance conditional-feasible: đo khoảng cách giữa `x_r` của network và `x_r*` đã sửa.
- [ ] Với instance conditional-infeasible: ghi lại integer assignment và equality target.

### Điều kiện hoàn thành
Phân loại được từng instance: `z` sai hay `z` đúng/sửa được nhưng `x_r` hỏng.

---

## step_5 — Geometry của mixed equality

### Mục tiêu
Kiểm chứng giả thuyết: mixed equality tạo họ feasible slices rời rạc theo integer assignment.

### Đầu việc
- [ ] Viết derivation khoảng dịch của hyperplane equality khi `z` đổi một đơn vị.
- [ ] Tính `Δh = h(x̂) − h(x̄)` từ log step_3 và vẽ phân phối.
- [ ] So sánh `|h(x̄)|` và `|h(x̂)|`.
- [ ] Đánh dấu các sample có integer flip.
- [ ] Đo tương quan giữa integer flip và bước nhảy của equality violation.

### Điều kiện hoàn thành
Có bằng chứng định lượng cho hoặc chống lại giả thuyết quantization-induced equality violation.

---

## step_6 — Surrogate-gradient mismatch

### Mục tiêu
Xác định surrogate gradient của RC/STE có phản ánh đúng biến đổi forward của equality violation hay không.

### Đầu việc
- [ ] Cố định noise Gumbel (seed hoặc eval mode) để forward có tính xác định.
- [ ] Tính surrogate gradient của `V_eq` theo latent integer variables.
- [ ] Perturb từng tọa độ latent với nhiều `ε` nhỏ và tính finite-difference của forward violation.
- [ ] Ghi integer assignment trước/sau perturbation.
- [ ] So sánh dấu và độ lớn giữa surrogate gradient và finite-difference.
- [ ] Đánh dấu vùng gradient khác 0 nhưng forward `z` không đổi.
- [ ] Đánh dấu các threshold crossing làm violation nhảy đột ngột.

### Điều kiện hoàn thành
Kết luận surrogate gradient là aligned, partially aligned hay systematically mismatched.

---

## step_7 — Oracle equality repair

### Mục tiêu
Kiểm tra bottleneck nằm ở bản thân equality hay ở gradient projection.

### Đầu việc
- [ ] Cài đặt chiếu closed-form `x_r⁺ = x_r − (aᵀx_r − c·bᵀz)/‖a‖² · a` với `z` cố định.
- [ ] Đo equality residual trước/sau chiếu.
- [ ] Đo vi phạm bất đẳng thức sau equality-only projection.
- [ ] Cài đặt bài toán repair `min ‖x_r − x̂_r‖²` với đủ bất đẳng thức và equality, giải bằng SCIP/Ipopt.
- [ ] Đo tỷ lệ oracle repair feasible.
- [ ] So sánh hai oracle với gradient projection của Tang.

### Điều kiện hoàn thành
- fixed-`z` repair thường thành công ⇒ cơ chế projection là bottleneck.
- fixed-`z` repair thường infeasible ⇒ integer selection là bottleneck chính.

---

## step_final — Intervention ablation và báo cáo

### Điều kiện bắt đầu
Chỉ bắt đầu khi step_4 và ít nhất một trong step_5–7 đã xác định được bottleneck có bằng chứng.

### Đầu việc
- [ ] Mỗi intervention ghi rõ giả thuyết nó kiểm chứng (không chạy thử ngẫu nhiên).
- [ ] RC → LT, kiểm tra phụ thuộc vào cơ chế integer correction.
- [ ] Squared equality penalty → L1/exact penalty.
- [ ] Penalty method → augmented Lagrangian.
- [ ] Step size dùng chung → `η_ineq`, `η_eq` riêng.
- [ ] Gradient projection → fixed-`z` equality/Newton repair.
- [ ] Chỉ cân nhắc differentiable constrained layer nếu các intervention đơn giản không giải quyết được bottleneck.
- [ ] Chạy lại protocol của step_1 sau mỗi intervention.

### Báo cáo cuối
- [ ] Viết `report.md` (rule 4-5 CLAUDE.md): kết quả đạt được, kết quả chưa đạt kèm nguyên nhân gốc trong 2-3 dòng.
- [ ] Trả lời 4 câu hỏi cuối trong kế hoạch gốc:
  - Equality bản thân có làm continuous learning/projection thất bại không?
  - Integer correction có làm equality residual nhảy tăng không?
  - Predicted integer assignment có thường làm conditional continuous problem infeasible không?
  - Surrogate gradient/projection có cung cấp sai tín hiệu repair sau rounding không?
- [ ] Nêu chuỗi nhân quả, chỉ giữ những mắt xích có experiment hỗ trợ.
