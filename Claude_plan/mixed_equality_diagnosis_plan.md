# Mixed Equality Diagnosis

## Mục tiêu

Xác định **vì sao ràng buộc đẳng thức trộn biến liên tục–nguyên làm suy giảm hoặc phá vỡ cơ chế learning/projection của Tang**, và chỉ ra chính xác failure xuất hiện ở đâu trong pipeline:

```latex
\[
\xi \rightarrow \bar{x} \rightarrow \phi(\bar{x}) \rightarrow \hat{x} \rightarrow \text{projection} \rightarrow x^{\mathrm{final}}
\]
```

Kết quả cuối cùng phải phân biệt được ít nhất ba khả năng:

```latex
\[
\text{equality geometry},\qquad
\text{integer correction},\qquad
\text{projection / surrogate gradient}.
\]
```

---

## Bước 1 — Khóa baseline đối chứng

### Mục tiêu
Tách ảnh hưởng của **equality**, **integrality** và **tương tác equality–integrality**.

### Yêu cầu bắt buộc
- Giữ nguyên cùng seed, \(K\), train/val/test split, optimizer, số epoch, penalty và projection setting.
- Chỉ thay đúng một yếu tố giữa các biến thể.
- Báo cáo cùng một bộ metrics cho tất cả biến thể.

### To-do
- [ ] Chạy MIRB gốc không equality.
- [ ] Chạy MIRB + equality chỉ chứa biến continuous.
- [ ] Chạy MIRB + mixed equality nhưng relax toàn bộ integer variables thành continuous.
- [ ] Chạy MIRB + mixed equality + integer correction RC như hiện tại.
- [ ] Lưu cùng format: objective, inequality violation, equality violation, feasibility rate.
- [ ] So sánh chênh lệch giữa bốn biến thể.

### Điều kiện hoàn thành
Bước này chỉ hoàn thành khi xác định được biến thể nào bắt đầu xuất hiện degradation đáng kể và có bảng so sánh trực tiếp giữa bốn case.

---

## Bước 2 — Tách feasible và infeasible regions

### Mục tiêu
Loại bỏ nhiễu do các instance vốn đã vô nghiệm theo cấu trúc.

### Yêu cầu bắt buộc

```latex
\[
p^\star=\left(\frac{2}{c_{\mathrm{eq}}}\right)^2.
\]
```

Phân chia dữ liệu thành:

```latex
\[
\mathcal D_{\mathrm{feas}}=\{p<p^\star\},\qquad
\mathcal D_{\mathrm{infeas}}=\{p\ge p^\star\}.
\]
```

### To-do
- [ ] Sinh dataset chỉ gồm \(p<p^\star\).
- [ ] Train/evaluate model trên feasible-only dataset.
- [ ] Train/evaluate model trên mixed dataset hiện tại.
- [ ] Đánh giá riêng trên \(p<p^\star\) và \(p\ge p^\star\).
- [ ] Kiểm tra SCIP để xác nhận feasible/infeasible ground truth.

### Điều kiện hoàn thành
Bước này chỉ hoàn thành khi có thể trả lời rõ: **model thất bại ngay cả trên các instance chắc chắn feasible hay chỉ thất bại mạnh khi dataset chứa vùng infeasible**.

---

## Bước 3 — Instrument pipeline theo từng stage

### Mục tiêu
Xác định equality violation phát sinh chủ yếu **trước rounding, sau rounding hay trong projection**.

### Yêu cầu bắt buộc

```latex
\[
h(\bar{x}),\qquad
h(\hat{x}),\qquad
h(x_{\mathrm{proj}}^{(1)}),\ldots,h(x_{\mathrm{proj}}^{(T)}).
\]
```

Đồng thời log:

```latex
\[
\|\nabla_{\bar{x}_r}V_{\mathrm{eq}}\|,\qquad
\|\nabla_{\bar{x}_z}V_{\mathrm{eq}}\|.
\]
```

### To-do
- [ ] Log equality residual tại output latent \(\bar{x}\).
- [ ] Log equality residual ngay sau RC/LT correction.
- [ ] Log equality residual tại từng projection iteration.
- [ ] Log inequality violation song song.
- [ ] Log số integer coordinates thay đổi ở mỗi iteration.
- [ ] Log khoảng cách của latent integer variables tới rounding threshold.
- [ ] Vẽ trajectory của \(h\) qua toàn pipeline.

### Điều kiện hoàn thành
Bước này chỉ hoàn thành khi chỉ ra được **stage đầu tiên làm violation tăng hoặc không thể tiếp tục giảm**.

---

## Bước 4 — Conditional feasibility với predicted integer variables

### Mục tiêu
Tách **integer-selection failure** khỏi **continuous-repair failure**.

```latex
\[
\text{find }x_r
\quad\text{s.t.}\quad
g(x_r,\hat z;\xi)\le 0,\qquad
h(x_r,\hat z;\xi)=0.
\]
```

### Yêu cầu bắt buộc
- Giữ cố định \(\hat z\) do network dự đoán.
- Dùng exact solver cho continuous conditional problem.
- Phân loại từng instance theo kết quả conditional feasibility.

### To-do
- [ ] Trích xuất \(\hat z\) từ RC cho từng test instance.
- [ ] Tạo conditional optimization problem với fixed \(\hat z\).
- [ ] Giải bằng SCIP/Pyomo.
- [ ] Ghi tỷ lệ conditional-feasible.
- [ ] Với conditional-feasible instances, đo khoảng cách giữa neural \(x_r\) và repaired \(x_r^\star\).
- [ ] Với conditional-infeasible instances, ghi lại integer assignment và equality target tương ứng.

### Điều kiện hoàn thành

```latex
\[
\text{wrong }z
\quad\text{vs}\quad
\text{correct/repairable }z\text{ but failed }x_r.
\]
```

Đây là milestone bắt buộc trước khi thay đổi thuật toán.

---

## Bước 5 — Phân tích geometry của mixed equality

### Mục tiêu
Kiểm chứng giả thuyết rằng mixed equality tạo ra một họ feasible slices rời rạc theo integer assignment.

```latex
\[
h(x_r,z)=a^\top x_r-c\,b^\top z=0,
\]
```

```latex
\[
\mathcal H_z=\{x_r:a^\top x_r=c\,b^\top z\},
\qquad
\mathcal F_h=\bigcup_{z\in\mathbb Z^{n_z}}\mathcal H_z.
\]
```

### Yêu cầu bắt buộc
- Phân tích cả lý thuyết và thực nghiệm.
- Đo tác động trực tiếp của integer rounding lên equality residual.

### To-do
- [ ] Viết derivation cho khoảng dịch của equality hyperplane khi \(z\) thay đổi một đơn vị.
- [ ] Định nghĩa và log:

```latex
\[
\Delta h=h(\hat{x})-h(\bar{x}).
\]
```

- [ ] Vẽ phân phối \(\Delta h\).
- [ ] So sánh \(|h(\bar{x})|\) và \(|h(\hat{x})|\).
- [ ] Đánh dấu các sample có integer flip.
- [ ] Kiểm tra correlation giữa integer flip và equality violation jump.

### Điều kiện hoàn thành
Bước này chỉ hoàn thành khi có bằng chứng định lượng cho hoặc chống lại giả thuyết **quantization-induced equality violation**.

---

## Bước 6 — Kiểm tra surrogate-gradient mismatch

### Mục tiêu
Xác định surrogate gradient của RC/STE có phản ánh đúng biến đổi của **forward equality violation** hay không.

```latex
\[
z(\bar z)=\operatorname{round}_{\mathrm{RC}}(\bar z)
\]
```

```latex
\[
\frac{\widetilde{\partial z}}{\partial \bar z}\neq 0.
\]
```

### Yêu cầu bắt buộc
So sánh ba tín hiệu trên cùng latent point:
1. surrogate gradient;
2. finite-difference forward change;
3. integer assignment thực tế.

### To-do
- [ ] Tính surrogate gradient của \(V_{\mathrm{eq}}\) theo latent integer variables.
- [ ] Perturb từng latent coordinate với nhiều \(\epsilon\) nhỏ.
- [ ] Tính finite-difference của forward equality violation.
- [ ] Ghi integer assignment trước/sau perturbation.
- [ ] So sánh dấu và độ lớn của surrogate gradient với finite-difference.
- [ ] Đánh dấu các vùng gradient khác 0 nhưng forward \(z\) không đổi.
- [ ] Đánh dấu các threshold crossing làm violation nhảy đột ngột.

### Điều kiện hoàn thành
Bước này chỉ hoàn thành khi kết luận được surrogate gradient là **aligned**, **partially aligned** hay **systematically mismatched** với forward discrete behavior.

---

## Bước 7 — Oracle equality repair

### Mục tiêu
Kiểm tra xem bottleneck nằm ở equality itself hay ở gradient projection của Tang.

```latex
\[
h(x_r,z)=a^\top x_r-cb^\top z,
\]
```

```latex
\[
x_r^+
=
x_r-
\frac{a^\top x_r-cb^\top z}{\|a\|_2^2}a.
\]
```

### Yêu cầu bắt buộc
Thử hai oracle:
1. equality-only closed-form projection;
2. constrained repair có cả inequalities.

### To-do
- [ ] Implement closed-form equality projection với fixed \(z\).
- [ ] Đo equality residual trước/sau oracle projection.
- [ ] Kiểm tra inequality violation sau equality-only projection.
- [ ] Implement repair problem:

```latex
\[
\min_{x_r}\|x_r-\hat{x}_r\|_2^2
\quad\text{s.t.}\quad
g(x_r,\hat z)\le0,\quad
h(x_r,\hat z)=0.
\]
```

- [ ] Đo tỷ lệ oracle repair feasible.
- [ ] So sánh oracle repair với Tang gradient projection.

### Điều kiện hoàn thành
- fixed-\(z\) repair thường thành công \(\Rightarrow\) projection mechanism là bottleneck;
- fixed-\(z\) repair thường infeasible \(\Rightarrow\) integer selection là bottleneck chính.

---

## Bước 8 — Intervention ablation

### Mục tiêu
Chỉ thử modification sau khi nguyên nhân failure đã được cô lập.

### Yêu cầu bắt buộc
Mỗi intervention phải gắn với một hypothesis cụ thể; không chạy thử ngẫu nhiên.

### To-do
- [ ] RC \(\rightarrow\) LT để kiểm tra dependence vào integer correction mechanism.
- [ ] Squared equality penalty \(\rightarrow\) \(L_1\)/exact penalty.
- [ ] Penalty method \(\rightarrow\) augmented Lagrangian.
- [ ] Shared step size \(\rightarrow\) \(\eta_{\mathrm{ineq}}\), \(\eta_{\mathrm{eq}}\) riêng.
- [ ] Gradient projection \(\rightarrow\) fixed-\(z\) equality/Newton repair.
- [ ] Chỉ cân nhắc differentiable constrained layer nếu các intervention đơn giản chưa giải quyết bottleneck.
- [ ] Chạy lại cùng baseline protocol sau mỗi intervention.

### Điều kiện hoàn thành
Bước này chỉ hoàn thành khi xác định được modification nào xử lý trực tiếp bottleneck đã được chứng minh ở các bước trước.

---

## Metrics bắt buộc

```latex
\[
|h(x)|,\qquad
V_{\mathrm{ineq}}(x),\qquad
f(x),\qquad
\|\nabla V_{\mathrm{eq}}\|.
\]
```

Bổ sung:
- conditional-feasible rate của predicted \(\hat z\);
- số integer flips trong projection;
- \(\Delta h\) do rounding;
- distance-to-threshold của latent integer variables;
- objective gap nếu có exact solution;
- feasibility rate theo feasible/infeasible region.

Dùng tolerance curve:

```latex
\[
F(\epsilon)
=
\Pr\!\left(
|h(x)|\le\epsilon,\;
V_{\mathrm{ineq}}(x)\le\epsilon
\right),
\qquad
\epsilon\in\{10^{-1},10^{-2},10^{-3},10^{-4},10^{-5}\}.
\]
```

---

## Kết quả cuối cùng cần đạt

- [ ] Equality bản thân có làm continuous learning/projection thất bại không?
- [ ] Integer correction có trực tiếp làm equality residual nhảy tăng không?
- [ ] Predicted integer assignment có thường làm conditional continuous problem infeasible không?
- [ ] Surrogate gradient/projection có cung cấp sai tín hiệu repair sau rounding không?

Kết luận cuối cùng phải có dạng nhân quả, ví dụ:

```latex
\[
\text{latent solution}
\rightarrow
\text{integer correction}
\rightarrow
\text{equality-slice jump}
\rightarrow
\text{surrogate-gradient mismatch}
\rightarrow
\text{projection failure}.
\]
```

Chuỗi trên chỉ được chấp nhận nếu từng mắt xích được hỗ trợ bởi experiment tương ứng.

---

## Thứ tự ưu tiên thực hiện

1. **Bước 1 — baseline đối chứng**
2. **Bước 2 — feasible-only study**
3. **Bước 3 — stage-wise instrumentation**
4. **Bước 4 — conditional feasibility với fixed \(\hat z\)**
5. **Bước 5–7 — geometry, gradient mismatch, oracle repair**
6. **Bước 8 — intervention ablation**

Không chuyển sang Bước 8 trước khi Bước 4 và ít nhất một trong Bước 5–7 đã xác định được bottleneck có bằng chứng.
