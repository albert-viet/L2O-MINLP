# step_6 — Surrogate-gradient mismatch

Kế hoạch tổng: [../plan.md](../plan.md). Áp dụng cho biến thể 4 (RC).

## Yêu cầu
So sánh ba tín hiệu trên cùng một điểm latent: (1) surrogate gradient của RC/STE, (2) finite-difference của forward equality violation, (3) integer assignment thực tế trước/sau perturbation.

## Mục tiêu cần đạt
Kết luận surrogate gradient là aligned, partially aligned hay systematically mismatched với hành vi forward rời rạc.

## To-do
- [ ] **Cố định noise Gumbel** (seed hoặc eval mode).
  *Mục tiêu:* hai lần forward cùng input cho đúng cùng `z`, để finite-difference có nghĩa.
- [ ] **Tính surrogate gradient của `V_eq`** theo latent integer variables.
  *Mục tiêu:* gradient cho mọi tọa độ, mọi instance trong tập mẫu.
- [ ] **Perturb từng tọa độ latent với nhiều `ε` nhỏ** và tính finite-difference của forward violation.
  *Mục tiêu:* bảng finite-difference theo `ε` và tọa độ.
- [ ] **Ghi integer assignment trước/sau perturbation.**
  *Mục tiêu:* biết perturbation nào làm đổi `z`.
- [ ] **So sánh dấu và độ lớn** giữa surrogate gradient và finite-difference.
  *Mục tiêu:* tỷ lệ khớp dấu và tỷ số độ lớn, kèm hình trong `figures/`.
- [ ] **Đánh dấu vùng gradient khác 0 nhưng forward `z` không đổi.**
  *Mục tiêu:* tỷ lệ mẫu rơi vào vùng này.
- [ ] **Đánh dấu các threshold crossing làm violation nhảy đột ngột.**
  *Mục tiêu:* độ lớn bước nhảy và tần suất.
- [ ] **Kết luận aligned / partially aligned / systematically mismatched.**
  *Mục tiêu:* một đoạn ngắn kèm số liệu.

## Điều kiện hoàn thành
Tick hết các to-do trên.
