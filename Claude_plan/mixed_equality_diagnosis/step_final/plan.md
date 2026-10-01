# step_final — Intervention ablation và báo cáo

Kế hoạch tổng: [../plan.md](../plan.md).

## Yêu cầu
- Chỉ bắt đầu khi step_4 và ít nhất một trong step_5–7 đã xác định được bottleneck có bằng chứng, và khi người dùng yêu cầu.
- Mỗi intervention phải gắn với một giả thuyết cụ thể; không chạy thử ngẫu nhiên.
- Sau mỗi intervention chạy lại đúng protocol của step_1 (cùng seed, split, metrics).

## Mục tiêu cần đạt
Xác định modification nào xử lý trực tiếp bottleneck đã được chứng minh, và viết báo cáo cuối với chuỗi nhân quả mà từng mắt xích đều có experiment hỗ trợ.

## To-do
### Intervention
- [ ] **Ghi giả thuyết cho từng intervention** trước khi chạy.
  *Mục tiêu:* mỗi intervention có một câu giả thuyết liên kết tới bottleneck từ step_4–7.
- [ ] **RC → LT.**
  *Mục tiêu:* kết luận kết quả có phụ thuộc cơ chế integer correction hay không.
- [ ] **Squared equality penalty → L1/exact penalty.**
  *Mục tiêu:* đo vi phạm dư của equality không projection có về gần 0 hay không.
- [ ] **Penalty method → augmented Lagrangian.**
  *Mục tiêu:* như trên, so với L1.
- [ ] **Step size dùng chung → `η_ineq`, `η_eq` riêng.**
  *Mục tiêu:* projection có hội tụ (chạm ngưỡng dừng) thay vì dao động hay không.
- [ ] **Gradient projection → fixed-`z` equality/Newton repair.**
  *Mục tiêu:* so với oracle ở step_7 và với gradient projection gốc.
- [ ] **Chỉ cân nhắc differentiable constrained layer** nếu các intervention trên chưa giải quyết bottleneck.
  *Mục tiêu:* quyết định có làm hay không, kèm lý do.
- [ ] **Chạy lại protocol step_1 sau mỗi intervention.**
  *Mục tiêu:* bảng so sánh trước/sau cho mọi intervention.

### Báo cáo cuối
- [ ] **Trả lời 4 câu hỏi cuối** của kế hoạch gốc.
  *Mục tiêu:* mỗi câu có câu trả lời ngắn kèm số liệu từ step tương ứng:
  equality có tự làm continuous learning/projection thất bại không; integer correction có làm equality residual nhảy tăng không; `ẑ` có thường làm conditional continuous problem infeasible không; surrogate gradient/projection có cung cấp sai tín hiệu repair sau rounding không.
- [ ] **Nêu chuỗi nhân quả.**
  *Mục tiêu:* chỉ giữ những mắt xích có experiment hỗ trợ.
- [ ] **Viết `../report.md`** theo rule 4-5 trong CLAUDE.md.
  *Mục tiêu:* kết quả đạt được, kết quả chưa đạt kèm nguyên nhân gốc trong 2-3 dòng.

## Điều kiện hoàn thành
Tick hết các to-do trên.
