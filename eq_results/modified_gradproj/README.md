# MIRB + ràng buộc đẳng thức — Gradient Projection chuẩn hóa theo nhóm

Thí nghiệm này kiểm chứng bản sửa của `gradientProjection` (`src/postprocess/project.py`): khi bật `normalize_by_group=True`, vi phạm được tách thành nhóm bất đẳng thức (inner+outer+linear) và nhóm đẳng thức (equality), mỗi nhóm được chuẩn hóa gradient về norm 1 trước khi cộng lại để cập nhật nghiệm — thay vì lấy một gradient duy nhất từ tổng vi phạm thô như trước đây (nguyên nhân gây lệch thang gradient đã chẩn đoán ở case 2).

Cấu hình bài toán (ràng buộc, hệ số `eq_coef`, ngưỡng `p*`) giữ **y hệt case 2** (`eq_results/infea_feas_case/`) để có thể so sánh trực tiếp — điểm khác biệt DUY NHẤT là cờ `--normalize_projection` được bật.

## Cấu hình

- num_blocks (K / --size): 3
- eq_coef: 0.942809 (ngưỡng khả thi giải tích p* = 4.5)
- penalty (bất đẳng thức) / penalty_eq (đẳng thức): 50.0 / 100.0
- normalize_projection (chuẩn hóa gradient theo nhóm): True
- train/val/test size: 2000 / 200 / 30

## Kết quả (trung bình trên tập test)

| Nguồn | Inner | Outer | Linear | Equality | Obj Val (tb) |
|---|---|---|---|---|---|
| Exact-SCIP | 0 | 0 | 0 | 0 | 2430.35 |
| Learned-NoProject | 4.58836 | 0 | 1.9486 | 0.0873219 | 20.2942 |
| Learned-WithProject (đã chuẩn hóa) | 4.2715 | 0 | 0.315306 | 0.515596 | 874.101 |

## Phân tích theo từng hình vẽ (xem thư mục figures/)

### `violation_breakdown.png`
So sánh vi phạm trung bình theo 4 loại ràng buộc, nhóm theo nguồn (Exact-SCIP / Learned-NoProject / Learned-WithProject). Ở đây, cột Equality của Learned-WithProject là **0.5156**, so với Learned-NoProject là **0.08732** — vẫn cao hơn (chưa giảm), cho thấy chuẩn hóa gradient đơn thuần chưa đủ để khắc phục hoàn toàn vấn đề, dù đã thay đổi cơ chế cập nhật.

### `equality_violation_hist.png`
Phân bố vi phạm đẳng thức trên từng test instance, so sánh No-project vs With-project (đã chuẩn hóa). Nếu phân bố With-project dịch chuyển rõ về phía 0 so với No-project, đó là bằng chứng trực quan cho D1; nếu hai phân bố chồng lấn nhiều hoặc With-project lệch phải, D1 chưa đạt rõ ràng ở mức phân phối (không chỉ ở trung bình).

### `objective_vs_eqviolation.png`
Tương quan giữa giá trị mục tiêu và vi phạm đẳng thức. Obj Val trung bình tăng từ **20.29** (No-project) lên **874.1** (With-project, đã chuẩn hóa) — tỉ lệ tăng **×43.1**. Vượt ngưỡng chấp nhận (≤×3) theo tiêu chí D2 — vẫn còn đánh đổi objective lớn để đổi lấy giảm vi phạm.

### `projection_convergence.png`
Đường cong tổng vi phạm (`viol.max()`, thang log) theo số vòng lặp projection, trên 1 instance đại diện. So với case 2 cũ (dao động suốt 1000 vòng, không chạm ngưỡng dừng `1e-6`), cần đối chiếu trực tiếp hình này để đánh giá D3 — xem hình để biết đường cong có phẳng ra / hội tụ đơn điệu hơn hay vẫn dao động tương tự.

### `region_split_violation.png`
Vi phạm trung bình theo từng loại ràng buộc, tách theo vùng lý thuyết (`p<4.5` vs `p>=4.5`), cho mỗi nguồn — cho biết chuẩn hóa gradient có giúp đều ở cả hai vùng hay chỉ cải thiện ở một vùng.

## So sánh trực tiếp D1 / D2 / D3 với case 2 (chưa chuẩn hóa gradient)

| Tiêu chí | Case 2 (cũ, chưa chuẩn hóa) | Case này (đã chuẩn hóa theo nhóm) | Đạt? |
|---|---|---|---|
| D1: Equality Viol giảm sau projection | 0.08732 → 0.2109 (TĂNG) | 0.08732 → 0.5156 (TĂNG) | ❌ |
| D2: Obj Val không tăng quá ×3 | ×56.8 | ×43.1 | ❌ |
| D3: Hội tụ chạm ngưỡng `1e-6` hoặc ổn định | Dao động suốt 1000 vòng, không chạm | Xem `projection_convergence.png` (mô tả định tính ở trên) | (xem hình) |

**Kết luận:** Chuẩn hóa gradient theo nhóm chưa giải quyết được cả D1 lẫn D2 trong cấu hình `step_size`/`max_iters` hiện tại, cho thấy đây là một cải thiện đúng hướng nhưng có thể cần thêm điều chỉnh (ví dụ trọng số riêng giữa 2 nhóm thay vì cộng đều sau chuẩn hóa, hoặc `step_size` nhỏ hơn) để đạt đầy đủ cả 3 tiêu chí.

## Hình bổ sung theo phong cách paper gốc (`result/`)

Tái tạo 3 loại hình từ `img/` cho case này (cùng `eq_coef=0.9428` như `infea_feas_case`, nhưng dùng `gradientProjection(..., normalize_by_group=True)`):

- `result/trajectory.png`, `result/method_diagram.png` — giống hệt các hình tương ứng của `infea_feas_case` về mặt huấn luyện nghiệm (normalize_by_group chỉ ảnh hưởng đến bước projection lúc đánh giá, không ảnh hưởng đến huấn luyện `smap`/`rnd`), nên 2 hình này không phản ánh trực tiếp sự khác biệt của case này — dùng `method_comparison.png` và `figures/projection_convergence.png` để so sánh mới đúng trọng tâm.
- `result/method_comparison.png` — kiểu `img/cq_s100_penalty.png`/`img/rb_s100_penalty.png`, quét `penalty_eq`∈{0.3,1,3,10,30,100} với projection ĐÃ chuẩn hóa theo nhóm. Kết quả thật: **toàn bộ 12/12 điểm đánh giá vẫn cho 0% feasibility** — giống hệt kết quả của `infea_feas_case` (chưa chuẩn hóa). Đây là bằng chứng định lượng bổ sung, nhất quán với bảng D1/D2/D3 ở trên: **chuẩn hóa gradient theo nhóm KHÔNG cải thiện được tỉ lệ feasibility** trong dải `penalty_eq` đã quét, củng cố thêm kết luận rằng cần điều chỉnh sâu hơn (trọng số riêng giữa 2 nhóm, hoặc giảm `step_size`) chứ không chỉ chuẩn hóa norm gradient.

