# MIRB + ràng buộc đẳng thức — kết quả thực nghiệm kiểm chứng

Ràng buộc đẳng thức được thêm vào bài toán Mixed-Integer Rosenbrock Problem (MIRB):

    sum_i x_2i  =  eq_coef * sum_i x_2i+1     (i = 0..K-1)

tức là: tổng của tất cả các biến liên tục (theo khối) bằng `eq_coef` lần tổng của tất cả các biến nguyên (theo khối). Ràng buộc này trộn cả hai loại biến trong một ràng buộc TỔNG HỢP toàn cục (không phải khớp từng cặp theo từng khối riêng lẻ).

Khi kết hợp ràng buộc đẳng thức này với 2 ràng buộc bất đẳng thức có sẵn — inner (`sum(x_2i+1) >= K*p/2`) và outer (`sum(x_2i^2) <= K*p`) — theo bất đẳng thức Cauchy–Schwarz, ta suy ra được một ngưỡng khả thi giải tích `p* = (2/eq_coef)^2`: các instance có `p > p*` chắc chắn vô nghĩa (infeasible) do cấu trúc bài toán, bất kể giá trị của `a` là gì.

## Cấu hình

- num_blocks (K / --size): 3
- steepness: 50
- eq_coef: 0.942809  (ngưỡng khả thi giải tích p* = 4.5)
- penalty (cho các ràng buộc bất đẳng thức): 50.0
- penalty_eq (cho ràng buộc đẳng thức): 100.0
- project (có bật gradient feasibility projection): True
- hsize / hlayers_sol / hlayers_rnd: 8 / 5 / 4
- kích thước train/val/test: 2000 / 200 / 30

Đây là một lần chạy quy mô nhỏ mang tính minh chứng/kiểm chứng (không phải bộ benchmark đầy đủ của paper), nhằm kiểm tra xem ràng buộc đẳng thức mới có hợp lệ (well-posed) hay không, đối với cả solver chính xác SCIP lẫn surrogate khả vi (neuromancer).

## Kết quả (trung bình trên tập test)

| Nguồn | Inner | Outer | Linear | Equality | Obj Val (trung bình) |
|---|---|---|---|---|---|
| Exact-SCIP | 0 | 0 | 0 | 0 | 2430.35 |
| Learned-NoProject | 4.58836 | 0 | 1.9486 | 0.0873219 | 20.2942 |
| Learned-WithProject | 3.81607 | 0 | 0.194825 | 0.210851 | 1152.91 |

## Hình minh họa (xem thư mục figures/)

- `violation_breakdown.png` — vi phạm trung bình theo từng loại ràng buộc, nhóm theo nguồn (Exact/NoProject/WithProject).
- `equality_violation_hist.png` — phân bố vi phạm ràng buộc đẳng thức trên từng test instance, so sánh có/không có projection.
- `objective_vs_eqviolation.png` — giá trị hàm mục tiêu so với vi phạm ràng buộc đẳng thức trên từng test instance.
- `projection_convergence.png` — đường cong hội tụ của gradient projection (trên 1 instance đại diện), nếu có dùng `--project`.

## Phân tích tách theo vùng (p < 4.5 so với p >= 4.5)

Ngưỡng giải tích `p* = 4.5` được chọn **có chủ đích** (không phải tình cờ) sao cho cả hai vùng đều tồn tại trong miền lấy mẫu `p~U(1,8)`: 17/30 test instance có `p < p*` (về mặt cần thiết là tương thích với inner+outer), 13/30 instance có `p >= p*` (vô nghĩa do cấu trúc bài toán, bất kể `a`).

Lưu ý: với chỉ 30 test instance, mỗi vùng chỉ có khoảng 15 instance — cần lưu ý nhiễu do cỡ mẫu nhỏ khi đọc các số trung bình theo vùng bên dưới.

- Exact-SCIP giải được 14/17 instance ở vùng `p<p*`, so với 0/13 instance ở vùng `p>=p*`.

| Nguồn | Vùng | Inner | Outer | Linear | Equality |
|---|---|---|---|---|---|
| Exact-SCIP | p<4.5 | 0 | 0 | 0 | 0 |
| Exact-SCIP | p>=4.5 | nan | nan | nan | nan |
| Learned-NoProject | p<4.5 | 2.018 | 0 | 2.074 | 0.05932 |
| Learned-NoProject | p>=4.5 | 7.95 | 0 | 1.784 | 0.1239 |
| Learned-WithProject | p<4.5 | 1.419 | 0 | 0.07898 | 0.2213 |
| Learned-WithProject | p>=4.5 | 6.95 | 0 | 0.3463 | 0.1972 |

(Ô "nan" ở vùng `p>=4.5` của Exact-SCIP là số liệu thật: SCIP không giải được instance nào trong vùng này — 0/13 — nên không có giá trị vi phạm nào để tính trung bình, khớp chính xác với dự đoán lý thuyết ở trên rằng vùng này vô nghĩa 100%.)

- `region_split_violation.png` — vi phạm trung bình theo từng loại ràng buộc, tách theo vùng, cho mỗi nguồn.

## Hình bổ sung theo phong cách paper gốc (`result/`)

Tái tạo 3 loại hình từ `img/` cho case này (`eq_coef=0.9428`, `p*=4.5`):

- `result/trajectory.png` — kiểu `img/example.png`/`example2.png`: quỹ đạo relaxed/rounded qua các bước huấn luyện trên lát cắt 2D khối 0, các khối còn lại cố định ở giá trị cuối cùng (phép chiếu minh họa, không phải hình chiếu chính xác 6 chiều).
- `result/method_diagram.png` — kiểu `img/method_RC.png`: sơ đồ RC (Gumbel) với số liệu thật của một instance cụ thể. Chỉ có RC, không có LT.
- `result/method_comparison.png` — kiểu `img/cq_s100_penalty.png`/`img/rb_s100_penalty.png`: %feasibility & objective quét theo `penalty_eq`∈{0.3,1,3,10,30,100} (bản rút gọn: 12/30 test instance, chỉ RC/RC-P). Kết quả thật: **toàn bộ 12/12 điểm đánh giá (6 mức penalty × {RC, RC-P}) đều cho 0% feasibility** trên tập con 12 instance này — không có điểm ngoại lệ nào (khác với case `infeasible_region_case` nơi có đúng 1 điểm đạt 41.7%). Điều này nhất quán với việc case này có `p*` cao hơn (4.5 > 4) nhưng vẫn không đủ để giúp phương pháp học đạt feasibility tuyệt đối trên các instance thuộc vùng lý thuyết khả thi — củng cố thêm kết luận rằng vấn đề nằm ở cơ chế học/projection, không chỉ ở việc chọn `eq_coef`.
