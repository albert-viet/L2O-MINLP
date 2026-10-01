# Báo cáo step_2 — Tách feasible và infeasible region

Cấu hình: như step_1 (K=3, c≈0.9428, p*=4.5, train 2000 / val 200 / test 200 mỗi seed, 5 seed, projection 1000 vòng). Feasible-only: `p~U(1, 4.5)` cho train/val/test (test có `p` tối đa 4.482). Dữ liệu: `data/feasible_only/` (huấn luyện), `data/exact/` (SCIP, 40 file = 2 dataset × 4 biến thể × 5 seed, đủ 200 instance mỗi file), bảng `data/*.csv`, hình `figures/feasible_subset_comparison.png`.

## Kết quả đạt được

**Câu trả lời cho câu hỏi của step: model (V4) thất bại ngay cả trên các instance chắc chắn feasible; vùng infeasible trong dữ liệu huấn luyện không phải nguyên nhân chính.**

| trên tập con chắc chắn feasible (mean ± std qua 5 seed) | V3 mixed, integer relaxed | V4 mixed + RC |
|---|---|---|
| `|h|`, huấn luyện feasible-only, không projection | 0.011 ± 0.001 | **0.428 ± 0.037** (×37) |
| `|h|`, huấn luyện feasible-only, có projection | 0.022 ± 0.004 | **0.420 ± 0.089** |
| feasibility có projection (tol 1e-6) | 21.1% | **1.3%** |
| `F(0.1)` có projection | 29.7% | **9.1%** |
| `|h|` không projection, huấn luyện mixed (so sánh) | 0.011 ± 0.001 | 0.547 ± 0.392 |

- Tập con "chắc chắn feasible" = instance mà SCIP/giải tích cho là có nghiệm (V4: `p<p*` và tồn tại số nguyên `n` với `Kp/2 ≤ n ≤ K√p/c`; 809/1000 instance của feasible-only dataset).
- Huấn luyện trên mixed hay feasible-only cho cùng kết luận: `|h|` của V4 là 0.547 ± 0.39 so với 0.428 ± 0.037 (chênh lệch nằm trong nhiễu seed của mixed), V3 gần như không đổi.
- **Projection xử lý được equality khi không có biến nguyên nhưng không xử lý được ở V4:** `|h|` trên tập feasible giảm 0.396 → 0.003 ở V2 và giữ thấp ở V3 (0.022), nhưng ở V4 gần như không đổi (0.428 → 0.420). Đây là bằng chứng chỉ điểm quanh rounding/projection, chưa phải nguyên nhân gốc.
- `|h|` của V4 không phụ thuộc vùng (mixed, không projection: 0.547 ở `p<p*` và 0.549 ở `p≥p*`), trong khi V2 phụ thuộc mạnh (0.38 so với 1.72), V3 không (0.011 so với 0.009).

**Ground truth SCIP khớp dự đoán giải tích** (`data/ground_truth_vs_analytic.csv`, mixed dataset, 1000 instance/biến thể):
- `p ≥ p*` (484 instance): V2, V3, V4 đều 0 giải được, 484/484 infeasible, đúng như dự đoán (kết quả cũ 0/13).
- `p < p*` (516 instance): V3 giải được 516/516; V2 giải được 478 (38 chạm giới hạn 60 giây); V4 giải được 397, **87 infeasible**, 29 chạm giới hạn, 3 lỗi solver.
- **Phát hiện mới:** với V4, `p<p*` chưa đủ để có nghiệm. 87/516 (mixed) và 191/1000 (feasible-only) instance ở `p<p*` vẫn infeasible vì `Σz` phải nguyên (không có số nguyên trong `[Kp/2, K√p/c]`); số này khớp chính xác dự đoán giải tích. Feasible-only dataset vì vậy vẫn chứa ~19% instance infeasible đối với V4.

**Vùng `p<p*` và `p≥p*` cho model huấn luyện trên mixed** (`data/mixed_by_region.csv`): `V_ineq` ở `p≥p*` cao gấp ~2 lần ở mọi biến thể (V4 không projection: 9.17 so với 4.09); feasibility có projection ở `p≥p*` bằng 0 ở cả 4 biến thể, còn ở `p<p*` là V1 9.7%, V2 5.1%, V3 18.1%, V4 0.7%.

## Kết quả chưa đạt hoặc cần lưu ý
- **Objective gap không đo được chất lượng tối ưu:** gap trung bình âm ở mọi biến thể (không projection khoảng −0.97, có projection −0.14 đến −0.44; `data/objective_gap.csv`), tức objective neural (25-40) thấp hơn nhiều so với nghiệm exact (2000-3000) ở 100% instance không projection. Nguyên nhân gốc: nghiệm neural vi phạm ràng buộc (`V_ineq` ≈ 3.7-4.7) nên có thể thấp hơn optimum khả thi.
- **Giả thuyết cần kiểm trong step sau (chưa kiểm chứng):** penalty bất đẳng thức 50 có thể quá nhỏ so với thang objective. Với V1, loss của nghiệm neural ≈ 25 + 50×4.2 ≈ 235, thấp hơn nhiều so với loss của nghiệm khả thi ≈ 2240, nên việc vi phạm ràng buộc là có lợi cho mô hình. Nếu đúng, feasibility ≈ 0% ở mọi biến thể (kể cả V1) không do equality; cần sweep `penalty` để tách ảnh hưởng này.
- **Tập đối chứng V1 không so sánh trực tiếp giữa hai cột trong hình:** tập "feasible" của V1 trên mixed gồm toàn bộ 1000 instance (cả `p≥p*`), nên `V_ineq` của V1 (6.61 so với 3.97) khác nhau do phân bố `p`, không do cách huấn luyện.
- **Ground truth chưa hoàn toàn đầy đủ:** 17-58 instance mỗi tổ hợp chạm giới hạn 60 giây (V1 19/1000, V2 38/516 ở `p<p*`, V4 29/516) và 3-4 instance lỗi solver (`ApplicationError`, chưa điều tra); chúng bị loại khỏi tính gap. Lần chạy SCIP đầu tiên bị sai do tranh luồng CPU (BLAS), đã chạy lại với 1 luồng mỗi tiến trình.
- 5 seed, độ lệch chuẩn của V4 ở mixed lớn (`|h|` 0.25-1.23), nên chênh lệch giữa mixed và feasible-only ở V4 không kết luận được là có hay không.
