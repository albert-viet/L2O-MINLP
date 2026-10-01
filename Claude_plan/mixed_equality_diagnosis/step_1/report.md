# Báo cáo step_1 — Baseline đối chứng

Cấu hình: MIRB K=3, mixed dataset `p~U(1,8)`, c≈0.9428 (p*=4.5), train 2000, test 200 instance/seed, 5 seed (0-4), 200 epoch tối đa (early stopping), penalty 50/100, projection 1000 vòng, `step_size=0.01`. Dữ liệu: `data/runs/`, bảng: `data/comparison_table.md`, hình: `figures/variant_comparison.png`.

## Kết quả đạt được (mean ± std qua 5 seed)

| | V1 không eq | V2 eq continuous | V3 mixed, integer relaxed | V4 mixed + RC |
|---|---|---|---|---|
| `|h|` không projection | 0 (không có eq) | 1.03 ± 0.10 | 0.010 ± 0.001 | **0.55 ± 0.39** |
| `V_ineq` không projection | 6.61 ± 0.25 | 11.2 ± 0.9 | 6.33 ± 0.28 | 6.56 ± 0.46 |
| Objective không projection | 24.5 ± 7.8 | 81.8 ± 15 | 25.2 ± 0.6 | 40.9 ± 17 |
| Objective có projection | 1358 ± 331 | 4596 ± 342 | 951 ± 24 | 1026 ± 313 |
| feasibility có projection (tol 1e-6) | 4.9% | 2.7% | **9.3%** | **0.4%** |
| `F(0.1)` có projection | 5.2% | 5.1% | 13.0% | 4.1% |

- **Kết luận chính:** degradation rõ ràng đầu tiên xuất hiện ở bước chuyển **V3 → V4**, tức từ mixed equality trong continuous relaxation sang mixed-integer formulation có RC. Phạm vi kết luận là equality residual `|h|` (và feasibility khi có projection), không phải `V_ineq`.
  - Ở V3, mô hình học mixed equality rất tốt: `|h|` ≈ 0.010, còn objective (25.2 so với 24.5) và `V_ineq` (6.33 so với 6.61) gần như tương đương V1. V3 cũng là biến thể tốt nhất khi có projection.
  - Khi chuyển sang V4, `|h|` tăng lên khoảng 0.55 (×54; cả 5 seed của V4 nằm trong 0.25-1.23, đều ≫ 0.009-0.011 của V3), feasibility có projection giảm 9.3% → 0.4%, `F(0.1)` 13% → 4.1%. Khoảng cách này lớn hơn đáng kể so với nhiễu giữa 5 seed. `V_ineq` (6.56 ± 0.46 so với 6.33 ± 0.28) và objective không projection (40.9 ± 17) chỉ đổi nhẹ, không vượt nhiễu.
  - Điều này hỗ trợ kết luận rằng mixed equality tự nó không nhất thiết gây failure trong continuous space (trong cấu hình đã chạy: K=3, một bộ siêu tham số, 5 seed), nhưng khi tính nguyên được enforce qua cơ chế RC/rounding của framework Tang thì mức thỏa equality suy giảm mạnh.
  - **Step 1 chưa đủ để xác định nguyên nhân gốc** là bản thân tính nguyên, integer assignment sai, discontinuity do rounding, surrogate gradient hay projection. Kết luận hợp lý nhất: **failure được khoanh vùng tại bước chuyển V3 → V4; chưa xác định root cause.**
- V4 khớp `eq_results/infea_feas_case` ở inner (4.50 so với 4.59), linear (2.05 so với 1.95) và objective sau projection (1026 so với 1153).
- Dữ liệu tổng hợp đủ 40 dòng (4 biến thể × 5 seed × 2 stage), không thiếu cột nào.

## Kết quả chưa đạt hoặc cần lưu ý
- **Feasibility không phân biệt được các biến thể:** không projection thì cả 4 biến thể đều 0%, kể cả V1 (không equality). Nguyên nhân gốc: ở cấu hình K=3, penalty 50 mô hình chưa thỏa ràng buộc inner (vi phạm ~4.8 ở V1), nên bất đẳng thức đã là điểm nghẽn trước khi có equality. Nên dùng `|h|`, `V_ineq` và `F(ε)` để so sánh.
- **Không khái quát được rằng "equality nói chung không khó":** V2 (equality chỉ trên biến continuous) có `|h|` = 1.03 và `V_ineq` = 11.2 (linear ~10.9), tệ nhất trong 4 biến thể, nên độ khó còn phụ thuộc geometry cụ thể của equality. Nguyên nhân gốc chưa kiểm chứng; giả thuyết: vế phải `c·K·p/2` dương và tăng theo `p`, xung đột với ràng buộc linear `bᵀx ≤ 0`. Cần kiểm trong các step sau, chưa dùng để kết luận.
- **Projection làm objective tăng mạnh ở mọi biến thể, kể cả V1 (24.5 → 1358, ×55):** vậy hiện tượng này không do equality gây ra. Điều này sửa lại cách quy nguyên nhân trong `analyze.md` (Hiện tượng 3 và 5), vốn gắn nó với equality. Còn "projection làm equality tệ hơn" của case cũ không lặp lại ổn định: `|h|` giảm ở V2 (1.03 → 0.63) và V4 (0.55 → 0.42), chỉ tăng ở V3 (0.010 → 0.028); `h²` của V4 theo từng seed lên xuống lẫn lộn. Case cũ (1 seed, 30 instance) có `h²` = 0.087, nằm ở đầu thấp của dải seed hiện tại (0.07-1.51).
- **Chưa tách được ảnh hưởng của structural infeasibility:** mixed dataset chứa ~50% instance `p ≥ p*` vô nghiệm theo cấu trúc. So sánh V3 so với V4 dùng đúng cùng dữ liệu nên không bị nhiễu bởi điều này, nhưng chưa biết kết luận có giữ trên feasible-only dataset hay không. Gợi ý thăm dò, chưa kết luận: `|h|` của V4 gần như bằng nhau ở hai vùng (0.544 so với 0.551), còn `V_ineq` ở `p ≥ p*` cao gấp ~2 lần ở cả V1.
- 5 seed, độ lệch chuẩn của V4 lớn (`|h|` 0.25-1.23): các so sánh nhỏ giữa V4 và biến thể khác chưa đủ tin cậy.

## Việc chuyển sang các step sau
Step 1 đóng với kết luận khoanh vùng ở trên. Việc xác định root cause thuộc các step sau: feasible-only analysis (step_2), stage-wise instrumentation (step_3), fixed-`ẑ` conditional feasibility (step_4), và phân tích geometry, gradient mismatch, oracle repair (step_5-7).
