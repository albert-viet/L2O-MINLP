# step_2 — Tách feasible và infeasible region

Kế hoạch tổng: [../plan.md](../plan.md). Dùng harness của step_0 và kết quả mixed dataset của step_1.

## Yêu cầu
- Ngưỡng giải tích `p* = (2/c)²`; chia dữ liệu thành `D_feas = {p<p*}` và `D_infeas = {p≥p*}`.
- Đánh giá 4 biến thể trên feasible-only dataset; đánh giá riêng hai vùng cho model huấn luyện trên mixed dataset.

## Mục tiêu cần đạt
Trả lời rõ: model thất bại ngay cả trên các instance chắc chắn feasible, hay chỉ thất bại mạnh khi dataset chứa vùng infeasible.

## To-do
- [x] **Sinh dataset feasible-only** (`p<p*`).
  *Mục tiêu:* mọi instance có `p<p*`; kích thước train/val/test tương đương mixed dataset.
- [x] **Train và đánh giá 4 biến thể trên feasible-only dataset.**
  *Mục tiêu:* metrics đủ các seed, cùng format với step_1.
- [x] **Đánh giá riêng `p<p*` và `p≥p*` cho model huấn luyện trên mixed dataset.**
  *Mục tiêu:* bảng metrics tách theo vùng cho từng biến thể.
- [x] **Xác nhận ground truth bằng SCIP** cho từng test instance.
  *Mục tiêu:* tỷ lệ instance SCIP báo infeasible ở `p≥p*` và solved ở `p<p*` được ghi lại, đối chiếu với dự đoán giải tích (kết quả cũ: 0/13 solved ở `p≥p*`).
- [x] **Tính objective gap** so với nghiệm exact ở các instance SCIP giải được.
  *Mục tiêu:* gap trung bình theo biến thể và theo vùng.
- [x] **Kết luận feasible-only vs mixed.**
  *Mục tiêu:* một đoạn ngắn trả lời câu hỏi ở phần mục tiêu, kèm số liệu.

## Điều kiện hoàn thành
Tick hết các to-do trên và câu hỏi ở phần mục tiêu có câu trả lời rõ ràng.

## Kết quả
Xem [report.md](report.md). Dữ liệu trong `data/`, hình `figures/feasible_subset_comparison.png`.
