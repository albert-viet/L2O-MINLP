# Phân tích: Thêm ràng buộc đẳng thức trộn biến nguyên–liên tục vào MIRB

Tài liệu này tổng hợp toàn bộ hiện tượng quan sát được khi thêm ràng buộc đẳng thức
`Σx₂ᵢ = eq_coef · Σx₂ᵢ₊₁` (tổng biến liên tục = eq_coef lần tổng biến nguyên, trộn cả
hai loại biến) vào bài toán Mixed-Integer Rosenbrock Problem (MIRB), cùng nguyên nhân
lý thuyết/kỹ thuật của từng hiện tượng. Toàn bộ số liệu dẫn chứng lấy từ các thực nghiệm
thật đã chạy, lưu tại `eq_results/` (3 case: `infeasible_region_case` — eq_coef=1.0,
`infea_feas_case` — eq_coef≈0.9428 (p*=4.5), `modified_gradproj` — cùng eq_coef nhưng
projection đã chuẩn hóa gradient theo nhóm).

## Bối cảnh kỹ thuật cần thiết để hiểu các nguyên nhân

- 4 ràng buộc bất đẳng thức gốc của MIRB được phạt bằng `relu(vi phạm)` trong loss huấn
  luyện (`src/problem/neuromancer/rosenbrock.py`).
- Ràng buộc đẳng thức mới được phạt bằng `(vi phạm)²` (quadratic penalty) trong
  `penaltyLoss_eq.cal_eq_violation` (`src/problem/neuromancer/rosenbrock_eq.py`).
- Bước hậu xử lý `gradientProjection` (`src/postprocess/project.py`) dùng gradient
  descent với một `step_size` cố định trên tổng vi phạm để "chiếu" nghiệm về gần vùng
  khả thi hơn, chạy tối đa `max_iters=1000` vòng, dừng sớm nếu `viol.max() < 1e-6`.

---

## Hiện tượng 1 — Một phần miền tham số vô nghĩa cấu trúc (structural infeasibility)

**Quan sát:** Với `eq_coef=1.0` (`infeasible_region_case`), SCIP báo `infeasible` (không
phải timeout) cho 20/30 test instance. Với `eq_coef≈0.9428` (`infea_feas_case`), tỉ lệ
là 13/30, và cụ thể **0/13 instance ở vùng `p≥4.5` được giải**, trong khi **14/17 instance
ở vùng `p<4.5` được giải** — khớp gần như tuyệt đối với ranh giới lý thuyết.

**Nguyên nhân:** Ràng buộc đẳng thức mới ép `Σx₂ᵢ = eq_coef·Σx₂ᵢ₊₁ ≥ eq_coef·Kp/2` (suy
từ ràng buộc *inner* có sẵn `Σx₂ᵢ₊₁ ≥ Kp/2`). Đồng thời, ràng buộc *outer* có sẵn
`Σx₂ᵢ² ≤ Kp` giới hạn (theo bất đẳng thức Cauchy–Schwarz) `Σx₂ᵢ ≤ K√p`. Hai điều kiện
này chỉ đồng thời thỏa mãn được khi `√p ≥ eq_coef·p/2`, tức `p ≤ p* = (2/eq_coef)²`. Với
`p > p*`, **giao của các ràng buộc là tập rỗng theo đúng nghĩa hình học đại số** — không
tồn tại nghiệm khả thi, độc lập với giá trị của `a`. Đây là thuộc tính của CHÍNH bài toán
(sự tương tác giữa ràng buộc mới và ràng buộc cũ), không phải lỗi của bất kỳ phương pháp
giải nào — kể cả SCIP cũng không thể "sửa" được, chỉ có thể phát hiện đúng.

---

## Hiện tượng 2 — Vi phạm đẳng thức không bao giờ tiến về 0, kể cả khi không có projection

**Quan sát:** Ở cả 3 case, mean Equality Violation sau huấn luyện (không projection) luôn
dương và ổn định quanh 0.084–0.087 — không giảm thêm dù kích thước mạng/huấn luyện đã
hội tụ (early-stopping kích hoạt bình thường, validation loss ổn định).

**Nguyên nhân:** Phạt `(vi phạm)²` là một **quadratic penalty — không phải exact penalty
function**. Theo lý thuyết tối ưu có ràng buộc, nghiệm của bài toán phạt bậc hai chỉ hội
tụ về đúng ràng buộc khi trọng số phạt `λ→∞`; với `λ` hữu hạn (ở đây `penalty_weight_eq
=100`), gradient của số hạng phạt là `2λ·g(x)·∇g(x)` — **lực kéo về 0 yếu dần tuyến tính
khi vi phạm nhỏ dần** (giống lò xo), trong khi lực kéo từ objective và các ràng buộc khác
không hề yếu đi tương ứng. Kết quả: tồn tại một điểm cân bằng ổn định với vi phạm dư khác
0, không phải do huấn luyện chưa đủ mà do **giới hạn toán học cố hữu của loại hàm phạt
đang dùng**. (Ngược lại, 4 ràng buộc bất đẳng thức dùng `relu` — một dạng exact penalty
ℓ1 — nên có thể đạt vi phạm chính xác bằng 0 với `λ` hữu hạn đủ lớn.)

---

## Hiện tượng 3 — Gradient projection làm vi phạm đẳng thức và objective TỆ HƠN, không tốt hơn

**Quan sát:** Ở `infeasible_region_case`: Equality Violation 0.084→0.215 (tăng ×2.6),
Obj Val 29.7→693.9 (tăng ×23). Ở `infea_feas_case`: 0.087→0.211 (×2.4), 20.3→1152.9
(×57). Ngay cả sau khi thử sửa (`modified_gradproj`, chuẩn hóa gradient theo nhóm):
0.087→0.516 (×5.9, còn tệ hơn bản gốc), Obj Val 20.3→874.1 (×43).

**Nguyên nhân:** `relu(vi phạm)` (4 ràng buộc cũ) là hàm không khả vi tại 0, gradient
không tăng theo độ lớn vi phạm (subgradient rời rạc 0/1). `(vi phạm)²` (ràng buộc mới)
khả vi trơn, gradient `2·g·∇g` **tăng tuyến tính không giới hạn khi rời xa 0**. Hai lớp
hàm có **hằng số Lipschitz của gradient khác nhau về bậc độ lớn tùy vị trí nghiệm**. Một
`step_size=0.01` cố định dùng chung cho tổng của cả hai không thể đồng thời ổn định
(`step_size < 2/L`) cho cả hai loại — hệ quả là bước cập nhật quá lớn với thành phần
bình phương (đặc biệt khi vi phạm ban đầu đã lớn), đẩy nghiệm ra xa hơn thay vì gần lại.
Việc chuẩn hóa gradient theo nhóm (norm-1) chỉ cân bằng ĐỘ LỚN bước đi giữa hai nhóm ở
mỗi vòng lặp, chứ không giải quyết được xung đột VỀ HƯỚNG giữa hai nhóm gradient (vẫn có
thể kéo nghiệm theo hai hướng gần như đối lập) — nên không đủ để khắc phục hiện tượng này
(đã kiểm chứng thực nghiệm: vẫn tệ hơn, thậm chí tệ hơn cả bản chưa sửa).

---

## Hiện tượng 4 — Nghiệm dao động (oscillate) không hội tụ trong suốt quá trình projection

**Quan sát:** `projection_convergence.png` (cả 2 case có `--project`) cho thấy đường
cong tổng vi phạm (thang log) có dạng **răng cưa lặp lại**, dao động trong khoảng giá trị
cố định suốt toàn bộ 1000 vòng lặp, không có xu hướng giảm dần và **không bao giờ chạm
ngưỡng dừng `1e-6`** ở bất kỳ instance nào được ghi nhận.

**Nguyên nhân:** Đây là hệ quả trực tiếp của Hiện tượng 3 ở dạng động học theo thời gian:
mỗi vòng lặp, gradient của nhóm bất đẳng thức (relu) kéo nghiệm theo một hướng, gradient
của nhóm đẳng thức (bình phương, độ lớn thay đổi theo vị trí) kéo theo hướng khác — khi
nghiệm di chuyển đủ xa để một nhóm "hài lòng" tạm thời, nhóm còn lại lại bị vi phạm nặng
hơn, tạo vòng lặp phản hồi kéo qua kéo lại. Đây là dấu hiệu kinh điển của **gradient
descent với step-size không phù hợp trên hàm mục tiêu có độ cong không đồng nhất theo
từng chiều/thành phần** — không hội tụ về một điểm cố định (fixed point), mà dao động
quanh một quỹ đạo tuần hoàn hoặc gần-tuần hoàn.

---

## Hiện tượng 5 — Objective bị "quét" đi rất xa khỏi vùng tốt (overshoot) sau projection

**Quan sát:** Boxplot Obj Val của "Learned-WithProject" luôn cao hơn hẳn "Learned-
NoProject" hàng chục lần (×23 đến ×57), dù bản thân "NoProject" đã có objective khá tốt
so với Exact-SCIP.

**Nguyên nhân:** Vì thuật toán không hội tụ (Hiện tượng 4) và luôn chạy hết `max_iters=
1000` vòng (không dừng sớm), nghiệm bị dịch chuyển tích lũy qua **rất nhiều bước cập nhật
có kích thước cố định** (`step_size` không giảm đáng kể dù có `decay`), đẩy nghiệm ra xa
khỏi vị trí ban đầu (sau rounding) — nơi vốn đã gần một điểm tốt của hàm Rosenbrock (một
hàm có "thung lũng" hẹp, dốc đứng hai bên). Khi nghiệm bị đẩy ra khỏi thung lũng này,
giá trị hàm mục tiêu tăng rất nhanh theo cấp số nhân (do số hạng bậc 4 `(x₂ᵢ₊₁-x₂ᵢ²)²`
trong công thức Rosenbrock) — giải thích trực tiếp mức tăng objective lớn bất thường.

---

## Hiện tượng 6 — Phân bố kết quả trở nên phân tán/không đồng nhất hơn sau projection

**Quan sát:** `equality_violation_hist.png`: trước projection, vi phạm đẳng thức của các
instance tập trung chặt gần 0; sau projection, phân bố trải rộng ra tới 0.8, một số
instance vẫn gần 0 nhưng một số khác lại lớn hơn hẳn ban đầu.

**Nguyên nhân:** Vì nghiệm dao động không hội tụ (Hiện tượng 4), kết quả cuối cùng sau
đúng 1000 vòng lặp phụ thuộc vào **pha dao động ngẫu nhiên tại thời điểm dừng** (do
`max_iters` cố định cắt ngang một quá trình chưa hội tụ) — khác nhau giữa các instance
tùy vào điểm khởi đầu (sau rounding) của từng instance rơi vào pha nào của chu kỳ dao
động. Đây không phải một cơ chế cải thiện có kiểm soát, nhất quán trên mọi instance, mà
là kết quả của việc "dừng lại tình cờ" giữa chừng một quá trình bất ổn định.

---

## Hiện tượng 7 — Vi phạm đẳng thức cao và objective cao xuất hiện ĐỒNG THỜI (không phải đánh đổi có chủ đích)

**Quan sát:** `objective_vs_eqviolation.png`: các instance có vi phạm đẳng thức cao nhất
sau projection (0.6–0.8) đồng thời cũng là các instance có objective cao nhất (600–1200)
— tương quan dương rõ rệt, không phải tương quan âm (đánh đổi ngược chiều) như kỳ vọng
nếu projection đang "trade-off" một cách có chủ đích giữa hai tiêu chí.

**Nguyên nhân:** Nếu projection hoạt động như một sự đánh đổi có kiểm soát (giảm vi phạm
bằng cách chấp nhận objective xấu hơn), ta sẽ thấy tương quan ÂM. Tương quan DƯƠNG quan
sát được cho thấy đây là hệ quả của **cùng một hiện tượng mất ổn định** (Hiện tượng 4–5):
những instance mà quá trình dao động "quét" nghiệm đi xa nhất sẽ xấu đi đồng thời trên
CẢ HAI tiêu chí — không có sự đánh đổi thực sự nào đang diễn ra, chỉ có nghiệm bị đẩy
ngẫu nhiên ra xa điểm tốt ban đầu.

---

## Hiện tượng 8 — Tăng penalty_eq (0.3→100, hơn 300 lần) không cải thiện feasibility

**Quan sát:** Sweep qua 6 mức `penalty_eq` (0.3, 1, 3, 10, 30, 100) trên cả 3 case: 35/36
lần đánh giá cho đúng 0% feasibility (0 trong 12 test instance thỏa TẤT CẢ ràng buộc
trong dung sai `1e-6`), không có xu hướng tăng dần đơn điệu theo `penalty_eq` như hành vi
đã thấy trong `img/cq_s100_penalty.png`/`img/rb_s100_penalty.png` gốc (nơi %feasibility
tăng đều 0%→100% khi tăng penalty weight, cho bài toán chỉ có bất đẳng thức).

**Nguyên nhân:** Kết hợp của Hiện tượng 1 (một phần instance vô nghĩa cấu trúc — không
`penalty_eq` nào "sửa" được) và Hiện tượng 2 (quadratic penalty không exact — tăng
`penalty_eq` chỉ làm giảm vi phạm dư theo tỷ lệ `O(1/λ)`, không bao giờ về đúng 0 tại
`λ` hữu hạn, đồng thời `λ` quá lớn còn làm loss landscape khó tối ưu hơn — mất cân bằng
với phần objective). Vì tiêu chí "feasibility" đòi hỏi vi phạm ĐÚNG 0 (trong dung sai
`1e-6`) trên TẤT CẢ 5 ràng buộc cùng lúc, chỉ cần MỘT ràng buộc (thường là Inner) chưa
đạt đúng 0 là đã đủ để tính là "infeasible" — nên dù `penalty_eq` có tăng bao nhiêu,
%feasibility vẫn có thể giữ ở 0% nếu Inner (vốn không phụ thuộc `penalty_eq`, mà phụ
thuộc `penalty` — trọng số bất đẳng thức, được giữ cố định = 50 trong sweep này) chưa
bao giờ đạt đúng 0.

---

## Tổng kết: chuỗi nhân quả giữa các hiện tượng

```
Chọn quadratic penalty cho ràng buộc đẳng thức (không phải exact penalty)
        │
        ├─→ Hiện tượng 2 (vi phạm dư dai dẳng, dù không projection)
        │
        └─→ Gradient có độ lớn/hướng lệch pha so với relu-penalty của 4 ràng buộc cũ
                │
                ├─→ Hiện tượng 3 (projection làm vi phạm + objective tệ hơn)
                │        │
                │        └─→ Hiện tượng 4 (dao động không hội tụ, chạy hết max_iters)
                │                │
                │                ├─→ Hiện tượng 5 (overshoot objective)
                │                ├─→ Hiện tượng 6 (phân bố kết quả phân tán)
                │                └─→ Hiện tượng 7 (vi phạm cao & objective cao cùng lúc)
                │
                └─→ Hiện tượng 8 (tăng penalty_eq không cứu được feasibility)

Riêng biệt, độc lập với lựa chọn loại penalty:
Ràng buộc đẳng thức mới tương tác với ràng buộc outer có sẵn (Cauchy–Schwarz)
        │
        └─→ Hiện tượng 1 (một phần miền tham số vô nghĩa cấu trúc — không thể sửa
             bằng bất kỳ phương pháp tối ưu nào, chỉ có thể tránh bằng cách chọn lại
             eq_coef/miền tham số, hoặc chấp nhận và báo cáo riêng theo vùng)
```
