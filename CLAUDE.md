# CLAUDE.md

File này hướng dẫn Claude Code (claude.ai/code) khi làm việc với mã nguồn trong repository này.

## Dự án

Cài đặt chính thức của bài báo "Learning to Optimize for Mixed-Integer Nonlinear Programming" (arXiv:2410.11061). Đây là một framework Learning-to-Optimize (L2O) huấn luyện mạng nơ-ron để dự đoán nghiệm chất lượng cao cho các bài toán MINLP có tham số, sử dụng **integer correction layers** (lớp hiệu chỉnh số nguyên, để đảm bảo tính nguyên) và **integer feasibility projection** (bước hậu xử lý dựa trên gradient để giảm vi phạm ràng buộc).

## Thiết lập môi trường

Repo không có sẵn `requirements.txt`/`pyproject.toml`; các phụ thuộc được cài qua `create_env.sh`, script này giả định một cụm HPC dùng module (kiểu Compute Canada: `module load`, `virtualenv`). Các phụ thuộc chính: PyTorch, NeuroMANCER (`==1.5.2`, cài với `--no-deps`), Pyomo, Gurobi (`gurobipy`), SCIP (qua `SolverFactory("scip")` của Pyomo), IPOPT, và Coin-HSL (biên dịch từ mã nguồn cho bộ giải tuyến tính của IPOPT). Gurobi và SCIP phải được cấp phép/cài đặt riêng trên hệ thống; không có cách cài chỉ bằng pip để có môi trường chạy được.

## Chạy thực nghiệm

Repo không có test suite, bước build hay linter. Ba entry point dưới đây là các lệnh "run" thực sự:

```bash
# Integer Quadratic Problems (IQP)
python run_qp.py --size 5 [--penalty 20] [--project] [--warmstart]

# Integer Non-Convex Problems (INP)
python run_nc.py --size 10 [--penalty 1] [--project]

# Mixed-Integer Rosenbrock Problems (MIRB)
python run_rb.py --size 100 [--penalty 10] [--project]
```

- `--size`: kích thước bài toán, một trong `{5, 10, 20, 50, 100, 200, 500, 1000}` (điều khiển cả `num_var`/`num_ineq` và độ rộng lớp ẩn thông qua bảng tra trong mỗi `run_*.py`).
- `--penalty`: trọng số của soft-penalty cho vi phạm ràng buộc trong hàm loss huấn luyện (mặc định 20 hoặc tùy cấu hình).
- `--project`: bật bước hậu xử lý chiếu khả thi dựa trên gradient (`src/postprocess/project.py`).
- `--warmstart` (chỉ IQP): so sánh nghiệm chính xác với warm start của Gurobi được dẫn bởi ML (`run/quadratic.py::warmstart`), và yêu cầu các file CSV nghiệm ML đã có sẵn trong `result/`.

Mỗi script `run_*.py` tạo dữ liệu tham số tổng hợp (vế phải `b` lấy ngẫu nhiên đều), chia dữ liệu bằng `src.utlis.data_split`, rồi chuyển sang module tương ứng trong `run/` (`run/quadratic.py`, `run/nonconvex.py`, `run/rosenbrock.py`). Module này chạy lần lượt nhiều baseline/phương pháp và ghi mỗi phương pháp một file CSV vào `result/<problem>_<method><penalty>_<size>-<size>.csv` (thư mục `result/` được tạo ngầm và không được commit vào git). Toàn bộ quá trình huấn luyện được hardcode chạy trên `cuda` (`.to("cuda")` xuyên suốt `run/*.py`), không có đường chạy dự phòng trên CPU.

Các random seed (`random`, `numpy`, `torch`, `torch.cuda`) được cố định bằng 42 ở đầu mỗi `run_*.py` và được đặt lại bên trong mỗi hàm phương pháp trong `run/*.py` để đảm bảo tái lập.

Thư mục `test/` chỉ chứa các Jupyter notebook dùng để phân tích/trực quan hóa kết quả (không có test tự động `pytest`/`unittest`).

## Kiến trúc

Framework xem một bài toán MINLP là hai biểu diễn song song của cùng một bài toán, và hai biểu diễn này phải nhất quán với nhau:

1. **`src/problem/math_solver/`** — các mô hình solver chính xác (Pyomo + SCIP/Gurobi), mỗi loại bài toán một file (`quadratic.py`, `nonconvex.py`, `rosenbrock.py`), tất cả kế thừa `abcParamSolver` (`abc_solver.py`). Lớp trừu tượng này bọc một mô hình Pyomo có các tham số có thể thay đổi (`self.params`), biến quyết định (`self.vars`) và ràng buộc (`self.cons`), đồng thời cung cấp các thao tác dùng xuyên suốt pipeline: `solve()`, `set_param_val()`, `relax()` (nới lỏng LP/NLP qua `TransformationFactory("core.relax_integer_vars")`), `penalty()` (chuyển ràng buộc cứng thành hàm mục tiêu soft-penalty thông qua biến slack), `first_solution_heuristic()`/`primal_heuristic()` (các lần giải chỉ dùng heuristic của SCIP, làm baseline nhanh), `set_warm_start()`, và các thước đo vi phạm (`cal_violation()`). Các property `int_ind`/`bin_ind` cho biết chỉ số biến nào là nguyên/nhị phân, thông tin này được truyền vào các lớp làm tròn của mạng nơ-ron để chúng biết đầu ra nào cần hiệu chỉnh số nguyên.

2. **`src/problem/neuromancer/`** — cùng các bài toán đó được biểu diễn dưới dạng hàm loss khả vi của NeuroMANCER (`quadratic.py`, `nonconvex.py`, `rosenbrock.py`, được import thành `nmQuadratic`/`nmNonconvex`/`nmRosenbrock` trong `src/problem/__init__.py`), dùng để huấn luyện end-to-end mạng nơ-ron ánh xạ nghiệm.

Cả hai được gom trong `src/problem/__init__.py` với quy ước đặt tên tương ứng `ms*`/`nm*`. Khi thêm một lớp bài toán mới, cần có cả bản math-solver và bản neuromancer.

**Pipeline nơ-ron** (xem `run/quadratic.py` làm mẫu chuẩn, được lặp lại trong `run/nonconvex.py`/`run/rosenbrock.py`):
- Một **mạng ánh xạ nghiệm** (`nm.system.Node` bọc một MLP) ánh xạ trực tiếp các tham số bài toán (ví dụ `b`) sang nghiệm nới lỏng liên tục `x`.
- Một **lớp làm tròn/hiệu chỉnh** (`src/func/rnd.py`, được export qua `src/func/__init__.py`) nhận `x` và tạo ra `x_rnd` thỏa tính nguyên. Các biến thể được chọn theo tên phương pháp trong `run/*.py`:
  - `roundGumbelModel` → "RC" (Rounding Classification)
  - `roundThresholdModel` → "LT" (Learnable Thresholding)
  - `roundSTEModel` → làm tròn dùng straight-through-estimator ("RS")
  - không có lớp làm tròn + heuristic `naive_round` (`src/heuristic/round.py`) → baseline "RL"
  - `src/func/ste.py` cài đặt các phép toán straight-through-estimator mà các lớp này dùng để lan truyền ngược qua phép làm tròn (không khả vi).
- Hai thành phần được nối thành `nn.ModuleList([smap, rnd])` và huấn luyện đồng thời qua `src.problem.neuromancer.trainer.trainer` (được bọc bởi `run/utils.py::train`), theo hàm loss `nm*` tương ứng (soft penalty cho vi phạm ràng buộc, nhân với trọng số `--penalty`).
- Ở giai đoạn đánh giá, `run/*.py::evaluate()` có thể bọc bước suy luận bằng `src.postprocess.project.gradientProjection` (cờ `--project`), tức là hiệu chỉnh lặp dựa trên gradient cho `x_rnd` theo các ràng buộc khả vi, trước khi trả giá trị về mô hình math-solver (`model.vars[...].value = ...`) để tính chính xác giá trị mục tiêu và vi phạm bằng `cal_violation()`.

**Quy ước tên phương pháp trong `run/*.py`** (baseline so với phương pháp học, tương ứng với tiền tố cq=quadratic, nc=nonconvex, rb=rosenbrock trong tên file CSV kết quả):
- `exact` — giải đầy đủ bằng SCIP/Gurobi (baseline sự thật nền, chậm).
- `relRnd` — giải bài toán nới lỏng LP/NLP rồi làm tròn đơn giản.
- `root` — heuristic nghiệm khả thi đầu tiên của solver (`first_solution_heuristic`).
- `rndCls`/`rndThd`/`rndSte` — các lớp làm tròn học được (RC/LT/RS) cộng với projection tùy chọn.
- `lrnRnd` — chỉ dùng ánh xạ nghiệm học được, làm tròn đơn giản ở bước sau (không có lớp làm tròn học được).
- `warmstart` (chỉ quadratic) — đưa file CSV nghiệm ML đã tính trước vào Gurobi làm MIP start và so sánh với lần giải khởi động nguội.

`src/heuristic/resolve.py` chứa các heuristic để giải lại/sửa chữa một instance bài toán (khác với phép làm tròn đơn giản trong `round.py`).

`src/utlis/data.py` cung cấp `data_split` (chia train/val/test cùng `torch.utils.data.Dataset`/`collate_fn` cho dict tham số) và `solve_test.py` có các hàm hỗ trợ đánh giá các instance đã giải, được dùng trong cả ba script `run_*.py`.

## Quy tắc quản lý kế hoạch

1. Mọi file markdown liên quan đến kế hoạch triển khai và to-do list của project phải nằm trong thư mục lớn `Claude_plan/`.
2. Luôn chia nhỏ công việc trong một kế hoạch thành các đầu việc nhỏ hơn dựa theo heading cấp 2 và cấp 3. Tạo to-do list các việc cần làm để đạt được mục tiêu và yêu cầu của kế hoạch.
3. Nếu một công việc rất lớn được chia thành nhiều bước thực hiện, phải tạo các subfolder nằm trong folder tương ứng của công việc lớn đó và đặt tên theo thứ tự các bước từ `step_0` đến `step_final`. Nếu kết quả có hình vẽ và dữ liệu dạng csv, txt, log, ... thì phải tạo các subfolder tên là `data/` và `figures/` nằm trong từng subfolder step để lưu kết quả.

4. Mỗi khi hoàn thành một công việc lớn, phải có một báo cáo ngắn gọn trình bày kết quả đạt được. Với kết quả chưa đạt được, nêu ngắn gọn nguyên nhân gốc, chỉ trong 2-3 dòng.
5. Lưu báo cáo dưới dạng file markdown (ví dụ `report.md`) ngay tại nơi lưu công việc lớn đó, tức là trong folder `Claude_plan/<ten_cong_viec>/`.

6. Trong mỗi subfolder step của công việc, phải tạo một file markdown plan thực hiện step đó (ví dụ `plan.md`).
7. Mỗi file markdown plan trong subfolder step phải mô tả yêu cầu và mục tiêu cần đạt được trong step đó. Sau đó đưa ra các đầu việc dạng to-do list, mỗi to-do có mục tiêu cụ thể. Chỉ được tick hoàn thành một to-do khi đã đáp ứng yêu cầu và hoàn thành mục tiêu của to-do đó. Chỉ khi tick hết các to-do thì step đó mới được tính là hoàn thành.
8. Không tự ý chuyển sang công việc hoặc step khác khi chưa hỏi người dùng hoặc người dùng chưa đưa ra yêu cầu thực hiện.

Ví dụ cấu trúc:

```
Claude_plan/
└── <ten_cong_viec>/
    ├── plan.md
    ├── report.md
    ├── step_0/
    │   ├── plan.md
    │   ├── data/
    │   └── figures/
    ├── step_1/
    │   ├── plan.md
    │   ├── data/
    │   └── figures/
    └── step_final/
        ├── plan.md
        ├── data/
        └── figures/
```
