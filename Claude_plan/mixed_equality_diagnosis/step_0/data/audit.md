# Rà soát code hiện có (step_0, to-do 1)

Nguồn: `run/rosenbrock_eq.py`, `run_rb_eq.py`, `src/postprocess/project.py`, `run/utils.py`, `src/problem/{math_solver,neuromancer}/rosenbrock*.py`.

## Môi trường chạy
- Dùng conda env `l2o-minlp`: `export PATH=/home/viethq/miniconda3/envs/l2o-minlp/bin:$PATH`.
- Env có torch 2.14+cu130 (RTX 5080, cuda khả dụng), NeuroMANCER 1.5.2, `scip` 10.0.3, `ipopt`.
- Python mặc định (`/home/viethq/miniconda3/bin/python`) không có neuromancer; `pyomo` chỉ thấy `scip`/`ipopt` khi PATH có bin của env.

## Giá trị đang hardcode và điểm cần tham số hóa
| Vị trí | Giá trị hardcode | Cách xử lý trong harness |
|---|---|---|
| `run_rb_eq.py` | seed 42; train/val/test = 2000/200/30; `steepness=50`; `batch_size=64`; `hlayers_sol=5`, `hlayers_rnd=4`; `lr=1e-3`; `p~U(1,8)`, `a~U(0.5,4.5)`; hsize theo bảng `{3: 8}` | CLI: `--seeds`, `--train_size`, `--n_test`; giữ các giá trị còn lại làm mặc định trong config |
| `run/rosenbrock_eq.py` | `"cuda"` ở nhiều chỗ; `timelimit=60`; `RESULT_DIR="eq_results"`; seed 42 đặt lại trong từng hàm | harness nhận `seed` và `out_dir`; device vẫn là cuda (cuda có sẵn) |
| `run/rosenbrock_eq.py::evaluate_eq` | chạy từng instance; dựng lại Pyomo qua `cal_violation()` mỗi instance; chỉ lưu `x_rnd` | metrics vectorized theo batch; Pyomo chỉ dùng để đối chiếu |
| `run/rosenbrock_eq.py::rndCls` | chỉ biến thể RC + mixed equality | `build_variant` tạo cả 4 biến thể |
| `run/utils.py::train` | `epochs=200`, `patience=20`, `warmup=20` | thêm tham số `epochs` (mặc định 200) |
| `src/postprocess/project.py` | `max_iters=1000`, `step_size=0.01`; chỉ lưu `viol.max()` mỗi vòng | thêm `record_states` để lưu `x` mỗi vòng |
| `src/problem/*/rosenbrock_eq.py` | chỉ có mixed equality, `eq_coef` cố định cho mọi test | thêm `eq_mode` và `relax_int` |

## Điểm cần lưu ý
- Hook `x̄` = output `x` của smap; `x̂` = `x_rnd` (trước projection). `gradientProjection` chạy lại cả smap và rnd (pre/post components) ở mỗi vòng, nên với rnd Gumbel noise sẽ khác nhau giữa các vòng nếu không cố định seed.
- `roundModel._round_vars` chưa kiểm với `int_ind` rỗng (cần ở biến thể 3); kiểm trong to-do 2.
