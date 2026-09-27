# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Official implementation of "Learning to Optimize for Mixed-Integer Nonlinear Programming" (arXiv:2410.11061). A Learning-to-Optimize (L2O) framework that trains neural networks to predict high-quality solutions to parametric MINLPs, using **integer correction layers** (to enforce integrality) and **integer feasibility projection** (a gradient-based post-processing step to reduce constraint violation).

## Environment setup

No `requirements.txt`/`pyproject.toml` is committed; dependencies are installed via `create_env.sh`, which assumes an HPC/module-based cluster (Compute Canada style: `module load`, `virtualenv`). Key dependencies: PyTorch, NeuroMANCER (`==1.5.2`, installed with `--no-deps`), Pyomo, Gurobi (`gurobipy`), SCIP (via Pyomo's `SolverFactory("scip")`), IPOPT, and Coin-HSL (compiled from source for IPOPT's linear solver). Gurobi and SCIP must be separately licensed/installed on the system; there is no pure-pip path to a working environment.

## Running experiments

There is no test suite, build step, or linter configured in this repo. The three entry points below are the actual "run" commands:

```bash
# Integer Quadratic Problems (IQP)
python run_qp.py --size 5 [--penalty 20] [--project] [--warmstart]

# Integer Non-Convex Problems (INP)
python run_nc.py --size 10 [--penalty 1] [--project]

# Mixed-Integer Rosenbrock Problems (MIRB)
python run_rb.py --size 100 [--penalty 10] [--project]
```

- `--size`: problem size, one of `{5, 10, 20, 50, 100, 200, 500, 1000}` (controls both `num_var`/`num_ineq` and the hidden layer width via a lookup table in each `run_*.py`).
- `--penalty`: penalty weight for constraint-violation soft-penalty in the training loss (default 20/config-specific).
- `--project`: enables the gradient-based feasibility projection post-processing step (`src/postprocess/project.py`).
- `--warmstart` (IQP only): compares exact solves against ML-guided Gurobi warm starts (`run/quadratic.py::warmstart`), and requires ML solution CSVs already present in `result/`.

Each `run_*.py` script builds synthetic parametric data (uniform-random RHS `b`), splits it via `src.utlis.data_split`, and dispatches to a matching module in `run/` (`run/quadratic.py`, `run/nonconvex.py`, `run/rosenbrock.py`) which runs several baselines/methods in sequence and writes one CSV per method to `result/<problem>_<method><penalty>_<size>-<size>.csv` (directory `result/` is created implicitly and is not checked into git). All training is hardcoded to run on `cuda` (`.to("cuda")` throughout `run/*.py`) — there's no CPU fallback path.

Random seeds (`random`, `numpy`, `torch`, `torch.cuda`) are fixed to 42 at the top of every `run_*.py` and re-seeded inside each method function in `run/*.py` for reproducibility.

The `test/` directory holds only Jupyter notebooks used for analysis/visualization of results (no automated `pytest`/`unittest` tests).

## Architecture

The framework treats a MINLP as two parallel representations of the same problem that must stay consistent:

1. **`src/problem/math_solver/`** — exact solver models (Pyomo + SCIP/Gurobi), one per problem type (`quadratic.py`, `nonconvex.py`, `rosenbrock.py`), all subclassing `abcParamSolver` (`abc_solver.py`). This ABC wraps a Pyomo model with mutable parameters (`self.params`), decision variables (`self.vars`), and constraints (`self.cons`), and provides the operations used throughout the pipeline: `solve()`, `set_param_val()`, `relax()` (LP/NLP relaxation via `TransformationFactory("core.relax_integer_vars")`), `penalty()` (converts hard constraints to a soft-penalty objective via slacks), `first_solution_heuristic()`/`primal_heuristic()` (SCIP heuristic-only solves used as fast baselines), `set_warm_start()`, and violation metrics (`cal_violation()`). `int_ind`/`bin_ind` properties expose which variable indices are integer/binary — this is threaded into the neural rounding layers so they know which outputs need integer correction.

2. **`src/problem/neuromancer/`** — the same problems expressed as differentiable NeuroMANCER loss functions (`quadratic.py`, `nonconvex.py`, `rosenbrock.py`, imported as `nmQuadratic`/`nmNonconvex`/`nmRosenbrock` in `src/problem/__init__.py`), used to train the neural solution-mapping network end-to-end.

Both are aggregated in `src/problem/__init__.py` with matching `ms*`/`nm*` naming — when adding a new problem class, both a math-solver and a neuromancer counterpart are expected.

**Neural pipeline** (see `run/quadratic.py` for the canonical pattern, mirrored in `run/nonconvex.py`/`run/rosenbrock.py`):
- A **solution-mapping network** (`nm.system.Node` wrapping an MLP) maps problem parameters (e.g. `b`) directly to a continuous relaxation `x`.
- A **rounding/correction layer** (`src/func/rnd.py`, exposed via `src/func/__init__.py`) takes `x` and produces an integer-feasible `x_rnd`. Variants dispatched by method name in `run/*.py`:
  - `roundGumbelModel` → "RC" (Rounding Classification)
  - `roundThresholdModel` → "LT" (Learnable Thresholding)
  - `roundSTEModel` → straight-through-estimator rounding ("RS")
  - no rounding layer at all + `naive_round` heuristic (`src/heuristic/round.py`) → "RL" baseline
  - `src/func/ste.py` implements the underlying straight-through-estimator ops these layers rely on for backprop through the (non-differentiable) rounding operation.
- Both components are chained as `nn.ModuleList([smap, rnd])` and trained jointly via `src.problem.neuromancer.trainer.trainer` (wrapped by `run/utils.py::train`), against the matching `nm*` loss function (soft penalty on constraint violation, weighted by `--penalty`).
- At eval time, `run/*.py::evaluate()` optionally wraps inference with `src.postprocess.project.gradientProjection` (the `--project` flag) — a gradient-based iterative correction of `x_rnd` against the differentiable constraints, before handing values back to the math-solver model (`model.vars[...].value = ...`) to compute exact objective/violation via `cal_violation()`.

**Method-name conventions across `run/*.py`** (baselines vs. learned methods, mirrored for cq=quadratic, nc=nonconvex, rb=rosenbrock in result CSV prefixes):
- `exact` — full SCIP/Gurobi solve (ground-truth baseline, slow).
- `relRnd` — solve LP/NLP relaxation then naive rounding.
- `root` — solver's first-feasible-solution heuristic (`first_solution_heuristic`).
- `rndCls`/`rndThd`/`rndSte` — learned rounding layers (RC/LT/RS) + optional projection.
- `lrnRnd` — learned solution map only, rounded naively post-hoc (no learned rounding layer).
- `warmstart` (quadratic only) — feeds a previously-computed ML solution CSV into Gurobi as a MIP start and compares against a cold solve.

`src/heuristic/resolve.py` holds heuristics for re-solving/repairing a problem instance (distinct from the naive rounding in `round.py`).

`src/utlis/data.py` provides `data_split` (train/val/test split + a `torch.utils.data.Dataset`/`collate_fn` for the parameter dict) and `solve_test.py` has helpers for evaluating solved instances — used across all three `run_*.py` scripts.
