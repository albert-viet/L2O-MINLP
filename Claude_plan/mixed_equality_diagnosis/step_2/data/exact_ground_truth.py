"""
step_2: SCIP ground truth cho tung test instance (4 bien the x 5 seed x {mixed, feasible_only}).
Test instance duoc sinh lai tu pipeline.make_data(seed, cfg) (cung du lieu voi cac run huan luyen).
Chay tu goc repo: python Claude_plan/mixed_equality_diagnosis/step_2/data/exact_ground_truth.py [--limit N]
Ghi: step_2/data/exact/{dataset}_{variant}_seed{seed}.csv  (status = termination condition cua SCIP)
"""
import sys
sys.path.insert(0, ".")
import argparse
import copy
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

OUT = "Claude_plan/mixed_equality_diagnosis/step_2/data/exact"
_model = {}


def _get_model(variant):
    if variant not in _model:
        from run.diag_eq.variants import VARIANTS
        from src.problem import msRosenbrockEq
        from run.diag_eq import pipeline
        from run.diag_eq.variants import eq_coef_from_pstar
        cfg = pipeline.DEFAULT_CFG
        spec = VARIANTS[variant]
        _model[variant] = msRosenbrockEq(cfg["steepness"], cfg["num_blocks"], timelimit=60,
                                         eq_coef=eq_coef_from_pstar(cfg["p_star"]),
                                         eq_mode=spec["eq_mode"], relax_int=spec["relax_int"])
    return _model[variant]


def solve_one(task):
    variant, p, a = task
    m = _get_model(variant)
    m.set_param_val({"p": p, "a": a})
    tick = time.time()
    try:
        xval, obj = m.solve()
        tc = str(m.res.solver.termination_condition)
        status = "solved" if obj is not None and tc == "optimal" else tc
        sol = list(xval["x"].values()) if xval else [None] * 6
    except Exception as e:                      # solver error
        status, obj, sol = "error:" + type(e).__name__, None, [None] * 6
    return status, obj, sol, time.time() - tick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N instances per (dataset, seed)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="+", default=["mixed", "feasible_only"])
    ap.add_argument("--variants", nargs="+", default=["V1", "V2", "V3", "V4"])
    ap.add_argument("--procs", type=int, default=24)
    args = ap.parse_args()
    from run.diag_eq import pipeline
    os.makedirs(OUT, exist_ok=True)
    with Pool(args.procs) as pool:
        for ds in args.datasets:
            cfg = copy.deepcopy(pipeline.DEFAULT_CFG)
            if ds == "feasible_only":
                cfg["p_range"] = (1.0, cfg["p_star"])
            for seed in args.seeds:
                data = pipeline.make_data(seed, cfg)
                p_all, a_all = data["p_test"], data["a_test"]
                n = args.limit or len(p_all)
                for v in args.variants:
                    tasks = [(v, p_all[i], a_all[i]) for i in range(n)]
                    tick = time.time()
                    res = pool.map(solve_one, tasks, chunksize=2)
                    df = pd.DataFrame({"idx": range(n), "p": p_all[:n, 0],
                                       **{f"a{j}": a_all[:n, j] for j in range(a_all.shape[1])},
                                       "status": [r[0] for r in res], "obj": [r[1] for r in res],
                                       "time": [r[3] for r in res]})
                    sol = np.array([[np.nan if s is None else s for s in r[2]] for r in res], dtype=float)
                    for j in range(sol.shape[1]):
                        df[f"x{j}"] = sol[:, j]
                    df.to_csv(f"{OUT}/{ds}_{v}_seed{seed}.csv", index=False)
                    print(f"{ds} {v} seed{seed}: {df.status.value_counts().to_dict()} "
                          f"max_time={df.time.max():.1f}s wall={time.time() - tick:.0f}s", flush=True)


if __name__ == "__main__":
    main()
