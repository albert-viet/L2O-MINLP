"""
step_0 / to-do 3: so metrics.py (vectorized, float64) voi model.cal_violation() cua Pyomo
tren nghiem that cua network (khong projection va co projection), bien the V4 va V2.
Chay tu goc repo: python Claude_plan/mixed_equality_diagnosis/step_0/data/check_metrics.py
"""
import sys
sys.path.insert(0, ".")
import copy
import numpy as np
from run.diag_eq import pipeline, metrics
from run.diag_eq.variants import build_variant, eq_coef_from_pstar

cfg = copy.deepcopy(pipeline.DEFAULT_CFG)
cfg.update({"test_size": 20, "epochs": 6})
c = eq_coef_from_pstar(cfg["p_star"])
K = cfg["num_blocks"]
worst = 0.0
for name in ["V4", "V2"]:
    seed = 0
    pipeline.set_seed(seed)
    data = pipeline.make_data(seed, cfg)
    v = build_variant(name, K, cfg["steepness"], c, cfg["penalty"], cfg["penalty_eq"])
    pipeline.set_seed(seed)
    comps = pipeline.build_components(v.model, cfg)
    from run import utils
    utils.train(comps, v.loss_fn, data["loaders"]["train"], data["loaders"]["val"], cfg["lr"], False, epochs=cfg["epochs"])
    p, a = data["p_test"], data["a_test"]
    stage = pipeline.forward_stages(comps, p, a)
    xs = {"noproj": stage["x_rnd"].cpu().numpy()}
    cfg_p = dict(cfg, max_iters=50)
    pipeline.project(comps, v.loss_fn, stage, cfg_p)
    xs["proj"] = stage["x_rnd"].detach().cpu().numpy()
    for st, x in xs.items():
        df = metrics.per_instance_metrics(x, p, a, K, cfg["steepness"], c, v.eq_mode)
        diffs = []
        for i in range(len(x)):
            v.model.set_param_val({"p": p[i], "a": a[i]})
            for j in range(2 * K):
                v.model.vars["x"][j].value = float(x[i, j])
            _, obj = v.model.get_val()
            viol = v.model.cal_violation()          # inner, outer, linear_b, linear_q, [eq]
            # pyomo returns 0 when the raw violation is <= 1e-5
            def thr(raw):
                return raw if raw > 1e-5 else 0.0
            b, q = metrics.coefficients(K)
            lin_b, lin_q = max(x[i, 0::2] @ b, 0.0), max(x[i, 1::2] @ q, 0.0)
            ref = [viol[0], viol[1], viol[2], viol[3]] + ([viol[4]] if len(viol) > 4 else [0.0])
            mine = [thr(df.inner[i]), thr(df.outer[i]), thr(lin_b), thr(lin_q), thr(df.abs_h[i])]
            diffs += [abs(r - m) for r, m in zip(ref, mine)] + [abs(obj - df.obj[i]) / max(1.0, abs(obj))]
        d = max(diffs)
        worst = max(worst, d)
        print(f"{name} {st}: {len(x)} instances, max |pyomo - metrics| = {d:.3e}; "
              f"mean v_ineq={df.v_ineq.mean():.4f} mean |h|={df.abs_h.mean():.4f}")
print(f"WORST DIFF = {worst:.3e}  ->  {'PASS' if worst < 1e-5 else 'FAIL'}")
