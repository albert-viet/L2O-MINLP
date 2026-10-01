"""
step_0 / to-do 6 (phan doi chieu): projection theo batch (200 instance) vs projection tung instance
(cach cu cua run/rosenbrock_eq.py::evaluate_eq) tren 5 instance dau, V4, cung model da train.
Chay tu goc repo: python Claude_plan/mixed_equality_diagnosis/step_0/data/check_projection.py
"""
import sys
sys.path.insert(0, ".")
import copy
import numpy as np
from run import utils
from run.diag_eq import pipeline, metrics
from run.diag_eq.variants import build_variant, eq_coef_from_pstar

cfg = copy.deepcopy(pipeline.DEFAULT_CFG)
cfg.update({"test_size": 200, "epochs": 10})          # max_iters = 1000 (default)
K, seed = cfg["num_blocks"], 0
c = eq_coef_from_pstar(cfg["p_star"])
pipeline.set_seed(seed)
data = pipeline.make_data(seed, cfg)
v = build_variant("V4", K, cfg["steepness"], c, cfg["penalty"], cfg["penalty_eq"])
pipeline.set_seed(seed)
comps = pipeline.build_components(v.model, cfg)
utils.train(comps, v.loss_fn, data["loaders"]["train"], data["loaders"]["val"], cfg["lr"], False, epochs=cfg["epochs"])
p, a = data["p_test"], data["a_test"]
# batched projection on all 200 instances
st = pipeline.forward_stages(comps, p, a)
pb = pipeline.project(comps, v.loss_fn, st, cfg)
x_batch = st["x_rnd"].detach().cpu().numpy()
n = 5
x_single, iters_single = [], []
for i in range(n):
    s1 = pipeline.forward_stages(comps, p[i:i + 1], a[i:i + 1])
    ps = pipeline.project(comps, v.loss_fn, s1, cfg)
    x_single.append(s1["x_rnd"].detach().cpu().numpy()[0])
    iters_single.append(len(ps.history))
x_single = np.array(x_single)
print(f"batched projection iterations: {len(pb.history)}; per-instance iterations: {iters_single}")
mb = metrics.per_instance_metrics(x_batch[:n], p[:n], a[:n], K, cfg["steepness"], c, "mixed")
ms = metrics.per_instance_metrics(x_single, p[:n], a[:n], K, cfg["steepness"], c, "mixed")
print("max |x_batch - x_single| per instance:", np.abs(x_batch[:n] - x_single).max(axis=1))
for col in ["obj", "v_ineq", "abs_h"]:
    print(f"{col}: batch={mb[col].round(5).tolist()} single={ms[col].round(5).tolist()}")
