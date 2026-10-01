"""
step_0 / to-do 4: hook khong lam doi ket qua + file hooks day du.
Chay tu goc repo: python Claude_plan/mixed_equality_diagnosis/step_0/data/check_hooks.py
"""
import sys
sys.path.insert(0, ".")
import copy
import os
import tempfile
import numpy as np
import pandas as pd
import torch
from run.diag_eq import pipeline

cfg = copy.deepcopy(pipeline.DEFAULT_CFG)
cfg.update({"test_size": 50, "epochs": 6, "max_iters": 100})
ok = True
with tempfile.TemporaryDirectory() as d_off, tempfile.TemporaryDirectory() as d_on:
    pipeline.run_variant("V4", 0, cfg, d_off, do_project=True, save_hooks=False)
    pipeline.run_variant("V4", 0, cfg, d_on, do_project=True, save_hooks=True)
    for st in ["noproj", "proj"]:
        a = pd.read_csv(f"{d_off}/V4_seed0_{st}.csv")
        b = pd.read_csv(f"{d_on}/V4_seed0_{st}.csv")
        same = a.equals(b)
        print(f"{st}: CSV hooks off == hooks on: {same} (max abs diff {np.abs(a.values - b.values).max():.3e})")
        ok &= same
    print("hooks file in 'off' run exists:", os.path.exists(f"{d_off}/V4_seed0_hooks.pt"))
    ok &= not os.path.exists(f"{d_off}/V4_seed0_hooks.pt")
    h = torch.load(f"{d_on}/V4_seed0_hooks.pt", weights_only=False)
    for k, v in h.items():
        print(f"  {k}: {np.asarray(v).shape if not isinstance(v, list) else ('list', len(v))}")
    need = ["x_bar", "x_hat", "proj_latent", "proj_rounded", "proj_history"]
    ok &= all(k in h for k in need)
    T = len(h["proj_history"])
    ok &= h["proj_latent"].shape[0] == T and h["proj_rounded"].shape[0] == T
    print(f"projection iterations: history={T}, latent_states={h['proj_latent'].shape[0]}, "
          f"rounded_states={h['proj_rounded'].shape[0]}")
    # x_hat saved in hooks equals the rounded output at iteration 0 of the projection
    same0 = np.allclose(h["x_hat"], h["proj_rounded"][0], atol=1e-6)
    same_lat = np.allclose(h["x_bar"], h["proj_latent"][0], atol=1e-6)
    print(f"x_hat == proj_rounded[0]: {same0}; x_bar == proj_latent[0]: {same_lat}")
    ok &= same0 and same_lat
print("PASS" if ok else "FAIL")
