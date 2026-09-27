#!/usr/bin/env python
# coding: utf-8
"""
Reproduce the paper's img/ figure styles (see img/example*.png,
img/method_RC.png, img/cq_s100_penalty.png, img/rb_s100_penalty.png) for the
MIRB+equality experiments saved under eq_results/<case>/.

This module is intentionally separate from run/rosenbrock_eq.py (which
produces the primary CSVs/README used for the scientific conclusions) - it
only re-trains small extra models purely for illustration/plotting purposes.
"""
import os
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

import neuromancer as nm
from neuromancer.dataset import DictDataset
from torch.utils.data import DataLoader

from run import utils
from run.rosenbrock_eq import evaluate_eq, BREAKDOWN_COLS

import logging
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

# same data-generation recipe as run_rb_eq.py, so p_test/a_test line up
# exactly with the already-saved rb_eq_exact_3.csv rows
K = 3
STEEPNESS = 50
TRAIN_SIZE, TEST_SIZE, VAL_SIZE = 2000, 30, 200
HSIZE, HLAYERS_SOL, HLAYERS_RND, LR, BATCH_SIZE = 8, 5, 4, 1e-3, 64
P_LOW, P_HIGH, A_LOW, A_HIGH = 1.0, 8.0, 0.5, 4.5


def build_data():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    p_train = np.random.uniform(P_LOW, P_HIGH, (TRAIN_SIZE, 1)).astype(np.float32)
    p_test = np.random.uniform(P_LOW, P_HIGH, (TEST_SIZE, 1)).astype(np.float32)
    p_val = np.random.uniform(P_LOW, P_HIGH, (VAL_SIZE, 1)).astype(np.float32)
    a_train = np.random.uniform(A_LOW, A_HIGH, (TRAIN_SIZE, K)).astype(np.float32)
    a_test = np.random.uniform(A_LOW, A_HIGH, (TEST_SIZE, K)).astype(np.float32)
    a_val = np.random.uniform(A_LOW, A_HIGH, (VAL_SIZE, K)).astype(np.float32)
    return p_train, a_train, p_test, a_test, p_val, a_val


def make_loaders(p_train, a_train, p_test, a_test, p_val, a_val, n_test=None):
    if n_test is not None:
        p_test, a_test = p_test[:n_test], a_test[:n_test]
    data_train = DictDataset({"p": p_train, "a": a_train}, name="train")
    data_test = DictDataset({"p": p_test, "a": a_test}, name="test")
    data_val = DictDataset({"p": p_val, "a": a_val}, name="dev")
    loader_train = DataLoader(data_train, BATCH_SIZE, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
    loader_test = DataLoader(data_test, BATCH_SIZE, num_workers=0, collate_fn=data_test.collate_fn, shuffle=False)
    loader_val = DataLoader(data_val, BATCH_SIZE, num_workers=0, collate_fn=data_val.collate_fn, shuffle=True)
    return loader_train, loader_test, loader_val


def build_model_and_components(eq_coef, penalty, penalty_eq):
    from src.problem import msRosenbrockEq, nmRosenbrockEq
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    model = msRosenbrockEq(STEEPNESS, K, timelimit=60, eq_coef=eq_coef)
    func = nm.modules.blocks.MLP(insize=K+1, outsize=2*K, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[HSIZE]*HLAYERS_SOL)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    layers_rnd = netFC(input_dim=3*K+1, hidden_dims=[HSIZE]*HLAYERS_RND, output_dim=2*K)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["p", "a"], var_keys=["x"], output_keys=["x_rnd"],
                           int_ind=model.int_ind, continuous_update=True, name="round")
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmRosenbrockEq(["p", "a", "x_rnd"], STEEPNESS, K, penalty, penalty_eq, eq_coef=eq_coef)
    return model, components, loss_fn


def sweep_penalty(case_dir, eq_coef, penalty_eq_values, penalty=50.0,
                  normalize_by_group=False, n_test_sweep=12):
    """
    Train+evaluate (no-project and with-project) across several penalty_eq
    values, reusing the already-solved Exact-SCIP CSV in case_dir for the EX
    reference (restricted to the same n_test_sweep instances for a fair
    comparison). Returns a dict ready for plot_penalty_sensitivity().
    """
    p_train, a_train, p_test, a_test, p_val, a_val = build_data()
    loader_train, loader_test, loader_val = make_loaders(
        p_train, a_train, p_test, a_test, p_val, a_val, n_test=n_test_sweep)

    df_exact_full = pd.read_csv(os.path.join(case_dir, "rb_eq_exact_3.csv"))
    df_exact = df_exact_full.iloc[:n_test_sweep]
    ex_feas = float(np.mean(df_exact["Num Violations"] == 0)) * 100
    ex_obj = df_exact["Obj Val"].dropna().values

    results = {"penalty_eq": list(penalty_eq_values), "ex_feasibility": ex_feas, "ex_obj": ex_obj,
              "rc_feasibility": [], "rcp_feasibility": [], "rc_obj": [], "rcp_obj": []}

    for pe in penalty_eq_values:
        print(f"[sweep] penalty_eq={pe}")
        model, components, loss_fn = build_model_and_components(eq_coef, penalty, pe)
        utils.train(components, loss_fn, loader_train, loader_val, LR, penalty_growth=False)

        df_noproj, _ = evaluate_eq(components, loss_fn, model, loader_test, project=False)
        rc_feas = float(np.mean(df_noproj["Num Violations"] == 0)) * 100
        results["rc_feasibility"].append(rc_feas)
        results["rc_obj"].append(df_noproj["Obj Val"].dropna().values)

        df_proj, _ = evaluate_eq(components, loss_fn, model, loader_test, project=True,
                                 normalize_by_group=normalize_by_group)
        rcp_feas = float(np.mean(df_proj["Num Violations"] == 0)) * 100
        results["rcp_feasibility"].append(rcp_feas)
        results["rcp_obj"].append(df_proj["Obj Val"].dropna().values)

    return results


def plot_penalty_sensitivity(results, title, out_path):
    """
    Reproduce img/cq_s100_penalty.png / img/rb_s100_penalty.png style:
    top = % feasibility vs. penalty weight (log-x), one line per method;
    bottom = objective value boxplots grouped by penalty weight, with a
    separate Exact-SCIP reference group on the left.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pen = results["penalty_eq"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

    ax1.plot(pen, results["rc_feasibility"], "o-", color="#6fa8dc", label="RC (no projection)")
    ax1.plot(pen, results["rcp_feasibility"], "D--", color="#1c3f6e", label="RC-P (with projection)")
    ax1.axhline(results["ex_feasibility"], color="#e1a721", linestyle=":", label="Exact-SCIP (reference)")
    ax1.set_xscale("log")
    ax1.set_ylabel("% Feasibility")
    ax1.set_ylim(-5, 105)
    ax1.set_title(title)
    ax1.legend()

    # bottom: boxplots, one group per penalty_eq value, plus an EX group at the left
    box_data = [results["ex_obj"]] + results["rc_obj"] + results["rcp_obj"]
    n = len(pen)
    positions = [0] + list(range(2, 2+2*n, 2)) + list(range(3, 3+2*n, 2))
    colors = ["#e1a721"] + ["#6fa8dc"]*n + ["#1c3f6e"]*n
    bp = ax2.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    for i in range(n):
        ax2.axvline(2*i + 1.5, color="gray", linestyle="--", linewidth=0.7)
    xt = [0] + [p for pair in zip(pen, pen) for p in pair]
    xt_pos = [0] + list(range(2, 2+2*n))
    ax2.set_xticks([0] + [2*i+2.5 for i in range(n)])
    ax2.set_xticklabels(["EX"] + [str(p) for p in pen])
    ax2.set_xlabel("penalty_eq (RC=light blue, RC-P=dark blue)")
    ax2.set_ylabel("Objective Value")
    ax2.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
