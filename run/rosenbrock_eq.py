#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline for MIRB + equality constraint (sum_i x_2i = sum_i x_2i+1)

This mirrors run/rosenbrock.py's style but is scoped down to a single
learned rounding method (RC / Gumbel) plus the exact SCIP baseline, since
its purpose is to validate/demonstrate the new equality constraint rather
than to reproduce the paper's full benchmark sweep.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from run import utils

# turn off warning
import logging
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

# default output directory (overridable via the `result_dir` parameter below,
# e.g. to keep the case-1 vs. case-2 experiments in separate subfolders)
RESULT_DIR = "eq_results"

# order in which msRosenbrockEq.cal_violation() reports constraints:
# [inner, outer, linear (b), linear (q), equality]
BREAKDOWN_COLS = ["Inner Viol", "Outer Viol", "Linear Viol", "Equality Viol"]


def exact(loader_test, config, result_dir=RESULT_DIR):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"EX(eq) in RB for size {config.size}, eq_coef={config.eq_coef}.")
    # config
    steepness = config.steepness
    num_blocks = config.size
    # init model
    from src.problem import msRosenbrockEq
    model = msRosenbrockEq(steepness, num_blocks, timelimit=60, eq_coef=config.eq_coef)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    inner_viols, outer_viols, linear_viols, eq_viols = [], [], [], []
    # go through test data
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # set params
        model.set_param_val({"p": p, "a": a})
        # solve
        tick = time.time()
        params.append(list(p) + list(a))
        try:
            xval, objval = model.solve()
            tock = time.time()
            # eval
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            viol = model.cal_violation()
            mean_viols.append(np.mean(viol))
            max_viols.append(np.max(viol))
            num_viols.append(np.sum(viol > 1e-6))
            inner_viols.append(viol[0])
            outer_viols.append(viol[1])
            linear_viols.append(viol[2] + viol[3])
            eq_viols.append(viol[4])
        except Exception:
            # infeasible / unsolved
            sols.append(None)
            objvals.append(None)
            mean_viols.append(None)
            max_viols.append(None)
            num_viols.append(None)
            inner_viols.append(None)
            outer_viols.append(None)
            linear_viols.append(None)
            eq_viols.append(None)
            tock = time.time()
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds,
                       "Inner Viol": inner_viols,
                       "Outer Viol": outer_viols,
                       "Linear Viol": linear_viols,
                       "Equality Viol": eq_viols})
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    os.makedirs(result_dir, exist_ok=True)
    df.to_csv(f"{result_dir}/rb_eq_exact_{num_blocks}.csv")
    return df


def rndCls(loader_train, loader_test, loader_val, config, result_dir=RESULT_DIR):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RC(eq) in RB for size {config.size}, eq_coef={config.eq_coef}.")
    import neuromancer as nm
    from src.problem import nmRosenbrockEq, msRosenbrockEq
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    # config
    steepness = config.steepness
    num_blocks = config.size
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    penalty_weight_eq = config.penalty_eq
    project = config.project
    # init model
    model = msRosenbrockEq(steepness, num_blocks, timelimit=60, eq_coef=config.eq_coef)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=3*num_blocks+1, hidden_dims=[hsize]*hlayers_rnd, output_dim=2*num_blocks)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["p", "a"], var_keys=["x"], output_keys=["x_rnd"],
                           int_ind=model.int_ind, continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmRosenbrockEq(["p", "a", "x_rnd"], steepness, num_blocks, penalty_weight,
                             penalty_weight_eq, eq_coef=config.eq_coef)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth=False)
    # eval without projection
    df_noproj, _ = evaluate_eq(components, loss_fn, model, loader_test, project=False, record_history=False)
    os.makedirs(result_dir, exist_ok=True)
    df_noproj.to_csv(f"{result_dir}/rb_eq_cls_noproj_{num_blocks}.csv")
    # eval with projection (if requested)
    df_proj, proj_history = None, None
    if project:
        df_proj, proj_history = evaluate_eq(components, loss_fn, model, loader_test, project=True, record_history=True,
                                            normalize_by_group=getattr(config, "normalize_projection", False))
        df_proj.to_csv(f"{result_dir}/rb_eq_cls_proj_{num_blocks}.csv")
    return df_noproj, df_proj, proj_history


def evaluate_eq(components, loss_fn, model, loader_test, project, record_history=False, normalize_by_group=False):
    """
    Same evaluation loop as run/rosenbrock.py::evaluate, but also logs the
    per-constraint-type violation breakdown (inner/outer/linear/equality)
    from the differentiable surrogate, and optionally records the
    gradient-projection convergence history for the first test instance.
    """
    proj = None
    if project:
        from src.postprocess.project import gradientProjection
        proj = gradientProjection([components[0]], [components[1]], loss_fn, "x",
                                  record_history=record_history, normalize_by_group=normalize_by_group)
    components.eval()
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    inner_viols, outer_viols, linear_viols, eq_viols = [], [], [], []
    proj_history = None
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for idx, (p, a) in enumerate(tqdm(list(zip(p_test, a_test)))):
        # data point as tensor
        datapoints = {"p": torch.tensor(np.array([p]), dtype=torch.float32).to("cuda"),
                      "a": torch.tensor(np.array([a]), dtype=torch.float32).to("cuda"),
                      "name": "test"}
        # infer
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        if project:
            proj(datapoints)
            if record_history and idx == 0:
                proj_history = list(proj.history)
        tock = time.time()
        # per-constraint-type violation breakdown from the surrogate
        with torch.no_grad():
            bd = loss_fn.cal_violation_breakdown(datapoints)
        inner_viols.append(bd["inner"].item())
        outer_viols.append(bd["outer"].item())
        linear_viols.append(bd["linear"].item())
        eq_viols.append(bd["equality"].item())
        # assign params
        model.set_param_val({"p": p, "a": a})
        # assign vars
        x = datapoints["x_rnd"]
        for i in range(len(model.vars["x"])):
            model.vars["x"][i].value = x[0, i].item()
        # get solutions
        xval, objval = model.get_val()
        params.append(list(p) + list(a))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds,
                       "Inner Viol": inner_viols,
                       "Outer Viol": outer_viols,
                       "Linear Viol": linear_viols,
                       "Equality Viol": eq_viols})
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    return df, proj_history


def make_figures(df_exact, df_noproj, df_proj, proj_history, num_blocks, result_dir=RESULT_DIR):
    """
    Save demonstration figures into <result_dir>/figures/, built from the
    actual dataframes produced by exact()/rndCls() above.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR = os.path.join(result_dir, "figures")
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # ---- Figure 1: mean violation breakdown, grouped bar chart ----
    groups = {}
    if df_exact is not None:
        groups["Exact-SCIP"] = [df_exact[c].dropna().mean() for c in BREAKDOWN_COLS]
    if df_noproj is not None:
        groups["Learned-NoProject"] = [df_noproj[c].dropna().mean() for c in BREAKDOWN_COLS]
    if df_proj is not None:
        groups["Learned-WithProject"] = [df_proj[c].dropna().mean() for c in BREAKDOWN_COLS]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(BREAKDOWN_COLS))
    width = 0.8 / max(len(groups), 1)
    for i, (name, vals) in enumerate(groups.items()):
        ax.bar(x + i*width, vals, width=width, label=name)
    ax.set_xticks(x + width*(len(groups)-1)/2)
    ax.set_xticklabels(["Inner", "Outer", "Linear", "Equality"])
    ax.set_ylabel("Mean constraint violation")
    ax.set_title(f"MIRB+equality (K={num_blocks}): violation breakdown by constraint type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "violation_breakdown.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 2: histogram of equality violation, no-project vs with-project ----
    fig, ax = plt.subplots(figsize=(7, 5))
    if df_noproj is not None:
        ax.hist(df_noproj["Equality Viol"].dropna(), bins=20, alpha=0.6, label="No projection")
    if df_proj is not None:
        ax.hist(df_proj["Equality Viol"].dropna(), bins=20, alpha=0.6, label="With projection")
    ax.set_xlabel("Equality constraint violation (per test instance)")
    ax.set_ylabel("Count")
    ax.set_title(f"MIRB+equality (K={num_blocks}): distribution of equality violation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "equality_violation_hist.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 3: objective vs. equality violation scatter ----
    if df_noproj is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        valid = df_noproj.dropna(subset=["Obj Val", "Equality Viol"])
        ax.scatter(valid["Equality Viol"], valid["Obj Val"], alpha=0.7, label="No projection")
        if df_proj is not None:
            valid_p = df_proj.dropna(subset=["Obj Val", "Equality Viol"])
            ax.scatter(valid_p["Equality Viol"], valid_p["Obj Val"], alpha=0.7, label="With projection")
        ax.set_xlabel("Equality constraint violation")
        ax.set_ylabel("Objective value")
        ax.set_title(f"MIRB+equality (K={num_blocks}): objective vs. equality violation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, "objective_vs_eqviolation.png"), dpi=150)
        plt.close(fig)

    # ---- Figure 4: gradient-projection convergence (1 representative instance) ----
    if proj_history:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(len(proj_history)), proj_history, marker=".")
        ax.set_yscale("log")
        ax.set_xlabel("Projection iteration")
        ax.set_ylabel("Max total violation (log scale)")
        ax.set_title(f"MIRB+equality (K={num_blocks}): gradient-projection convergence\n(1 representative test instance)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, "projection_convergence.png"), dpi=150)
        plt.close(fig)


def analyze_regions(df_exact, df_noproj, df_proj, p_test, p_threshold, num_blocks, result_dir=RESULT_DIR):
    """
    Split each result dataframe into a "feasible-region" (p < p_threshold)
    and "infeasible-region" (p >= p_threshold) group, based on the analytic
    Cauchy-Schwarz threshold derived from combining the equality constraint
    with the pre-existing inner/outer constraints. Row i of every dataframe
    corresponds to p_test[i] (same fixed order, loader_test uses shuffle=False).

    Saves `region_split_violation.png` and returns a stats dict used by
    write_readme() to report real, computed numbers (not fabricated).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_flat = np.asarray(p_test).reshape(-1)
    region_mask = p_flat < p_threshold  # True = analytically-feasible region

    stats = {"p_threshold": p_threshold, "n_total": len(p_flat),
             "n_feasible_region": int(region_mask.sum()),
             "n_infeasible_region": int((~region_mask).sum()),
             "sources": {}}

    sources = [("Exact-SCIP", df_exact), ("Learned-NoProject", df_noproj), ("Learned-WithProject", df_proj)]
    sources = [(name, df) for name, df in sources if df is not None]

    fig, axes = plt.subplots(1, len(sources), figsize=(6*len(sources), 5), squeeze=False)
    axes = axes[0]
    x = np.arange(len(BREAKDOWN_COLS))
    width = 0.35

    for ax, (name, df) in zip(axes, sources):
        mask = region_mask[:len(df)]
        df_feas = df[mask]
        df_infeas = df[~mask]
        means_feas = [df_feas[c].dropna().mean() for c in BREAKDOWN_COLS]
        means_infeas = [df_infeas[c].dropna().mean() for c in BREAKDOWN_COLS]
        ax.bar(x - width/2, means_feas, width=width, label=f"p<{p_threshold:g}")
        ax.bar(x + width/2, means_infeas, width=width, label=f"p>={p_threshold:g}")
        ax.set_xticks(x)
        ax.set_xticklabels(["Inner", "Outer", "Linear", "Equality"])
        ax.set_title(name)
        ax.set_ylabel("Mean constraint violation")
        ax.legend()

        source_stats = {
            "n_feasible_region": int(mask.sum()),
            "n_infeasible_region": int((~mask).sum()),
            "means_feasible_region": dict(zip(BREAKDOWN_COLS, means_feas)),
            "means_infeasible_region": dict(zip(BREAKDOWN_COLS, means_infeas)),
        }
        if name == "Exact-SCIP":
            source_stats["n_solved_feasible_region"] = int(df_feas["Sol"].notna().sum())
            source_stats["n_solved_infeasible_region"] = int(df_infeas["Sol"].notna().sum())
        stats["sources"][name] = source_stats

    fig.suptitle(f"MIRB+equality (K={num_blocks}): violation split by analytic region (p*={p_threshold:g})")
    fig.tight_layout()
    figure_dir = os.path.join(result_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    fig.savefig(os.path.join(figure_dir, "region_split_violation.png"), dpi=150)
    plt.close(fig)

    return stats


def write_readme(config, df_exact, df_noproj, df_proj, result_dir=RESULT_DIR, region_stats=None):
    """
    Auto-generate <result_dir>/README.md filled with the real numbers
    computed in this run (no hand-written / fabricated numbers).
    """
    lines = []
    lines.append("# MIRB + equality constraint — demo/validation results\n")
    lines.append("Equality constraint added to the Mixed-Integer Rosenbrock Problem (MIRB):\n")
    lines.append("    sum_i x_2i  =  eq_coef * sum_i x_2i+1     (i = 0..K-1)\n")
    lines.append("i.e. the total of all continuous block-variables equals `eq_coef` times the "
                "total of all integer block-variables. This mixes both variable types in a "
                "single global (aggregate) constraint, as opposed to a per-block match.\n")
    lines.append("Combining this equality with the pre-existing inner (`sum(x_2i+1) >= K*p/2`) "
                "and outer (`sum(x_2i^2) <= K*p`) constraints gives, via Cauchy-Schwarz, an "
                "analytic feasibility threshold `p* = (2/eq_coef)^2`: instances with `p > p*` "
                "are infeasible by construction, independent of `a`.\n")
    lines.append("## Configuration\n")
    lines.append(f"- num_blocks (K / --size): {config.size}")
    lines.append(f"- steepness: {config.steepness}")
    lines.append(f"- eq_coef: {config.eq_coef:.6g}  (analytic threshold p* = {(2/config.eq_coef)**2:.6g})")
    lines.append(f"- penalty (inequality): {config.penalty}")
    lines.append(f"- penalty_eq (equality): {config.penalty_eq}")
    lines.append(f"- project (gradient projection enabled): {config.project}")
    lines.append(f"- hsize / hlayers_sol / hlayers_rnd: {config.hsize} / {config.hlayers_sol} / {config.hlayers_rnd}")
    lines.append(f"- train/val/test size: {config.train_size} / {config.val_size} / {config.test_size}\n")
    lines.append("This is a small-scale demo/validation run (not the paper's full benchmark "
                "sweep) meant to check that the new equality constraint is well-posed for both "
                "the exact SCIP solver and the differentiable neuromancer surrogate.\n")

    lines.append("## Results (mean over test set)\n")
    lines.append("| Source | Inner | Outer | Linear | Equality | Obj Val (mean) |")
    lines.append("|---|---|---|---|---|---|")
    for name, df in [("Exact-SCIP", df_exact), ("Learned-NoProject", df_noproj), ("Learned-WithProject", df_proj)]:
        if df is None:
            continue
        row = [f"{df[c].dropna().mean():.6g}" for c in BREAKDOWN_COLS]
        obj_mean = df["Obj Val"].dropna().mean()
        lines.append(f"| {name} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {obj_mean:.6g} |")

    lines.append("\n## Figures (see figures/)\n")
    lines.append("- `violation_breakdown.png` — mean violation per constraint type, grouped by source.")
    lines.append("- `equality_violation_hist.png` — distribution of the equality-violation across test instances, with vs. without projection.")
    lines.append("- `objective_vs_eqviolation.png` — objective value vs. equality violation per test instance.")
    lines.append("- `projection_convergence.png` — gradient-projection convergence curve (1 representative instance), if `--project` was used.")

    if region_stats is not None:
        pth = region_stats["p_threshold"]
        lines.append("\n## Region split analysis (p < {:.4g} vs. p >= {:.4g})\n".format(pth, pth))
        lines.append(f"Analytic threshold `p* = {pth:.6g}` was chosen deliberately (not by "
                    "accident) so both regions are populated in the `p~U(1,8)` sampling range: "
                    f"{region_stats['n_feasible_region']}/{region_stats['n_total']} test "
                    f"instances have `p < p*` (analytically compatible with inner+outer), "
                    f"{region_stats['n_infeasible_region']}/{region_stats['n_total']} have "
                    "`p >= p*` (infeasible by construction, independent of `a`).\n")
        lines.append("Note: with only {} test instances total, each region has roughly {} "
                    "instances — small-sample noise should be expected in the per-region means "
                    "below.\n".format(region_stats["n_total"], region_stats["n_total"]//2))
        if "Exact-SCIP" in region_stats["sources"]:
            es = region_stats["sources"]["Exact-SCIP"]
            lines.append(f"- Exact-SCIP solved {es['n_solved_feasible_region']}/{es['n_feasible_region']} "
                        f"instances in the `p<p*` region, vs. {es['n_solved_infeasible_region']}/{es['n_infeasible_region']} "
                        "in the `p>=p*` region.\n")
        lines.append("| Source | Region | Inner | Outer | Linear | Equality |")
        lines.append("|---|---|---|---|---|---|")
        for name, s in region_stats["sources"].items():
            mf = s["means_feasible_region"]
            mi = s["means_infeasible_region"]
            lines.append(f"| {name} | p<{pth:g} | " + " | ".join(f"{mf[c]:.4g}" for c in BREAKDOWN_COLS) + " |")
            lines.append(f"| {name} | p>={pth:g} | " + " | ".join(f"{mi[c]:.4g}" for c in BREAKDOWN_COLS) + " |")
        lines.append("\n- `region_split_violation.png` — mean violation per constraint type, split by region, for each source.")

    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
