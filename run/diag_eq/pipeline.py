"""
Harness of the mixed-equality diagnosis study: train + evaluate one variant for
one seed, with optional batched gradient projection and pipeline hooks
(x_bar = latent output of the solution map, x_hat = rounded output, projection
states), and aggregation over seeds.
"""
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from run import utils
from run.diag_eq import metrics
from run.diag_eq.variants import build_variant, eq_coef_from_pstar

DEVICE = "cuda"

DEFAULT_CFG = {
    "num_blocks": 3, "steepness": 50,
    "p_range": (1.0, 8.0), "a_range": (0.5, 4.5),
    "p_star": 4.5,
    "train_size": 2000, "val_size": 200, "test_size": 200,
    "batch_size": 64, "hsize": 8, "hlayers_sol": 5, "hlayers_rnd": 4, "lr": 1e-3,
    "penalty": 50.0, "penalty_eq": 100.0,
    "epochs": 200,
    "max_iters": 1000, "step_size": 0.01,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_data(seed, cfg):
    """
    Same data for every variant given a seed (same sampling order as run_rb_eq.py).
    Use cfg["p_range"]=(1, p_star) for a feasible-only dataset.
    """
    from neuromancer.dataset import DictDataset
    from torch.utils.data import DataLoader
    rng = np.random.RandomState(seed)
    k = cfg["num_blocks"]
    (p_lo, p_hi), (a_lo, a_hi) = cfg["p_range"], cfg["a_range"]
    sizes = {"train": cfg["train_size"], "test": cfg["test_size"], "val": cfg["val_size"]}
    p = {s: rng.uniform(p_lo, p_hi, (n, 1)).astype(np.float32) for s, n in sizes.items()}
    a = {s: rng.uniform(a_lo, a_hi, (n, k)).astype(np.float32) for s, n in sizes.items()}
    names = {"train": "train", "test": "test", "val": "dev"}
    sets = {s: DictDataset({"p": p[s], "a": a[s]}, name=names[s]) for s in sizes}
    shuffle = {"train": True, "test": False, "val": True}
    loaders = {s: DataLoader(sets[s], cfg["batch_size"], num_workers=0,
                             collate_fn=sets[s].collate_fn, shuffle=shuffle[s]) for s in sizes}
    return {"p_test": p["test"], "a_test": a["test"], "loaders": loaders}


def build_components(model, cfg):
    """
    solution map + RC rounding layer (identical architecture for all variants)
    """
    import neuromancer as nm
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    k = cfg["num_blocks"]
    func = nm.modules.blocks.MLP(insize=k + 1, outsize=2 * k, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[cfg["hsize"]] * cfg["hlayers_sol"])
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    layers_rnd = netFC(input_dim=3 * k + 1, hidden_dims=[cfg["hsize"]] * cfg["hlayers_rnd"],
                       output_dim=2 * k)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["p", "a"], var_keys=["x"],
                           output_keys=["x_rnd"], int_ind=model.int_ind,
                           continuous_update=True, name="round")
    return nn.ModuleList([smap, rnd]).to(DEVICE)


def _batch(p, a):
    return {"p": torch.tensor(p, dtype=torch.float32, device=DEVICE),
            "a": torch.tensor(a, dtype=torch.float32, device=DEVICE),
            "name": "test"}


def forward_stages(components, p, a):
    """
    inference without projection: returns the data dict with x (x_bar) and x_rnd (x_hat)
    """
    components.eval()
    data = _batch(p, a)
    with torch.no_grad():
        for comp in components:
            data.update(comp(data))
    return data


def project(components, loss_fn, data, cfg, record_states=False):
    """
    gradient projection on a batch (same algorithm as src/postprocess/project.py)
    """
    from src.postprocess.project import gradientProjection
    proj = gradientProjection([components[0]], [components[1]], loss_fn, "x",
                              max_iters=cfg["max_iters"], step_size=cfg["step_size"],
                              record_history=True, record_states=record_states)
    components.eval()
    proj(data)
    return proj


def _frame(df_metrics, p, a):
    out = df_metrics.copy()
    out.insert(0, "p", np.asarray(p).reshape(-1))
    for i in range(a.shape[1]):
        out.insert(1 + i, f"a{i}", a[:, i])
    return out


def run_variant(name, seed, cfg, out_dir, do_project=False, save_hooks=False):
    """
    train + evaluate one variant for one seed. Writes, under out_dir:
      {name}_seed{seed}_noproj.csv / _proj.csv  per-instance metrics
      {name}_seed{seed}_hooks.pt                x_bar, x_hat, (projection states) if save_hooks
    Returns a list of summary rows (one per stage).
    """
    os.makedirs(out_dir, exist_ok=True)
    k = cfg["num_blocks"]
    eq_coef = eq_coef_from_pstar(cfg["p_star"])
    set_seed(seed)
    data = make_data(seed, cfg)
    v = build_variant(name, k, cfg["steepness"], eq_coef, cfg["penalty"], cfg["penalty_eq"])
    # same initialisation / data order for every variant of this seed
    set_seed(seed)
    components = build_components(v.model, cfg)
    tick = time.time()
    trn = utils.train(components, v.loss_fn, data["loaders"]["train"], data["loaders"]["val"],
                      cfg["lr"], penalty_growth=False, epochs=cfg["epochs"])
    train_time = time.time() - tick
    p_test, a_test = data["p_test"], data["a_test"]
    rows, hooks = [], {"p": p_test, "a": a_test}
    # ---- stage: no projection (x_bar -> x_hat)
    stage = forward_stages(components, p_test, a_test)
    x_bar, x_hat = stage["x"].cpu().numpy(), stage["x_rnd"].cpu().numpy()
    hooks.update({"x_bar": x_bar, "x_hat": x_hat})
    stages = {"noproj": x_hat}
    # ---- stage: projection
    if do_project:
        proj = project(components, v.loss_fn, stage, cfg, record_states=save_hooks)
        stages["proj"] = stage["x_rnd"].detach().cpu().numpy()
        hooks["proj_history"] = list(proj.history)
        if save_hooks:
            hooks["proj_latent"] = torch.stack(proj.latent_states).cpu().numpy()    # (T, N, 2K)
            hooks["proj_rounded"] = torch.stack(proj.rounded_states).cpu().numpy()  # (T, N, 2K)
    for st, x in stages.items():
        df = metrics.per_instance_metrics(x, p_test, a_test, k, cfg["steepness"], eq_coef, v.eq_mode)
        _frame(df, p_test, a_test).to_csv(f"{out_dir}/{name}_seed{seed}_{st}.csv", index=False)
        row = {"variant": name, "seed": seed, "stage": st,
               "train_iters": getattr(trn, "total_iters", None), "train_time": train_time}
        row.update(metrics.summarize(df))
        rows.append(row)
    if save_hooks:
        torch.save(hooks, f"{out_dir}/{name}_seed{seed}_hooks.pt")
    return rows


def aggregate(rows, out_dir):
    """
    per-seed summary + mean/std over seeds for each (variant, stage)
    """
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/summary_per_seed.csv", index=False)
    cols = [c for c in df.columns if c not in ("variant", "seed", "stage")]
    agg = df.groupby(["variant", "stage"])[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(f"{out_dir}/summary_mean_std.csv", index=False)
    return df, agg
