#!/usr/bin/env python
# coding: utf-8
"""
Submit a demo/validation experiment for MIRB + an equality constraint that
mixes continuous and integer variables:

    sum_i x_2i = sum_i x_2i+1

Results (CSVs) are saved under eq_results/, and demonstration figures under
eq_results/figures/. This is a small-scale run intended to validate the new
constraint end-to-end (exact SCIP solve + differentiable surrogate +
gradient projection), not to reproduce the paper's full benchmark sweep.
"""
import argparse
import math
import random

import numpy as np
import torch

# random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set parser
parser = argparse.ArgumentParser()
parser.add_argument("--size",
                    type=int,
                    default=3,
                    choices=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
                    help="number of blocks (problem size)")
parser.add_argument("--penalty",
                    type=float,
                    default=50,
                    help="penalty weight for the 4 original inequality constraints")
parser.add_argument("--penalty_eq",
                    type=float,
                    default=100,
                    help="penalty weight for the new equality constraint")
parser.add_argument("--project",
                    action="store_true",
                    help="apply gradient-based feasibility projection")
parser.add_argument("--eq_p_threshold",
                    type=float,
                    default=None,
                    help="analytic feasibility threshold p* to design the equality coefficient "
                         "around (eq_coef = 2/sqrt(p*)). Omit to reproduce case-1 (eq_coef=1.0).")
parser.add_argument("--out_dir",
                    type=str,
                    default="eq_results",
                    help="directory to save CSVs/figures/README.md into")
config = parser.parse_args()

# equality coefficient: sum(x_2i) = eq_coef * sum(x_2i+1)
# derived from the target analytic threshold p*, or 1.0 (case-1 default)
if config.eq_p_threshold is not None:
    config.eq_coef = 2 / math.sqrt(config.eq_p_threshold)
else:
    config.eq_coef = 1.0

# init problem (small-scale demo, not the paper's full benchmark)
config.steepness = 50               # steepness factor
num_blocks = config.size            # number of blocks
config.train_size = 2000            # number of train (reduced from 8000 for a fast demo run)
config.test_size = 30               # number of test instances (SCIP exact solve is the bottleneck)
config.val_size = 200               # number of validation

# hyperparameters (same lookup table as run_rb.py)
hsize_dict = {1: 4, 3: 8, 10: 16, 30: 32, 100: 64, 300: 128, 1000: 256, 3000: 512, 10000: 1024}
config.batch_size = 64
config.hlayers_sol = 5
config.hlayers_rnd = 4
config.hsize = hsize_dict[config.size]
config.lr = 1e-3

# parameters as input data
p_low, p_high = 1.0, 8.0
a_low, a_high = 0.5, 4.5
p_train = np.random.uniform(p_low, p_high, (config.train_size, 1)).astype(np.float32)
p_test = np.random.uniform(p_low, p_high, (config.test_size, 1)).astype(np.float32)
p_val = np.random.uniform(p_low, p_high, (config.val_size, 1)).astype(np.float32)
a_train = np.random.uniform(a_low, a_high, (config.train_size, num_blocks)).astype(np.float32)
a_test = np.random.uniform(a_low, a_high, (config.test_size, num_blocks)).astype(np.float32)
a_val = np.random.uniform(a_low, a_high, (config.val_size, num_blocks)).astype(np.float32)

from neuromancer.dataset import DictDataset
data_train = DictDataset({"p": p_train, "a": a_train}, name="train")
data_test = DictDataset({"p": p_test, "a": a_test}, name="test")
data_val = DictDataset({"p": p_val, "a": a_val}, name="dev")

from torch.utils.data import DataLoader
loader_train = DataLoader(data_train, config.batch_size, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
loader_test = DataLoader(data_test, config.batch_size, num_workers=0, collate_fn=data_test.collate_fn, shuffle=False)
loader_val = DataLoader(data_val, config.batch_size, num_workers=0, collate_fn=data_val.collate_fn, shuffle=True)

import run
print("Rosenbrock + equality constraint (demo/validation run)")
print(config)

df_exact = run.rosenbrock_eq.exact(loader_test, config, result_dir=config.out_dir)
df_noproj, df_proj, proj_history = run.rosenbrock_eq.rndCls(loader_train, loader_test, loader_val, config, result_dir=config.out_dir)
run.rosenbrock_eq.make_figures(df_exact, df_noproj, df_proj, proj_history, num_blocks, result_dir=config.out_dir)

region_stats = None
if config.eq_p_threshold is not None:
    region_stats = run.rosenbrock_eq.analyze_regions(df_exact, df_noproj, df_proj, p_test,
                                                      config.eq_p_threshold, num_blocks,
                                                      result_dir=config.out_dir)

run.rosenbrock_eq.write_readme(config, df_exact, df_noproj, df_proj, result_dir=config.out_dir, region_stats=region_stats)
print(f"Done. See {config.out_dir}/ (CSVs) and {config.out_dir}/figures/ (plots).")
