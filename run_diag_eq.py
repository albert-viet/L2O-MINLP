#!/usr/bin/env python
# coding: utf-8
"""
Mixed-equality diagnosis harness: run the 4 MIRB variants (V1-V4, see
run/diag_eq/variants.py) with identical seed/split/config and the same metrics.

Run inside the conda env:  export PATH=/home/viethq/miniconda3/envs/l2o-minlp/bin:$PATH
    python run_diag_eq.py --variants V1 V2 V3 V4 --seeds 0 1 2 --project --out_dir <dir>
"""
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--variants", nargs="+", default=["V1", "V2", "V3", "V4"],
                    choices=["V1", "V2", "V3", "V4"])
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--size", type=int, default=3, help="number of blocks K")
parser.add_argument("--n_test", type=int, default=200)
parser.add_argument("--train_size", type=int, default=2000)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--penalty", type=float, default=50.0)
parser.add_argument("--penalty_eq", type=float, default=100.0)
parser.add_argument("--p_star", type=float, default=4.5)
parser.add_argument("--feasible_only", action="store_true", help="sample p in [1, p*) only")
parser.add_argument("--project", action="store_true", help="also evaluate with gradient projection")
parser.add_argument("--save_hooks", action="store_true", help="save x_bar, x_hat and projection states")
parser.add_argument("--out_dir", type=str, required=True)
config = parser.parse_args()

from run.diag_eq import pipeline

cfg = copy.deepcopy(pipeline.DEFAULT_CFG)
cfg.update({"num_blocks": config.size, "test_size": config.n_test, "train_size": config.train_size,
            "epochs": config.epochs, "penalty": config.penalty, "penalty_eq": config.penalty_eq,
            "p_star": config.p_star})
if config.feasible_only:
    cfg["p_range"] = (1.0, config.p_star)
print(cfg)

rows = []
for seed in config.seeds:
    for name in config.variants:
        print(f"===== {name} seed {seed} =====")
        rows += pipeline.run_variant(name, seed, cfg, config.out_dir, do_project=config.project,
                                     save_hooks=config.save_hooks)
df, agg = pipeline.aggregate(rows, config.out_dir)
print(df.to_string())
print(agg.to_string())
