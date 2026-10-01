"""
step_1: bang so sanh 4 bien the (mean +- std qua seed) + hinh.
Chay tu goc repo: python Claude_plan/mixed_equality_diagnosis/step_1/data/make_comparison.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP = "Claude_plan/mixed_equality_diagnosis/step_1"
per_seed = pd.read_csv(f"{STEP}/data/runs/summary_per_seed.csv")

LABEL = {"V1": "V1\nno eq", "V2": "V2\ncont-only eq", "V3": "V3\nmixed eq,\nint relaxed", "V4": "V4\nmixed eq\n+ RC"}
STAGES = {"noproj": "Without projection", "proj": "With gradient projection"}
METRICS = [("abs_h", "|h|  (equality residual)", "linear"),
           ("v_ineq", "V_ineq  (inequality violation)", "linear"),
           ("obj", "Objective", "log"),
           ("F_0.1", "F(0.1)  (share within tol 0.1)", "linear")]

# ---- table (markdown + csv)
rows = []
for st in STAGES:
    for v in LABEL:
        d = per_seed[(per_seed.variant == v) & (per_seed.stage == st)]
        r = {"stage": st, "variant": v, "n_seeds": len(d)}
        for m in ["abs_h", "v_ineq", "inner", "linear", "obj", "feas_rate", "F_0.1", "F_0.01", "F_0.001"]:
            r[m] = f"{d[m].mean():.4g} ± {d[m].std():.2g}"
        rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(f"{STEP}/data/comparison_table.csv", index=False)
with open(f"{STEP}/data/comparison_table.md", "w") as f:
    f.write("# step_1 - bang so sanh 4 bien the (mean ± std qua 5 seed, 200 test instance/seed)\n\n")
    f.write("feas_rate: ty le instance co |h|<=1e-6 va V_ineq<=1e-6. F(eps): |h|<=eps va V_ineq<=eps.\n")
    f.write("V1 khong co equality nen |h| = 0 theo dinh nghia.\n\n")
    for st, name in STAGES.items():
        f.write(f"## {name}\n\n")
        f.write(tab[tab.stage == st].drop(columns="stage").to_markdown(index=False))
        f.write("\n\n")

# ---- figure
COL = {"V1": "#2a78d6", "V2": "#eb6834", "V3": "#1baf7a", "V4": "#eda100"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), facecolor=SURF)
rng = np.random.RandomState(0)
for i, st in enumerate(STAGES):
    for j, (m, title, scale) in enumerate(METRICS):
        ax = axes[i, j]
        ax.set_facecolor(SURF)
        for k, v in enumerate(LABEL):
            vals = per_seed[(per_seed.variant == v) & (per_seed.stage == st)][m].values
            x = k + rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(x, vals, s=34, color=COL[v], edgecolor=SURF, linewidth=1.2, zorder=3)
            mu = vals.mean()
            ax.hlines(mu, k - 0.28, k + 0.28, color=COL[v], linewidth=2, zorder=2)
            txt = f"{mu:.3g}"
            if m == "abs_h" and v == "V1":
                txt = "0 (n/a)"
            ax.annotate(txt, (k, max(vals.max(), mu)), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.set_yscale(scale)
        if scale == "linear":
            top = ax.get_ylim()[1]
            ax.set_ylim(-0.02 * top, top * 1.12)
        else:
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 2.2)
        ax.set_xticks(range(4))
        ax.set_xticklabels([LABEL[v] for v in LABEL], fontsize=7.5)
        ax.set_xlim(-0.6, 3.6)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        if i == 0:
            ax.set_title(title, loc="left", fontsize=10, color=INK)
        if j == 0:
            ax.set_ylabel(STAGES[st], fontsize=9.5, color=INK)
fig.suptitle("step_1: 4 variants, mixed p~U(1,8) dataset, K=3, 5 seeds (dots = seeds, bar = mean)",
             x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{STEP}/figures/variant_comparison.png", dpi=150, facecolor=SURF)
print("saved")
