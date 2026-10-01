"""
step_2: hinh so sanh model huan luyen tren mixed vs feasible-only, tren tap con instance CHAC CHAN co nghiem.
Chay tu goc repo: python Claude_plan/mixed_equality_diagnosis/step_2/data/make_figure.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S2 = "Claude_plan/mixed_equality_diagnosis/step_2"
t = pd.read_csv(f"{S2}/data/certain_feasible_subset.csv")
t = t[t.subset == "feasible"]
LAB = {"V1": "V1\nno eq", "V2": "V2\ncont-only eq", "V3": "V3\nmixed eq,\nint relaxed", "V4": "V4\nmixed eq\n+ RC"}
STAGES = {"noproj": "Without projection", "proj": "With gradient projection"}
METRICS = [("abs_h", "|h|  (equality residual)"), ("v_ineq", "V_ineq  (inequality violation)"),
           ("F_0.1", "F(0.1)  (share within tol 0.1)")]
COL = {"mixed": "#2a78d6", "feasible_only": "#eb6834"}
NAME = {"mixed": "trained on mixed p~U(1,8)", "feasible_only": "trained on feasible-only p<p*"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), facecolor=SURF)
w = 0.36
for i, st in enumerate(STAGES):
    for j, (m, title) in enumerate(METRICS):
        ax = axes[i, j]
        ax.set_facecolor(SURF)
        for k, v in enumerate(LAB):
            for o, ds in enumerate(["mixed", "feasible_only"]):
                r = t[(t.variant == v) & (t.stage == st) & (t.trained_on == ds)].iloc[0]
                x = k + (o - 0.5) * w * 1.1
                ax.bar(x, r[m], width=w, color=COL[ds], zorder=2, label=NAME[ds] if (k == 0 and i == 0 and j == 0) else None)
                ax.errorbar(x, r[m], yerr=r[m + "_std"], color=INK2, linewidth=1, capsize=2, zorder=3)
                txt = "0" if (m == "abs_h" and v == "V1") else f"{r[m]:.3g}"
                ax.annotate(txt, (x, r[m] + r[m + "_std"]), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(4))
        ax.set_xticklabels([LAB[v] for v in LAB], fontsize=7.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        if i == 0:
            ax.set_title(title, loc="left", fontsize=10, color=INK)
        if j == 0:
            ax.set_ylabel(STAGES[st], fontsize=9.5, color=INK)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0.995, 0.995))
fig.suptitle("step_2: certainly-feasible test instances only (K=3, 5 seeds; bars = mean, whiskers = std)",
             x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{S2}/figures/feasible_subset_comparison.png", dpi=150, facecolor=SURF)
print("saved")
