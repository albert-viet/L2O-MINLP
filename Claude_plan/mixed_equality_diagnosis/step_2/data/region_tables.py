"""
step_2: (1) bang metrics tach vung p<p* / p>=p* cho model huan luyen tren mixed dataset (tu cac run cua step_1),
        (2) bang metrics cho model huan luyen tren feasible-only dataset (run cua step_2),
        (3) doi chieu ground truth SCIP voi du doan giai tich, (4) objective gap so voi exact.
Chay tu goc repo: python Claude_plan/mixed_equality_diagnosis/step_2/data/region_tables.py <phan>
  phan = regions | ground_truth | gap | all
"""
import sys
sys.path.insert(0, ".")
import math
import numpy as np
import pandas as pd

S1 = "Claude_plan/mixed_equality_diagnosis/step_1/data/runs"
S2 = "Claude_plan/mixed_equality_diagnosis/step_2/data"
P_STAR = 4.5
C = 2 / math.sqrt(P_STAR)
K = 3
VARS = ["V1", "V2", "V3", "V4"]
SEEDS = [0, 1, 2, 3, 4]
METR = ["abs_h", "v_ineq", "obj", "feas_rate", "F_0.1", "F_0.01"]


def load(root, v, seed, stage):
    return pd.read_csv(f"{root}/{v}_seed{seed}_{stage}.csv")


def add_flags(df, eps_list=(1e-1, 1e-2)):
    df = df.copy()
    df["feas_rate"] = ((df.abs_h <= 1e-6) & (df.v_ineq <= 1e-6)).astype(float)
    for e in eps_list:
        df[f"F_{e:g}"] = ((df.abs_h <= e) & (df.v_ineq <= e)).astype(float)
    return df


def analytic_int_feasible(p):
    """exists integer n=sum z with K p/2 <= n <= K sqrt(p)/c  (inner + outer + eq, ignores linear constraints)"""
    lo, hi = K * p / 2, K * math.sqrt(p) / C
    return math.ceil(lo - 1e-9) <= math.floor(hi + 1e-9)


def regions():
    rows = []
    for v in VARS:
        for stage in ["noproj", "proj"]:
            df = pd.concat([add_flags(load(S1, v, s, stage)).assign(seed=s) for s in SEEDS])
            df["region"] = np.where(df.p < P_STAR, "p<p*", "p>=p*")
            df["analytic_int_feas"] = df.p.apply(analytic_int_feasible)
            for reg, d in df.groupby("region"):
                per_seed = d.groupby("seed")[METR].mean()
                row = {"variant": v, "stage": stage, "region": reg, "n": len(d)}
                for m in METR:
                    row[m] = f"{per_seed[m].mean():.4g} ± {per_seed[m].std():.2g}"
                rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f"{S2}/mixed_by_region.csv", index=False)
    return t


N_EXACT = 200    # tat ca test instance moi (dataset, seed) duoc giai bang SCIP


def load_exact(ds, v, seed):
    return pd.read_csv(f"{S2}/exact/{ds}_{v}_seed{seed}.csv")


def ground_truth():
    """
    so sanh trang thai SCIP voi du doan giai tich theo vung p<p* / p>=p*
      V1: luon khoi tao duoc (khong co equality); V2, V3: du doan feasible iff p<p*;
      V4: du doan feasible iff p<p* va ton tai so nguyen n co Kp/2 <= n <= K sqrt(p)/c
    """
    rows = []
    for ds in ["mixed", "feasible_only"]:
        for v in VARS:
            df = pd.concat([load_exact(ds, v, s).assign(seed=s) for s in SEEDS])
            df["region"] = np.where(df.p < P_STAR, "p<p*", "p>=p*")
            if v == "V1":
                df["pred_feas"] = True
            elif v == "V4":
                df["pred_feas"] = (df.p < P_STAR) & df.p.apply(analytic_int_feasible)
            else:
                df["pred_feas"] = df.p < P_STAR
            for reg, d in df.groupby("region"):
                vc = d.status.value_counts().to_dict()
                rows.append({"dataset": ds, "variant": v, "region": reg, "n": len(d),
                             "solved": vc.get("solved", 0), "infeasible": vc.get("infeasible", 0),
                             "timeout": vc.get("maxTimeLimit", 0),
                             "other": len(d) - vc.get("solved", 0) - vc.get("infeasible", 0) - vc.get("maxTimeLimit", 0),
                             "pred_feasible": int(d.pred_feas.sum()),
                             "solved&pred_feasible": int(((d.status == "solved") & d.pred_feas).sum()),
                             "solved&~pred_feasible": int(((d.status == "solved") & ~d.pred_feas).sum()),
                             "infeasible&pred_feasible": int(((d.status == "infeasible") & d.pred_feas).sum())})
    t = pd.DataFrame(rows)
    t.to_csv(f"{S2}/ground_truth_vs_analytic.csv", index=False)
    return t


def gap():
    """
    objective gap = (obj_neural - obj_exact) / |obj_exact| tren cac instance SCIP 'solved' (optimal),
    theo bien the, dataset, stage, vung. Luu y: nghiem neural vi pham rang buoc nen gap co the am.
    """
    rows = []
    for ds, root in [("mixed", S1), ("feasible_only", f"{S2}/feasible_only")]:
        for v in VARS:
            for stage in ["noproj", "proj"]:
                recs = []
                for s in SEEDS:
                    nn = load(root, v, s, stage).iloc[:N_EXACT]
                    ex = load_exact(ds, v, s)
                    assert np.allclose(nn.p.values, ex.p.values, atol=1e-5), (ds, v, s)
                    d = pd.DataFrame({"seed": s, "p": ex.p, "status": ex.status, "obj_ex": ex.obj, "obj_nn": nn.obj.values,
                                      "v_ineq": nn.v_ineq.values, "abs_h": nn.abs_h.values})
                    recs.append(d)
                d = pd.concat(recs)
                d = d[d.status == "solved"].copy()
                d["region"] = np.where(d.p < P_STAR, "p<p*", "p>=p*")
                d["gap"] = (d.obj_nn - d.obj_ex) / d.obj_ex.abs()
                d["nn_below_exact"] = (d.obj_nn < d.obj_ex).astype(float)
                for reg, g in d.groupby("region"):
                    rows.append({"dataset": ds, "variant": v, "stage": stage, "region": reg, "n_solved": len(g),
                                 "obj_exact": g.obj_ex.mean(), "obj_neural": g.obj_nn.mean(),
                                 "gap_mean": g.gap.mean(), "gap_median": g.gap.median(),
                                 "share_neural_below_exact": g.nn_below_exact.mean(),
                                 "v_ineq": g.v_ineq.mean(), "abs_h": g.abs_h.mean()})
    t = pd.DataFrame(rows)
    t.to_csv(f"{S2}/objective_gap.csv", index=False)
    return t


def pred_feasible(v, p):
    """du doan giai tich (da doi chieu voi SCIP o ground_truth()) instance co nghiem hay khong"""
    if v == "V1":
        return np.ones(len(p), dtype=bool)
    ok = p < P_STAR
    if v == "V4":
        ok = ok & np.array([analytic_int_feasible(x) for x in p])
    return ok


def certain_feasible():
    """
    metrics tren tap con instance chac chan co nghiem (pred_feasible) vs tap con khong co nghiem,
    cho model huan luyen tren mixed va tren feasible-only
    """
    rows = []
    for ds, root in [("mixed", S1), ("feasible_only", f"{S2}/feasible_only")]:
        for v in VARS:
            for stage in ["noproj", "proj"]:
                df = pd.concat([add_flags(load(root, v, s, stage)).assign(seed=s) for s in SEEDS])
                df["subset"] = np.where(pred_feasible(v, df.p.values), "feasible", "infeasible")
                for sub, d in df.groupby("subset"):
                    per_seed = d.groupby("seed")[METR].mean()
                    row = {"trained_on": ds, "variant": v, "stage": stage, "subset": sub, "n": len(d)}
                    for m in METR:
                        row[m] = per_seed[m].mean()
                        row[m + "_std"] = per_seed[m].std()
                    rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f"{S2}/certain_feasible_subset.csv", index=False)
    return t


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("regions", "all"):
        t = regions()
        print(t.to_string())
    if what in ("ground_truth", "all"):
        print(ground_truth().to_string())
    if what in ("subset", "all"):
        print(certain_feasible().round(4).to_string())
    if what in ("gap", "all"):
        print(gap().round(3).to_string())
