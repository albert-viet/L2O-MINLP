"""
Shared metrics of the mixed-equality diagnosis study (float64, vectorized).

All violations are RAW (unsquared, unweighted) and follow the same formulas as
the Pyomo constraints in src/problem/math_solver/rosenbrock{,_eq}.py:
    inner  = relu(K*p/2 - sum x_2i+1)
    outer  = relu(sum x_2i^2 - K*p)
    linear = relu(b^T x_even) + relu(q^T x_odd)
    h      = sum x_2i - c*sum x_2i+1   (mixed) | sum x_2i - c*K*p/2 (cont) | 0 (none)
"""
import numpy as np
import pandas as pd

EPS_LIST = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
FEAS_TOL = 1e-6   # tolerance used by the original framework (check_violation, num_viols)


def coefficients(num_blocks):
    """
    linear constraint coefficients, same RandomState(17) as the models
    """
    rng = np.random.RandomState(17)
    b = rng.normal(scale=1, size=(num_blocks))
    q = rng.normal(scale=1, size=(num_blocks))
    return b, q


def per_instance_metrics(x, p, a, num_blocks, steepness, eq_coef, eq_mode):
    """
    x: (N, 2K), p: (N, 1), a: (N, K) -> DataFrame with one row per instance:
    obj, inner, outer, linear, v_ineq, h (signed), abs_h
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    a = np.asarray(a, dtype=np.float64)
    b, q = coefficients(num_blocks)
    x_e, x_o = x[:, 0::2], x[:, 1::2]
    obj = np.sum((a - x_e) ** 2 + steepness * (x_o - x_e ** 2) ** 2, axis=1)
    inner = np.maximum(num_blocks * p / 2 - x_o.sum(axis=1), 0.0)
    outer = np.maximum((x_e ** 2).sum(axis=1) - num_blocks * p, 0.0)
    linear = np.maximum(x_e @ b, 0.0) + np.maximum(x_o @ q, 0.0)
    if eq_mode == "mixed":
        h = x_e.sum(axis=1) - eq_coef * x_o.sum(axis=1)
    elif eq_mode == "cont":
        h = x_e.sum(axis=1) - eq_coef * num_blocks * p / 2
    elif eq_mode == "none":
        h = np.zeros(len(x))
    else:
        raise ValueError(f"unknown eq_mode '{eq_mode}'")
    return pd.DataFrame({"obj": obj, "inner": inner, "outer": outer, "linear": linear,
                         "v_ineq": inner + outer + linear, "h": h, "abs_h": np.abs(h)})


def tolerance_curve(df, eps_list=EPS_LIST):
    """
    F(eps) = Pr(|h| <= eps and V_ineq <= eps) over the instances in df
    """
    return {f"F_{eps:g}": float(np.mean((df["abs_h"] <= eps) & (df["v_ineq"] <= eps)))
            for eps in eps_list}


def summarize(df):
    """
    run-level summary of a per-instance DataFrame
    """
    out = {"n": len(df),
           "obj": df["obj"].mean(),
           "v_ineq": df["v_ineq"].mean(),
           "inner": df["inner"].mean(),
           "outer": df["outer"].mean(),
           "linear": df["linear"].mean(),
           "abs_h": df["abs_h"].mean(),
           "feas_rate": float(np.mean((df["abs_h"] <= FEAS_TOL) & (df["v_ineq"] <= FEAS_TOL)))}
    out.update(tolerance_curve(df))
    return out
