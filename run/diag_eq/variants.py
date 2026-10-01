"""
The four problem variants of the mixed-equality diagnosis study (MIRB).

All variants share the same architecture, seed, split and penalties; only the
equality constraint and the integrality of x_2i+1 change.

    V1: no equality, integer             (original MIRB)
    V2: sum x_2i = c*K*p/2, integer      (equality on continuous variables only)
    V3: sum x_2i = c*sum x_2i+1, relaxed (mixed equality, integer variables relaxed)
    V4: sum x_2i = c*sum x_2i+1, integer (mixed equality + RC integer correction)
"""
import math
from collections import namedtuple

VARIANTS = {
    "V1": {"eq_mode": "none", "relax_int": False, "desc": "no equality"},
    "V2": {"eq_mode": "cont", "relax_int": False, "desc": "continuous-only equality"},
    "V3": {"eq_mode": "mixed", "relax_int": True, "desc": "mixed equality, integers relaxed"},
    "V4": {"eq_mode": "mixed", "relax_int": False, "desc": "mixed equality + RC"},
}

Variant = namedtuple("Variant", ["name", "model", "loss_fn", "eq_mode", "relax_int", "eq_coef"])


def eq_coef_from_pstar(p_star):
    """
    equality coefficient c such that the analytic feasibility threshold is
    p* = (2/c)^2  <=>  c = 2/sqrt(p*)
    """
    return 2 / math.sqrt(p_star)


def build_variant(name, num_blocks, steepness, eq_coef, penalty, penalty_eq, timelimit=60):
    """
    Build the exact solver model (Pyomo) and the differentiable loss of a variant
    """
    from src.problem import msRosenbrockEq, nmRosenbrockEq
    spec = VARIANTS[name]
    model = msRosenbrockEq(steepness, num_blocks, timelimit=timelimit, eq_coef=eq_coef,
                           eq_mode=spec["eq_mode"], relax_int=spec["relax_int"])
    loss_fn = nmRosenbrockEq(["p", "a", "x_rnd"], steepness, num_blocks, penalty, penalty_eq,
                             eq_coef=eq_coef, eq_mode=spec["eq_mode"])
    return Variant(name, model, loss_fn, spec["eq_mode"], spec["relax_int"], eq_coef)
