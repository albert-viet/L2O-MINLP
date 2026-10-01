"""
Parametric Mixed Integer Constrained Rosenbrock Problem with an extra
equality constraint mixing continuous and integer variables:

    sum_i x_2i = eq_coef * sum_i x_2i+1

With eq_coef=1 (default) this is "total continuous mass = total integer
count". eq_coef controls the analytic feasibility threshold p* = (2/eq_coef)^2
(from combining this equality with the existing inner/outer constraints via
Cauchy-Schwarz): instances with p > p* are infeasible by construction,
independent of `a`.
"""

from pyomo import environ as pe

from src.problem.math_solver.rosenbrock import rosenbrock

EQ_MODES = ("none", "cont", "mixed")


class rosenbrock_eq(rosenbrock):
    """
    MIRB with an additional equality constraint linking the continuous
    block variables (x_2i) to the integer block variables (x_2i+1).

    eq_mode (used by the mixed-equality diagnosis study):
      "mixed" - sum_i x_2i = eq_coef * sum_i x_2i+1 (default)
      "cont"  - sum_i x_2i = eq_coef * K * p / 2, involves continuous variables only
                (same RHS as "mixed" when the inner constraint is tight, same p*)
      "none"  - no equality (original MIRB)
    relax_int: if True, the integer variables x_2i+1 become continuous.
    """
    def __init__(self, steepness, num_blocks, timelimit=None, eq_coef=1.0,
                 eq_mode="mixed", relax_int=False):
        super().__init__(steepness, num_blocks, timelimit)
        if eq_mode not in EQ_MODES:
            raise ValueError(f"eq_mode must be one of {EQ_MODES}, got '{eq_mode}'")
        self.eq_coef = eq_coef
        self.eq_mode = eq_mode
        m = self.model
        if relax_int:
            for i in range(num_blocks):
                m.x[2*i+1].domain = pe.Reals
        if eq_mode == "mixed":
            # equality constraint: sum of continuous vars = eq_coef * sum of integer vars
            m.cons.add(sum(m.x[2*i] for i in range(num_blocks)) ==
                       eq_coef * sum(m.x[2*i+1] for i in range(num_blocks)))
        elif eq_mode == "cont":
            # equality constraint on continuous vars only, RHS depends on p
            m.cons.add(sum(m.x[2*i] for i in range(num_blocks)) ==
                       eq_coef * num_blocks * m.p / 2)


if __name__ == "__main__":

    from src.utlis import ms_test_solve

    steepness = 50    # steepness factor
    num_blocks = 3    # number of expression blocks
    timelimit = 60    # time limit

    # params
    p, a = 3.2, (2.4, 1.8, 1.5)
    params = {"p": p, "a": a}
    # init model
    model = rosenbrock_eq(steepness=steepness, num_blocks=num_blocks, timelimit=timelimit)

    print("======================================================")
    print("Solve MINLP problem with equality constraint:")
    model.set_param_val(params)
    solvals, _ = ms_test_solve(model, tee=True)
