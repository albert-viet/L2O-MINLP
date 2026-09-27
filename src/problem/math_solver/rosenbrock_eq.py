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

from src.problem.math_solver.rosenbrock import rosenbrock


class rosenbrock_eq(rosenbrock):
    """
    MIRB with an additional equality constraint linking the continuous
    block variables (x_2i) to the integer block variables (x_2i+1).
    """
    def __init__(self, steepness, num_blocks, timelimit=None, eq_coef=1.0):
        super().__init__(steepness, num_blocks, timelimit)
        self.eq_coef = eq_coef
        m = self.model
        # equality constraint: sum of continuous vars = eq_coef * sum of integer vars
        m.cons.add(sum(m.x[2*i] for i in range(num_blocks)) ==
                   eq_coef * sum(m.x[2*i+1] for i in range(num_blocks)))


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
