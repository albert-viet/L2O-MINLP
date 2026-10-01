"""
Parametric Mixed Integer Rosenbrock Problem with an extra differentiable
penalty term for the equality constraint sum_i x_2i = sum_i x_2i+1
(continuous block variables vs. integer block variables).
"""

import torch

from src.problem.neuromancer.rosenbrock import penaltyLoss


class penaltyLoss_eq(penaltyLoss):
    """
    Penalty loss for the Rosenbrock problem with an additional equality
    constraint mixing continuous and integer variables.

    eq_mode (must match math_solver/rosenbrock_eq.py):
      "mixed" - sum_i x_2i = eq_coef * sum_i x_2i+1 (default)
      "cont"  - sum_i x_2i = eq_coef * K * p / 2
      "none"  - no equality, equality violation is identically 0
    """
    def __init__(self, input_keys, steepness, num_blocks, penalty_weight=50,
                 penalty_weight_eq=None, eq_coef=1.0, output_key="loss", eq_mode="mixed"):
        super().__init__(input_keys, steepness, num_blocks, penalty_weight, output_key)
        if eq_mode not in ("none", "cont", "mixed"):
            raise ValueError(f"unknown eq_mode '{eq_mode}'")
        # separate penalty weight for the equality term (defaults to penalty_weight)
        self.penalty_weight_eq = penalty_weight_eq if penalty_weight_eq is not None else penalty_weight
        # coefficient c in: sum(x_2i) = eq_coef * sum(x_2i+1)
        self.eq_coef = eq_coef
        self.eq_mode = eq_mode

    def forward(self, input_dict):
        """
        forward pass
        """
        obj = self.cal_obj(input_dict)
        viol_ineq = super().cal_constr_viol(input_dict)
        viol_eq = self.cal_eq_violation(input_dict)
        loss = obj + self.penalty_weight * viol_ineq + self.penalty_weight_eq * viol_eq
        input_dict[self.output_key] = torch.mean(loss)
        return input_dict

    def cal_eq_violation(self, input_dict):
        """
        squared violation of the equality constraint selected by eq_mode
        """
        x = input_dict[self.x_key]
        return self.eq_residual(input_dict) ** 2 if self.eq_mode != "none" \
            else torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def eq_residual(self, input_dict):
        """
        signed equality residual h(x) = lhs - rhs (0 if eq_mode is "none")
        """
        x, p = input_dict[self.x_key], input_dict[self.p_key]
        lhs = torch.sum(x[:, ::2], dim=1)
        if self.eq_mode == "mixed":
            rhs = self.eq_coef * torch.sum(x[:, 1::2], dim=1)
        elif self.eq_mode == "cont":
            rhs = self.eq_coef * self.num_blocks * p[:, 0] / 2
        else:
            return torch.zeros_like(lhs)
        return lhs - rhs

    def cal_constr_viol(self, input_dict):
        """
        total RAW violation (inequality + equality, unweighted) - used by
        gradientProjection, which expects a single scalar-per-sample
        violation to minimize via gradient descent.
        """
        return super().cal_constr_viol(input_dict) + self.cal_eq_violation(input_dict)

    def cal_violation_breakdown(self, input_dict):
        """
        per-sample violation broken down by constraint type, for logging
        and plotting (duplicates the parent's inner/outer/linear terms
        since the parent does not expose them separately).
        """
        x, p = input_dict[self.x_key], input_dict[self.p_key]
        if self.device is None:
            self.device = x.device
            self.b = self.b.to(self.device)
            self.q = self.q.to(self.device)
        # inner constraint violation
        lhs_inner = torch.sum(x[:, 1::2], dim=1)
        rhs_inner = self.num_blocks * p[:, 0] / 2
        inner = torch.relu(rhs_inner - lhs_inner)
        # outer constraint violation
        lhs_outer = torch.sum(x[:, ::2] ** 2, dim=1)
        rhs_outer = self.num_blocks * p[:, 0]
        outer = torch.relu(lhs_outer - rhs_outer)
        # linear constraints violation
        lhs_1 = torch.matmul(x[:, 0::2], self.b)
        lhs_2 = torch.matmul(x[:, 1::2], self.q)
        linear = torch.relu(lhs_1) + torch.relu(lhs_2)
        # equality constraint violation
        equality = self.cal_eq_violation(input_dict)
        return {"inner": inner, "outer": outer, "linear": linear, "equality": equality}
