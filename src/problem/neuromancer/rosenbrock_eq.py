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
    """
    def __init__(self, input_keys, steepness, num_blocks, penalty_weight=50,
                 penalty_weight_eq=None, eq_coef=1.0, output_key="loss"):
        super().__init__(input_keys, steepness, num_blocks, penalty_weight, output_key)
        # separate penalty weight for the equality term (defaults to penalty_weight)
        self.penalty_weight_eq = penalty_weight_eq if penalty_weight_eq is not None else penalty_weight
        # coefficient c in: sum(x_2i) = eq_coef * sum(x_2i+1)
        self.eq_coef = eq_coef

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
        squared violation of sum_i x_2i = eq_coef * sum_i x_2i+1
        """
        x = input_dict[self.x_key]
        lhs = torch.sum(x[:, ::2], dim=1)
        rhs = self.eq_coef * torch.sum(x[:, 1::2], dim=1)
        return (lhs - rhs) ** 2

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
