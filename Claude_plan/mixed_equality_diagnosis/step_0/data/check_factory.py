"""
step_0 / to-do 2: kiem tra factory 4 bien the.
Chay tu goc repo: python Claude_plan/mixed_equality_diagnosis/step_0/data/check_factory.py
"""
import sys
sys.path.insert(0, ".")
import torch
from run.diag_eq.variants import VARIANTS, build_variant, eq_coef_from_pstar
from run.diag_eq import pipeline

K, STEEP = 3, 50
c = eq_coef_from_pstar(4.5)
p, a = 3.2, (2.4, 1.8, 1.5)           # p < p* = 4.5 -> feasible
expected_cons = {"V1": 4, "V2": 5, "V3": 5, "V4": 5}
expected_nint = {"V1": 3, "V2": 3, "V3": 0, "V4": 3}
ok = True
for name in VARIANTS:
    v = build_variant(name, K, STEEP, c, 50.0, 100.0, timelimit=60)
    ncons = len(v.model.model.cons)
    nint = len(v.model.int_ind["x"])
    v.model.set_param_val({"p": p, "a": a})
    xval, obj = v.model.solve()
    viol = v.model.cal_violation()
    # loss breakdown
    x = torch.tensor([list(xval["x"].values())], dtype=torch.float32)
    d = {"p": torch.tensor([[p]]), "a": torch.tensor([list(a)]), "x_rnd": x}
    bd = v.loss_fn.cal_violation_breakdown(d)
    keys_ok = sorted(bd) == ["equality", "inner", "linear", "outer"]
    cons_ok, int_ok = ncons == expected_cons[name], nint == expected_nint[name]
    ok &= cons_ok and int_ok and keys_ok and float(abs(viol).max()) < 1e-5
    print(f"{name}: cons={ncons} (exp {expected_cons[name]}) n_int={nint} (exp {expected_nint[name]}) "
          f"obj={obj:.4f} max_viol={abs(viol).max():.2e} breakdown_keys_ok={keys_ok} "
          f"eq_breakdown={bd['equality'].item():.2e}")
    # network with the variant's int_ind builds and rounds (V3 -> int_ind empty)
    comps = pipeline.build_components(v.model, pipeline.DEFAULT_CFG)
    st = pipeline.forward_stages(comps, [[p]], [list(a)])
    x_hat_odd = st["x_rnd"][0, 1::2]
    is_integer = bool(torch.all(torch.abs(x_hat_odd - torch.round(x_hat_odd)) < 1e-6))
    rounding_ok = (not is_integer) if name == "V3" else is_integer   # V3: no rounding applied
    ok &= rounding_ok
    print(f"    forward ok: x_bar={tuple(st['x'].shape)} x_hat={tuple(st['x_rnd'].shape)}; "
          f"x_hat integer coords={x_hat_odd.tolist()} integral={is_integer} "
          f"(expected {'False' if name == 'V3' else 'True'}) rounding_ok={rounding_ok}")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
