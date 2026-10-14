from pathlib import Path

p = Path(r"backend/Imaging/05_gradcam.py")
s = p.read_text(encoding="utf-8")

old = '''    ig=torch.autograd.grad(s,xf)[0]; direction=ig/ig.abs().mean().clamp_min(1e-8); eps=a.finite_difference_eps
    with torch.no_grad():
        plus=get_score(model((xf+eps*direction).detach()),a.attribute,a.target_index)[0]
        minus=get_score(model((xf-eps*direction).detach()),a.attribute,a.target_index)[0]
    fd=float(((plus-minus)/(2*eps)).cpu()); directional=float((ig*direction).sum().cpu())
    passed=bool(np.isfinite(fd) and fd>0 and directional>0)
'''

new = '''    ig=torch.autograd.grad(s,xf)[0]
    direction=ig/ig.abs().mean().clamp_min(1e-8)
    base_eps=float(a.finite_difference_eps)
    directional=float((ig*direction).sum().detach().cpu())

    # A single tiny epsilon can produce identical float32 scores.
    # Probe progressively larger steps without changing the model or CAM.
    checks=[]
    with torch.no_grad():
        for scale in (1.0, 10.0, 100.0):
            eps=base_eps*scale
            plus=get_score(
                model((xf+eps*direction).detach()),a.attribute,a.target_index
            )[0]
            minus=get_score(
                model((xf-eps*direction).detach()),a.attribute,a.target_index
            )[0]
            fd=float(((plus-minus)/(2*eps)).detach().cpu())
            pscore=float(plus.detach().cpu())
            mscore=float(minus.detach().cpu())
            # Compare against autograd with a permissive relative tolerance.
            # This is a numerical gradient check, not a faithfulness test.
            consistent=(
                np.isfinite(fd)
                and np.isfinite(directional)
                and directional > 0
                and fd > 0
                and fd >= directional*0.01
                and fd <= directional*100.0
            )
            checks.append({
                "epsilon":eps,
                "score_plus":pscore,
                "score_minus":mscore,
                "finite_difference_derivative":fd,
                "gradient_directional_derivative":directional,
                "passed":bool(consistent),
            })

    passed=any(item["passed"] for item in checks)
    best_check=max(
        checks,
        key=lambda item: (
            item["passed"],
            item["finite_difference_derivative"] > 0,
        ),
    )
    eps=best_check["epsilon"]
    plus_value=best_check["score_plus"]
    minus_value=best_check["score_minus"]
    fd=best_check["finite_difference_derivative"]
'''

if old not in s:
    raise SystemExit(
        "PATCH STOPPED: expected original diagnostic block was not found. "
        "No changes were written."
    )

s = s.replace(old, new, 1)

old_meta = '''      "score_expected_rating":scalar,"rank_sensitivity_check":{"method":"central finite difference along normalized input gradient","epsilon":eps,
      "score_plus":float(plus.cpu()),"score_minus":float(minus.cpu()),"finite_difference_derivative":fd,
      "gradient_directional_derivative":directional,"passed":passed,"scope":"gradient plumbing check, not clinical faithfulness"}}
'''

new_meta = '''      "score_expected_rating":scalar,"rank_sensitivity_check":{"method":"multi-epsilon central finite difference along normalized input gradient",
      "epsilon":eps,"score_plus":plus_value,"score_minus":minus_value,
      "finite_difference_derivative":fd,
      "gradient_directional_derivative":directional,
      "checks":checks,"passed":passed,
      "scope":"numerical gradient plumbing check, not clinical faithfulness"}}
'''

if old_meta not in s:
    raise SystemExit(
        "PATCH STOPPED: expected metadata block was not found. "
        "The original file was left unchanged."
    )

s = s.replace(old_meta, new_meta, 1)
p.write_text(s, encoding="utf-8")
print("Patched multi-epsilon rank-sensitivity diagnostic:", p)
