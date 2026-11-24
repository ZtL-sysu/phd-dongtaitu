"""CPU semantic comparison of the legacy source-map path and current fast path.

Also supports an exact reproduction of the pending fixed T26 micro-overfit
fixture via ``--mode t26``. This is a functional smoke, not a scientific run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

RUN_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = RUN_ROOT / "edsgn"
SRC = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC))
from edsgn.model import ForecastModel, ModelConfig
from edsgn.scheduling import Policy

OLD_PATH = RUN_ROOT / "snapshots" / "model-before-full-source-fastpath-v1.py"


def load_legacy():
    alias = "edsgn._model_before_full_source_fastpath_v1"
    spec = importlib.util.spec_from_file_location(alias, OLD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy model snapshot: {OLD_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module  # dataclasses resolves the defining module here
    spec.loader.exec_module(module)
    return module


def _tensor_diff(a, b):
    if a.shape != b.shape:
        return {"shape_match": False, "left_shape": list(a.shape), "right_shape": list(b.shape)}
    if a.numel() == 0:
        return {"shape_match": True, "max_abs_diff": 0.0, "allclose": True}
    delta = (a.detach().to(torch.float64).cpu() - b.detach().to(torch.float64).cpu()).abs()
    return {"shape_match": True, "max_abs_diff": float(delta.max()),
            "allclose": bool(torch.allclose(a.detach().cpu(), b.detach().cpu(), atol=1e-6, rtol=1e-5))}


def _python_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _python_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_value(v) for v in value]
    return value


def _compare_tree(left, right, path="root", ignored=()):
    if path.split(".")[-1] in ignored:
        return []
    issues = []
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        d = _tensor_diff(left, right)
        if not d["shape_match"] or not d["allclose"]:
            issues.append({"path": path, **d})
    elif isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            issues.append({"path": path, "key_match": False,
                           "left_only": sorted(set(left)-set(right)),
                           "right_only": sorted(set(right)-set(left))})
        for key in sorted(set(left) & set(right), key=str):
            issues.extend(_compare_tree(left[key], right[key], f"{path}.{key}", ignored))
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            issues.append({"path": path, "length_match": False, "left": len(left), "right": len(right)})
        for i, (a, b) in enumerate(zip(left, right)):
            issues.extend(_compare_tree(a, b, f"{path}[{i}]", ignored))
    elif left != right:
        issues.append({"path": path, "left": _python_value(left), "right": _python_value(right)})
    return issues


def make_models(legacy_module, variant, n=7, features=2, history=6, horizon=3):
    common = dict(n_nodes=n, n_features=features, history=history, hmax=horizon,
                  variant=variant, d_total=32 if n > 1 else 16,
                  k=min(3, n-1), candidate_cap=min(5, max(0, n-1)),
                  encoder_width=8, summary_length=3, candidate_seed=113)
    torch.manual_seed(1729)
    old = legacy_module.ForecastModel(legacy_module.ModelConfig(**common))
    new = ForecastModel(ModelConfig(**common))
    result = new.load_state_dict(old.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"weight state mismatch: {result}")
    return old, new


def one_case(legacy_module, variant, policy_kind, *, force_reference=False, n=7):
    old, new = make_models(legacy_module, variant, n=n)
    torch.manual_seed(551 + n)
    batch, history, features, horizon = 2, 6, 2, 3
    x = torch.randn(batch, history, n, features)
    mask = torch.rand(batch, history, n, features) > .22
    # Guarantee a nonempty observed set per step/node, with nonzero observed inputs.
    mask[:, :, 0, :] = True
    x = x + torch.where(mask, torch.full_like(x, .37), torch.zeros_like(x))
    age = torch.rand_like(x) * 3.
    raw = x + .19
    target = torch.randn(batch, horizon, n, features)
    if policy_kind == "full":
        policy = Policy(kind="full")
    else:
        policy = Policy(kind="adaptive", graph_fraction=.35, state_fraction=.35,
                        max_graph_age=100, max_state_age=100,
                        graph_threshold=-1., state_threshold=-1., audit_interval=0,
                        seed=23, force_initialization=True)
    old._force_reference = force_reference
    new._force_reference = force_reference
    old_out = old(x, mask=mask, age=age, raw=raw, policy=policy,
                  return_trace=True, return_intermediate=True)
    new_out = new(x, mask=mask, age=age, raw=raw, policy=policy,
                  return_trace=True, return_intermediate=True)
    old_loss = (old_out["predictions"] - target).square().mean() + old_out["graph_loss"]
    new_loss = (new_out["predictions"] - target).square().mean() + new_out["graph_loss"]
    old_loss.backward(); new_loss.backward()

    checks = {}
    for field in ("predictions", "graph_loss"):
        checks[field] = _tensor_diff(old_out[field], new_out[field])
    checks["forward_loss"] = _tensor_diff(old_loss, new_loss)
    checks["intermediate"] = _compare_tree(old_out["intermediate"], new_out["intermediate"], "intermediate")
    runtime_fields = (
        "neighbors", "weights", "neighbor_valid", "initialized", "u", "spike_pending",
        "pending_log_g", "graph_age", "state_age", "last_input", "last_input_valid",
        "previous_pred_1", "previous_pred_valid", "q", "r_smooth", "levels",
        "candidate_pool", "exploration_phase", "contacts", "user_history", "step", "counters",
    )
    checks["runtime"] = {}
    for field in runtime_fields:
        checks["runtime"][field] = _compare_tree(getattr(old_out["runtime"], field),
                                                   getattr(new_out["runtime"], field),
                                                   f"runtime.{field}")
    # Exclude component host timings only; all trace semantics/counters remain compared.
    checks["trace"] = _compare_tree(old_out["trace"], new_out["trace"], "trace",
                                    ignored=("component_host_ms",))
    old_named = dict(old.named_parameters()); new_named = dict(new.named_parameters())
    gradient = {"parameter_keys_match": set(old_named) == set(new_named),
                "none_pattern_match": True, "max_abs_diff": 0.0, "allclose": True,
                "compared_non_none_parameters": 0, "none_parameters": []}
    for name in sorted(set(old_named) & set(new_named)):
        a, b = old_named[name].grad, new_named[name].grad
        if (a is None) != (b is None):
            gradient["none_pattern_match"] = False
            gradient.setdefault("none_mismatch", []).append(name)
        elif a is None:
            gradient["none_parameters"].append(name)
        else:
            diff = _tensor_diff(a, b)
            gradient["compared_non_none_parameters"] += 1
            gradient["max_abs_diff"] = max(gradient["max_abs_diff"], diff["max_abs_diff"])
            gradient["allclose"] = gradient["allclose"] and diff["allclose"]
    checks["gradients"] = gradient
    partial_actions = {
        "graph": any(0 < len(row["graph_ids"]) < batch*n for row in new_out["trace"]),
        "state": any(0 < len(row["state_ids"]) < batch*n for row in new_out["trace"]),
        "graph_action_sizes": [len(row["graph_ids"]) for row in new_out["trace"]],
        "state_action_sizes": [len(row["state_ids"]) for row in new_out["trace"]],
    }
    semantic_issues = []
    for key in ("predictions", "graph_loss", "forward_loss"):
        if not checks[key]["allclose"]: semantic_issues.append(key)
    semantic_issues.extend(checks["intermediate"])
    semantic_issues.extend(checks["trace"])
    for field, issues in checks["runtime"].items(): semantic_issues.extend(issues)
    if not gradient["parameter_keys_match"] or not gradient["none_pattern_match"] or not gradient["allclose"]:
        semantic_issues.append("gradients")
    if policy_kind == "adaptive" and not force_reference:
        # V0's static graph is intentionally initialized once and never refreshed;
        # its adaptive state path must still perform real partial actions.
        graph_partial_required = variant != "V0"
        if not partial_actions["state"] or (graph_partial_required and not partial_actions["graph"]):
            semantic_issues.append("adaptive_did_not_execute_expected_real_partial_actions")
    return {"variant": variant, "policy": policy_kind,
            "reference_source_map_path": bool(force_reference), "n": n, "features": features,
            "batch": batch, "history": history, "horizon": horizon,
            "input_observed_nonzero_count": int((x[mask] != 0).sum()),
            "partial_action_evidence": partial_actions,
            "checks": checks, "status": "pass" if not semantic_issues else "fail",
            "issues": semantic_issues}


def run_equivalence():
    legacy = load_legacy()
    torch.set_num_threads(2)
    rows = []
    variants = ("V0", "V1", "V2", "V3", "V4", "V9")
    for variant in variants:
        rows.append(one_case(legacy, variant, "full"))
        rows.append(one_case(legacy, variant, "adaptive"))
        rows.append(one_case(legacy, variant, "adaptive", force_reference=True))
    # N=1 has an empty neighbor support and no valid partial action by definition.
    rows.append(one_case(legacy, "V4", "full", n=1))
    failures = [{"variant": r["variant"], "policy": r["policy"],
                "reference_source_map_path": r["reference_source_map_path"], "issues": r["issues"]}
               for r in rows if r["status"] != "pass"]
    return {"scope": "CPU semantic equality; no host timing comparison",
            "torch_threads": torch.get_num_threads(), "legacy_snapshot": str(OLD_PATH),
            "current_model": str(PACKAGE_ROOT / "src" / "edsgn" / "model.py"),
            "variants": list(variants), "case_count": len(rows), "cases": rows,
            "status": "pass" if not failures else "fail", "failures": failures}


def run_t26():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = ForecastModel(ModelConfig(n_nodes=3, n_features=1, history=3, hmax=2,
                                      variant="V4", k=2, candidate_cap=2, d_total=16))
    x = torch.randn(4, 4, 3, 1)
    last = x[:, -1]
    target = (last * .4).unsqueeze(1).expand(-1, 2, -1, -1).contiguous()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    losses = []
    for _ in range(70):
        optimizer.zero_grad()
        pred = model(x)["predictions"]
        loss = (pred-target).square().mean()
        losses.append(float(loss.detach()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    initial, final = losses[0], losses[-1]
    ratio = final / initial if initial else float("inf")
    return {"scope": "approved fixed T26 micro-overfit engineering smoke; not formal accuracy/P4",
            "torch_threads": torch.get_num_threads(), "seed": 7, "windows": 4,
            "steps": 70, "variant": "V4 SNN retained", "target_rule": "0.4 * each window's last value, repeated across H=2",
            "initial_loss": initial, "final_loss_before_last_optimizer_step": final,
            "final_over_initial_ratio": ratio, "required_ratio_lt": .5,
            "criterion_pass": bool(torch.isfinite(torch.tensor(final)) and final < initial*.5),
            "loss_curve": losses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("equivalence", "t26"), default="equivalence")
    args = parser.parse_args()
    result = run_equivalence() if args.mode == "equivalence" else run_t26()
    print(json.dumps(result, indent=2))
    return 0 if result.get("status", "pass" if result.get("criterion_pass") else "fail") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
