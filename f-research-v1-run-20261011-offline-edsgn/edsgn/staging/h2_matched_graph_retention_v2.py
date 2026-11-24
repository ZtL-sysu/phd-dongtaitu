"""CPU/toy-only prospective H2 matched-graph retention intervention.

This staging prototype has no CLI and never accepts targets or oracle matrices.
It is not integrated into the package or test/replay controller.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math

import torch


ARMS = ("informative", "shuffled", "constant", "reset")


@dataclass(frozen=True)
class H2TeacherDescriptor:
    variant: str = "V4"
    policy_kind: str = "full"
    source: str = "causal_inputs_only"
    oracle_accessed: bool = False
    target_accessed: bool = False
    eval_mode: bool = True
    gradients_enabled: bool = False


EXPECTED_DESCRIPTOR = H2TeacherDescriptor()


def parameter_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate(model, policy, descriptor, arms):
    from edsgn.scheduling import Policy

    if descriptor != EXPECTED_DESCRIPTOR:
        raise ValueError("Unsupported teacher descriptor; require causal V4 inputs without oracle/targets")
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("Matched retention probe requires eval mode and disabled gradients")
    if model.config.variant != "V4" or not model.spiking or not model.retention:
        raise ValueError("Matched retention probe is restricted to spiking, retention-enabled V4")
    if model.config.retention_mode != "informative" or model.config.k <= 0:
        raise ValueError("Teacher must use informative retention and a nonempty graph support")
    if not isinstance(policy, Policy) or policy.kind != "full" or policy.cost_table or policy.runtime_budget_ms is not None:
        raise ValueError("Probe requires the full reference policy without a calibrated runtime budget")
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("Unknown or duplicate retention arm")


@contextmanager
def _replace_refresh_graph(model, wrapper):
    had_local = "refresh_graph" in model.__dict__
    local_value = model.__dict__.get("refresh_graph")
    original = model.refresh_graph
    model.refresh_graph = wrapper(original)
    try:
        yield
    finally:
        if had_local:
            model.refresh_graph = local_value
        else:
            delattr(model, "refresh_graph")


def _retention_log_g(arm, c, initialized_before, row_ids, batch_size, n_nodes, beta):
    initialized_before = initialized_before.bool()
    if arm == "informative":
        values = -beta * c
    elif arm == "shuffled":
        signal = c.clone()
        for batch_index in range(batch_size):
            positions = (row_ids // n_nodes == batch_index).nonzero().flatten()
            if positions.numel():
                signal[positions] = torch.roll(c[positions], shifts=1, dims=0)
        values = -beta * signal
    elif arm == "constant":
        values = torch.full_like(c, math.log(.9))
    elif arm == "reset":
        values = torch.full_like(c, -80.)
    else:
        raise ValueError(f"Unknown retention arm: {arm}")
    # Initial graph creation always uses g=1; only later refreshes intervene.
    return torch.where(initialized_before, values, torch.zeros_like(values))


def _run_teacher(model, inputs, policy):
    rows = []

    def factory(original):
        def wrapped(observation, runtime, row_ids, summary, monitor):
            initialized_before = runtime.initialized[row_ids].detach().clone()
            pending_before = runtime.pending_log_g.detach().clone()
            result = original(observation, runtime, row_ids, summary, monitor)
            c, loss, audit = result
            rows.append({
                "step": int(runtime.step), "row_ids": row_ids.detach().clone(),
                "initialized_before": initialized_before,
                "supports": runtime.neighbors[row_ids].detach().clone(),
                "weights": runtime.weights[row_ids].detach().clone(),
                "valid": runtime.neighbor_valid[row_ids].detach().clone(),
                "c": c.detach().clone(), "pending_before": pending_before,
                "state_ids": None,
            })
            return result
        return wrapped

    with _replace_refresh_graph(model, factory):
        result = model(**inputs, policy=policy, return_trace=True)
    if len(rows) != len(result["trace"]):
        raise RuntimeError("Teacher graph captures do not align with model steps")
    for row, trace in zip(rows, result["trace"]):
        row["state_ids"] = torch.as_tensor(trace["state_ids"], dtype=torch.long,
                                             device=row["row_ids"].device)
    return result, rows


def _run_arm(model, inputs, policy, teacher_rows, arm):
    cursor = 0
    diagnostics = []

    def factory(original):
        def wrapped(observation, runtime, row_ids, summary, monitor):
            nonlocal cursor
            if cursor >= len(teacher_rows):
                raise RuntimeError("Arm refreshed more graph rows than the teacher")
            prescribed = teacher_rows[cursor]
            cursor += 1
            if int(runtime.step) != prescribed["step"] or not torch.equal(row_ids, prescribed["row_ids"]):
                raise RuntimeError("Teacher and arm graph action indices differ")
            initialized_before = runtime.initialized[row_ids].detach().clone()
            if not torch.equal(initialized_before, prescribed["initialized_before"]):
                raise RuntimeError("Teacher and arm graph initialization histories differ")
            pending_before = runtime.pending_log_g.detach().clone()
            if not torch.equal(pending_before, prescribed["pending_before"]):
                raise RuntimeError("Teacher and arm deferred-retention state differs before graph refresh")
            actual_c, candidate_loss, candidate_audit = original(
                observation, runtime, row_ids, summary, monitor)
            if candidate_audit is None:
                raise RuntimeError("V4 full refresh did not produce a graph audit")

            runtime.neighbors = runtime.neighbors.index_copy(0, row_ids, prescribed["supports"])
            runtime.weights = runtime.weights.index_copy(0, row_ids, prescribed["weights"])
            runtime.neighbor_valid = runtime.neighbor_valid.index_copy(0, row_ids, prescribed["valid"])
            # Keep real candidate/scorer work and counters, while replacing only
            # the relation used by update_states/readout with the teacher graph.
            log_g = _retention_log_g(arm, prescribed["c"], initialized_before, row_ids,
                                     runtime.batch_size, runtime.n_nodes, model.beta)
            runtime.pending_log_g = pending_before.index_add(0, row_ids, log_g)
            audit = dict(candidate_audit)
            audit["supports"] = prescribed["supports"]
            audit["weights"] = prescribed["weights"]
            audit["valid"] = prescribed["valid"]
            diagnostics.append({"step": int(runtime.step), "row_ids": row_ids.detach().clone(),
                                "candidate_c": actual_c.detach().clone(),
                                "teacher_c": prescribed["c"],
                                "candidate_loss": candidate_loss.detach().clone(),
                                "prescribed_log_g": log_g.detach().clone(),
                                "initialized_before": initialized_before.detach().clone(),
                                "matched_supports": runtime.neighbors[row_ids].detach().clone(),
                                "matched_weights": runtime.weights[row_ids].detach().clone(),
                                "matched_valid": runtime.neighbor_valid[row_ids].detach().clone(),
                                "state_ids": prescribed["state_ids"]})
            return prescribed["c"], candidate_loss, audit
        return wrapped

    with _replace_refresh_graph(model, factory):
        result = model(**inputs, policy=policy, return_trace=True)
    if cursor != len(teacher_rows):
        raise RuntimeError("Arm did not consume every teacher graph step")
    for step, (trace, expected) in enumerate(zip(result["trace"], teacher_rows)):
        if not torch.equal(torch.as_tensor(trace["graph_ids"], device=expected["row_ids"].device), expected["row_ids"]):
            raise RuntimeError(f"Arm graph IDs differ at step {step}")
        if not torch.equal(torch.as_tensor(trace["state_ids"], device=expected["state_ids"].device), expected["state_ids"]):
            raise RuntimeError(f"Arm state IDs differ at step {step}")
        if not torch.equal(torch.as_tensor(trace["supports"], device=expected["supports"].device), expected["supports"]):
            raise RuntimeError(f"Arm support IDs differ at step {step}")
    return result, diagnostics


def run_matched_retention_probe(model, *, x, mask, age, raw, policy, event_priors=None,
                                descriptor=EXPECTED_DESCRIPTOR, arms=ARMS):
    """Run same-weight full-policy teacher plus prescribed-graph retention arms.

    Only causal model inputs are accepted. There are deliberately no targets,
    oracle matrices, dataset paths, checkpoints, or split arguments.
    """
    _validate(model, policy, descriptor, arms)
    inputs = {"x": x, "mask": mask, "age": age, "raw": raw, "event_priors": event_priors}
    if x.ndim != 4 or x.shape[2] != model.config.n_nodes or x.shape[1] != model.config.history:
        raise ValueError("Toy/probe input shape does not match the fixed V4 cohort/history")
    before_hash = parameter_sha256(model)
    initial_training = model.training
    teacher, teacher_rows = _run_teacher(model, inputs, policy)
    results = {"teacher": teacher, "teacher_rows": teacher_rows, "arms": {},
               "parameter_sha256_before": before_hash,
               "teacher_candidate_work_is_diagnostic_only": True,
               "timing_claim": "none; teacher and discarded candidate/scorer work add diagnostic overhead"}
    for arm in arms:
        result, diagnostics = _run_arm(model, inputs, policy, teacher_rows, arm)
        results["arms"][arm] = {"result": result, "candidate_diagnostics": diagnostics,
                                "candidate_graph_loss_is_discarded_scorer_diagnostic": True}
    after_hash = parameter_sha256(model)
    if after_hash != before_hash or model.training != initial_training:
        raise RuntimeError("Probe changed model parameters or mode")
    results["parameter_sha256_after"] = after_hash
    informative = results["arms"].get("informative")
    if informative is not None:
        _assert_informative_matches(teacher, informative["result"])
    results["informative_teacher_parity"] = True if informative is not None else None
    results["informative_teacher_parity_status"] = "passed" if informative is not None else "not-executed"
    results["all_required_retention_arms_executed"] = set(arms) == set(ARMS)
    return results


def _assert_close(name, left, right, atol, rtol):
    if torch.is_tensor(left):
        left = left.detach()
        right = right.detach()
        if not torch.allclose(left, right, atol=atol, rtol=rtol):
            diff = float((left-right).abs().max().cpu()) if left.numel() else 0.0
            raise RuntimeError(f"Informative replay differs for {name}: max_abs={diff}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"Informative replay keys differ for {name}")
        for key in left:
            _assert_close(f"{name}.{key}", left[key], right[key], atol, rtol)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError(f"Informative replay length differs for {name}")
        for i, (a, b) in enumerate(zip(left, right)):
            _assert_close(f"{name}[{i}]", a, b, atol, rtol)
    elif left is None or right is None:
        if left is not right:
            raise RuntimeError(f"Informative replay nullability differs for {name}")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol):
            raise RuntimeError(f"Informative replay differs for {name}: {left} vs {right}")
    elif left != right:
        raise RuntimeError(f"Informative replay differs for {name}: {left!r} vs {right!r}")


def _assert_informative_matches(teacher, replay):
    _assert_close("predictions", teacher["predictions"], replay["predictions"], 1e-6, 1e-6)
    _assert_close("graph_loss", teacher["graph_loss"], replay["graph_loss"], 1e-6, 1e-6)
    _assert_close("final_u", teacher["runtime"].u, replay["runtime"].u, 1e-5, 1e-6)
    for name in ("spike_pending", "pending_log_g"):
        _assert_close(name, getattr(teacher["runtime"], name), getattr(replay["runtime"], name), 1e-5, 1e-6)
    for index, (left, right) in enumerate(zip(teacher["trace"], replay["trace"])):
        for field in ("graph_ids", "state_ids", "supports"):
            _assert_close(f"trace[{index}].{field}", left[field], right[field], 0., 0.)
    _assert_close("counters", teacher["counters"], replay["counters"], 0., 0.)

