"""Toy-only case-signal sensitivity kernel for frozen V4 causal replay.

No CLI, filesystem, target, oracle, checkpoint, or dataset interface is exposed.
The kernel is not integrated into the model package or any execution gate.
"""
from __future__ import annotations

from contextlib import contextmanager
import torch

from h2_matched_graph_retention_v2 import parameter_sha256, _run_teacher, _assert_informative_matches


def _validate(model, inputs, focal_node, neighbor_node, policy):
    from edsgn.scheduling import Policy
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("case-signal replay requires eval mode and torch.no_grad()")
    if model.config.variant != "V4" or not model.spiking or not model.retention:
        raise ValueError("case-signal kernel requires the frozen spiking retention V4 model")
    if model.config.retention_mode != "informative" or model.config.k < 1:
        raise ValueError("teacher must use informative retention and positive graph k")
    if not isinstance(policy, Policy) or policy.kind != "full" or policy.cost_table or policy.runtime_budget_ms is not None:
        raise ValueError("case-signal replay requires the full reference policy")
    if set(inputs) - {"x", "mask", "age", "raw", "event_priors"}:
        raise ValueError("only causal inputs are accepted; labels/oracle are forbidden")
    x = inputs.get("x")
    if not torch.is_tensor(x) or x.ndim != 4 or x.shape[2] != model.config.n_nodes:
        raise ValueError("x must be [B,L,N,F] and match the fixed model cohort")
    if not 0 <= focal_node < model.config.n_nodes or not 0 <= neighbor_node < model.config.n_nodes:
        raise ValueError("focal and selected neighbor IDs must be valid nodes")
    if focal_node == neighbor_node:
        raise ValueError("selected relationship must be directed and nonself")


@contextmanager
def _temporary_method(obj, name, replacement):
    had = name in obj.__dict__
    old = obj.__dict__.get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if had:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def filter_candidate_rows(candidates, frequencies, row_ids, *, n_nodes, focal_node, neighbor_node):
    """Replace the removed edge with a self sentinel; never append/refill candidates.

    The sentinel occupies the removed slot so [rows, K] and the frozen cap remain
    unchanged. The refresh wrapper gives sentinels a very low score and marks any
    selected sentinel invalid before c/weights are recomputed.
    """
    if candidates.ndim != 2 or frequencies.shape != candidates.shape or row_ids.ndim != 1:
        raise ValueError("candidate/frequency/row shapes are inconsistent")
    result = candidates.clone()
    freq = frequencies.clone()
    targeted = (row_ids % n_nodes == focal_node) & (result == neighbor_node).any(dim=-1)
    if targeted.any():
        match = result == neighbor_node
        result = torch.where(match & targeted[:, None], row_ids[:, None] % n_nodes, result)
        freq = torch.where(match & targeted[:, None], torch.zeros_like(freq), freq)
    return result, freq, targeted


def remove_neighbor_support(ids, weights, valid, row_ids, *, n_nodes, focal_node):
    """Clear focal self-sentinel slots and renormalize remaining valid weights."""
    if ids.shape != weights.shape or ids.shape != valid.shape:
        raise ValueError("support IDs, weights, and valid mask must have equal shapes")
    blocked = (row_ids % n_nodes == focal_node)[:, None] & (ids == focal_node)
    out_valid = valid.bool() & ~blocked
    out_weights = torch.where(out_valid, weights, torch.zeros_like(weights))
    total = out_weights.sum(-1, keepdim=True)
    out_weights = torch.where(total > 0, out_weights / total.clamp_min(torch.finfo(weights.dtype).tiny),
                              torch.zeros_like(out_weights))
    return out_weights, out_valid, blocked


def _c_for_support(model, old_ids, old_weights, old_valid, new_ids, new_weights, new_valid, initialized):
    from edsgn.model import aligned_change
    return aligned_change(old_ids, old_weights, old_valid, new_ids, new_weights, new_valid, initialized)


def _relationship_replay(model, inputs, policy, focal_node, neighbor_node):
    n_nodes = model.config.n_nodes
    candidate_diagnostics = []
    original_candidates = model._candidates
    original_refresh = model.refresh_graph
    sentinel_slots = {"mask": None}

    def candidates(ids, runtime):
        candidate_ids, freq = original_candidates(ids, runtime)
        candidate_ids, freq, targeted = filter_candidate_rows(candidate_ids, freq, ids,
            n_nodes=n_nodes, focal_node=focal_node, neighbor_node=neighbor_node)
        sentinel_slots["mask"] = (candidate_ids == (ids % n_nodes)[:, None]) & targeted[:, None]
        candidate_diagnostics.append({"row_ids": ids.detach().clone(), "candidate_ids": candidate_ids.detach().clone(),
                                     "targeted": targeted.detach().clone(), "cap": candidate_ids.shape[-1]})
        return candidate_ids, freq

    def refresh_factory(original):
        def wrapped(observation, runtime, row_ids, summary, monitor):
            if not row_ids.numel():
                return original(observation, runtime, row_ids, summary, monitor)
            old_ids = runtime.neighbors[row_ids].detach().clone()
            old_weights = runtime.weights[row_ids].detach().clone()
            old_valid = runtime.neighbor_valid[row_ids].detach().clone()
            initialized = runtime.initialized[row_ids].detach().clone()
            pending_before = runtime.pending_log_g.detach().clone()
            c, _, audit = original(observation, runtime, row_ids, summary, monitor)
            if audit is None:
                return c, runtime.u.sum() * 0, audit
            ids = runtime.neighbors[row_ids]
            raw_weights = runtime.weights[row_ids]
            valid = runtime.neighbor_valid[row_ids]
            new_weights, new_valid, blocked = remove_neighbor_support(ids, raw_weights, valid, row_ids,
                n_nodes=n_nodes, focal_node=focal_node)
            runtime.weights = runtime.weights.index_copy(0, row_ids, new_weights)
            runtime.neighbor_valid = runtime.neighbor_valid.index_copy(0, row_ids, new_valid)
            new_ids = runtime.neighbors[row_ids]
            c = _c_for_support(model, old_ids, old_weights, old_valid, new_ids, new_weights,
                               new_valid, initialized)
            q = monitor["q"].reshape(-1)[row_ids]
            r = monitor["r"].reshape(-1)[row_ids]
            reliability = .1 + .9 / (1 + q * r)
            temporal = (2 * c * reliability.detach()).mean() * model.config.lambda_temporal
            logg = torch.where(initialized, -model.beta * c, torch.zeros_like(c))
            runtime.pending_log_g = pending_before.index_add(0, row_ids, logg)
            audit = dict(audit)
            audit.update(supports=new_ids, weights=new_weights, valid=new_valid)
            candidate_diagnostics.append({"refresh_row_ids": row_ids.detach().clone(),
                "step": int(runtime.step),
                "removed_selected_edge": blocked.detach().clone(), "actual_c": c.detach().clone(),
                "old_support_ids": old_ids.detach().clone(), "old_support_weights": old_weights.detach().clone(),
                "old_support_valid": old_valid.detach().clone(), "initialized_before": initialized.detach().clone(),
                "new_support_ids": new_ids.detach().clone(), "new_support_weights": new_weights.detach().clone(),
                "new_support_valid": new_valid.detach().clone(),
                "valid_support_count": new_valid.sum(-1).detach().clone(),
                "support_weight_sum": new_weights.sum(-1).detach().clone(),
                "candidate_cap": audit["candidates"].shape[-1]})
            return c, temporal, audit
        return wrapped

    scorer = model.graph_scorer
    def mark_sentinel(module, args):
        features = args[0].clone()
        mask = sentinel_slots["mask"]
        if mask is not None:
            if mask.shape != features.shape[:-1]:
                raise RuntimeError("candidate sentinel/scorer rows are misaligned")
            features[..., -1] = torch.where(mask, torch.full_like(features[..., -1], -1.), features[..., -1])
        return (features, *args[1:])
    def suppress_marked_sentinel(module, args, output):
        sentinel = args[0][..., -1] < 0
        return output.masked_fill(sentinel.unsqueeze(-1), -1e30)
    prehandle = scorer.register_forward_pre_hook(mark_sentinel)
    handle = scorer.register_forward_hook(suppress_marked_sentinel)
    try:
        with _temporary_method(model, "_candidates", candidates), _temporary_method(model, "refresh_graph", refresh_factory(original_refresh)):
            result = model(**inputs, policy=policy, return_trace=True)
    finally:
        handle.remove()
        prehandle.remove()
    result["case_diagnostics"] = candidate_diagnostics
    return result


def _retention_replay(model, inputs, policy, teacher_rows, focal_node, *, remove_focal_signal):
    n_nodes = model.config.n_nodes
    cursor = 0
    records = []
    def factory(original):
        def wrapped(observation, runtime, row_ids, summary, monitor):
            nonlocal cursor
            if cursor >= len(teacher_rows):
                raise RuntimeError("retention arm refreshed beyond teacher trace")
            teacher = teacher_rows[cursor]; cursor += 1
            if int(runtime.step) != teacher["step"] or not torch.equal(row_ids, teacher["row_ids"]):
                raise RuntimeError("teacher/retention graph action IDs differ")
            initialized = runtime.initialized[row_ids].detach().clone()
            pending_before = runtime.pending_log_g.detach().clone()
            if not torch.equal(initialized, teacher["initialized_before"]):
                raise RuntimeError("retention replay graph initialization differs from teacher")
            if not torch.equal(pending_before, teacher["pending_before"]):
                raise RuntimeError("retention replay deferred g differs before consumption")
            c_actual, loss, audit = original(observation, runtime, row_ids, summary, monitor)
            if audit is None:
                raise RuntimeError("V4 full graph refresh lacks graph audit")
            # The teacher graph and c are prescribed at each refresh; only the
            # focal node's neighbor-channel retention multiplier may differ.
            runtime.neighbors = runtime.neighbors.index_copy(0, row_ids, teacher["supports"])
            runtime.weights = runtime.weights.index_copy(0, row_ids, teacher["weights"])
            runtime.neighbor_valid = runtime.neighbor_valid.index_copy(0, row_ids, teacher["valid"])
            logg = torch.where(initialized, -model.beta * teacher["c"], torch.zeros_like(teacher["c"]))
            focal_rows = row_ids % n_nodes == focal_node
            if remove_focal_signal:
                logg = torch.where(focal_rows, torch.zeros_like(logg), logg)
            runtime.pending_log_g = pending_before.index_add(0, row_ids, logg)
            audit = dict(audit); audit.update(supports=teacher["supports"], weights=teacher["weights"], valid=teacher["valid"])
            records.append({"step": int(runtime.step), "row_ids": row_ids.detach().clone(),
                            "teacher_c": teacher["c"], "actual_candidate_c": c_actual.detach().clone(),
                            "log_g": logg.detach().clone(), "focal_rows": focal_rows.detach().clone(),
                            "supports": runtime.neighbors[row_ids].detach().clone(),
                            "weights": runtime.weights[row_ids].detach().clone(),
                            "valid": runtime.neighbor_valid[row_ids].detach().clone()})
            return teacher["c"], loss, audit
        return wrapped
    with _temporary_method(model, "refresh_graph", factory(model.refresh_graph)):
        result = model(**inputs, policy=policy, return_trace=True)
    if cursor != len(teacher_rows):
        raise RuntimeError("retention arm did not consume all teacher graph steps")
    return result, records


def _assert_graph_actions(teacher, arm, *, compare_supports=True):
    if len(teacher["trace"]) != len(arm["trace"]):
        raise RuntimeError("teacher/arm trace lengths differ")
    for i, (left, right) in enumerate(zip(teacher["trace"], arm["trace"])):
        keys = ("graph_ids", "state_ids", "supports") if compare_supports else ("graph_ids", "state_ids")
        for key in keys:
            if not torch.equal(torch.as_tensor(left[key]), torch.as_tensor(right[key])):
                raise RuntimeError(f"teacher/arm {key} differs at step {i}")


def run_case_signal_sensitivity(model, *, x, mask, age, raw, policy, focal_node, neighbor_node, event_priors=None):
    """CPU/toy frozen V4 teacher plus relationship- and focal-retention-removal arms."""
    inputs = {"x": x, "mask": mask, "age": age, "raw": raw, "event_priors": event_priors}
    _validate(model, inputs, focal_node, neighbor_node, policy)
    before = parameter_sha256(model)
    teacher, teacher_rows = _run_teacher(model, inputs, policy)
    # First run an unmodified informative replay to prove the replay mechanism itself.
    informative, informative_rows = _retention_replay(model, inputs, policy, teacher_rows,
                                                        focal_node, remove_focal_signal=False)
    _assert_informative_matches(teacher, informative)
    _assert_graph_actions(teacher, informative)
    relationship = _relationship_replay(model, inputs, policy, focal_node, neighbor_node)
    _assert_graph_actions(teacher, relationship, compare_supports=False)
    retention, retention_rows = _retention_replay(model, inputs, policy, teacher_rows,
                                                   focal_node, remove_focal_signal=True)
    _assert_graph_actions(teacher, retention)
    after = parameter_sha256(model)
    if before != after:
        raise RuntimeError("case-signal kernel changed parameter bytes")
    # Retention arm must have g=1 exactly on focal graph rows and informative g elsewhere.
    for row in retention_rows:
        expected = row["log_g"].clone()
        if bool(row["focal_rows"].any()) and bool((row["log_g"][row["focal_rows"]] != 0).any()):
            raise RuntimeError("focal retention removal did not set g=1")
        nonfocal = ~row["focal_rows"]
        if nonfocal.any():
            target = torch.where(torch.ones_like(row["teacher_c"][nonfocal], dtype=torch.bool),
                                 -model.beta * row["teacher_c"][nonfocal], torch.zeros_like(row["teacher_c"][nonfocal]))
            if not torch.equal(row["log_g"][nonfocal], target):
                raise RuntimeError("nonfocal retention signal differs from informative teacher")
    return {"teacher": teacher, "informative_replay": informative,
            "relationship_removal": relationship,
            "focal_retention_removal": retention,
            "teacher_rows": teacher_rows, "retention_rows": retention_rows,
            "parameter_sha256_before": before, "parameter_sha256_after": after,
            "informative_teacher_parity": True,
            "retention_graph_and_actions_exact": True,
            "relationship_rule": "exclude selected directed neighbor; keep cap; no candidate refill; renormalize remaining valid support; empty support has zero weights",
            "retention_rule": "focal node neighbor-channel g=1 at teacher consumption times; local and all other node channels follow informative teacher",
            "claim_scope": "CPU/toy engineering semantics only; no scientific inference or full-domain execution"}
