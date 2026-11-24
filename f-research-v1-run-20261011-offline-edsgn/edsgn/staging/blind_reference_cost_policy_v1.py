"""Blind reference cost fitter for staged periodic/random scheduler controls.

This helper patches only per-instance methods and does not alter dataclass
fields or serialized policy contents. It consumes offered masks and mandatory
masks, but never scores, q/r values, or other adaptive quantities.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import types

import numpy as np
import torch

RULE_VERSION = "blind-reference-cost-v1"
FREEZE_SOURCE_KEY = "blind_reference_cost_policy_source_sha256"
FREEZE_RULE_KEY = "blind_reference_cost_policy_rule_version"


def _source_sha256():
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def descriptor():
    return {
        "schema": "edsgn.blind-reference-cost-policy.v1",
        "rule_version": RULE_VERSION,
        "source_path": Path(__file__).resolve().as_posix(),
        "source_sha256": _source_sha256(),
        "priority_rule": {
            "periodic": "priority=(node-step*max(ceil(action_fraction*N),1))%N; lower first",
            "random": "torch.rand node priorities seeded seed+step*1009+action_salt; higher first",
            "action_salts": {"graph": 11, "state": 37},
            "merge": "normalized offered rank / quota, then graph before state, then node id",
            "budget": "same Policy.incremental_cost interpolation and cost_table.base_ms; mandatory actions retained",
        },
    }


def freeze_extra_fields():
    d = descriptor()
    return {FREEZE_SOURCE_KEY: d["source_sha256"], FREEZE_RULE_KEY: RULE_VERSION}


def validate_freeze_extra(extra):
    """Require exact helper source and rule binding before frozen controls are used."""
    if not isinstance(extra, dict):
        raise ValueError("frozen extra must contain the blind cost policy binding")
    current = _source_sha256()
    if extra.get(FREEZE_SOURCE_KEY) != current:
        raise ValueError("blind reference cost policy source hash missing or stale")
    if extra.get(FREEZE_RULE_KEY) != RULE_VERSION:
        raise ValueError("blind reference cost policy rule version missing or stale")
    return {"source_sha256": current, "rule_version": RULE_VERSION}


def _recording_select(policy, scores, mandatory, fraction, threshold, step, salt):
    result = policy._blind_original_select(scores, mandatory, fraction, threshold, step, salt)
    action = "graph" if int(salt) == 11 else "state" if int(salt) == 37 else None
    if action is not None:
        policy._blind_step_by_action[action] = int(step)
    return result


def _priority(policy, n, step, action, device):
    quota = max(int(np.ceil((policy.graph_fraction if action == "graph" else policy.state_fraction) * n)), 1)
    node = torch.arange(n, dtype=torch.int64, device=device)
    if policy.kind == "periodic":
        return (node - int(step) * quota) % n, False
    salt = 11 if action == "graph" else 37
    gen = torch.Generator(device=device).manual_seed(int(policy.seed) + int(step) * 1009 + salt)
    return torch.rand(n, generator=gen, device=device), True


def _blind_fit_cost_budget(policy, g, s, mg, ms, sg, ss):
    """Fit optional offers with data-independent reference order."""
    if policy.kind not in {"periodic", "random"}:
        raise ValueError("blind cost fitter is only for periodic/random controls")
    if policy.runtime_budget_ms is None or not policy.cost_table:
        raise ValueError("blind cost fitting requires the accepted B1 cost table and budget")
    B, N = g.shape
    if B != 1:
        raise ValueError("blind reference cost fitting requires batch size one")
    packed = torch.cat([g.reshape(-1).to(torch.float32), s.reshape(-1).to(torch.float32),
                        mg.reshape(-1).to(torch.float32), ms.reshape(-1).to(torch.float32)]).detach().cpu()
    width = B * N
    def take_bool(offset):
        return packed[offset:offset + width].reshape(B, N).numpy() > .5
    offered_graph, offered_state = take_bool(0), take_bool(width)
    mandatory_graph, mandatory_state = take_bool(width * 2), take_bool(width * 3)
    selected_graph = mandatory_graph.copy()
    selected_state = mandatory_state.copy()
    graph_count = int(mandatory_graph.sum())
    state_count = int(mandatory_state.sum())
    budget = max(0.0, float(policy.runtime_budget_ms) - float(policy.cost_table.get("base_ms", 0.0)))
    quotas = {"graph": max(int(np.ceil(policy.graph_fraction * N)), 1),
              "state": max(int(np.ceil(policy.state_fraction * N)), 1)}
    offers = {"graph": offered_graph, "state": offered_state}
    musts = {"graph": mandatory_graph, "state": mandatory_state}
    candidates = []
    for action in ("graph", "state"):
        ids = np.flatnonzero(offers[action][0] & ~musts[action][0])
        step = policy._blind_step_by_action.get(action, 0)
        priorities, descending = _priority(policy, N, step, action, g.device)
        priority_np = priorities.detach().cpu().numpy()
        ordered = sorted(ids.tolist(), key=lambda node: ((-priority_np[node]) if descending else priority_np[node], node))
        quota = quotas[action]
        for rank, node in enumerate(ordered):
            candidates.append((rank / quota, 0 if action == "graph" else 1, node, action))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    for _, _, node, action in candidates:
        next_graph = graph_count + (action == "graph")
        next_state = state_count + (action == "state")
        cost = (policy.incremental_cost("graph", next_graph)
                + policy.incremental_cost("state", next_state))
        if cost <= budget:
            if action == "graph":
                selected_graph[0, node] = True
                graph_count = next_graph
            else:
                selected_state[0, node] = True
                state_count = next_state
    result = torch.as_tensor(np.stack([selected_graph, selected_state]), dtype=torch.bool, device=g.device)
    return result[0], result[1]


def apply_blind_reference_cost(policy):
    """Bind blind fit behavior to one periodic/random Policy instance in place."""
    from edsgn.scheduling import Policy
    if not isinstance(policy, Policy):
        raise TypeError("policy must be edsgn.scheduling.Policy")
    if policy.kind not in {"periodic", "random"}:
        raise ValueError("only periodic/random controls can use blind reference costs")
    if getattr(policy, "_blind_reference_cost_rule_version", None) == RULE_VERSION:
        return policy
    if hasattr(policy, "_blind_reference_cost_rule_version"):
        raise RuntimeError("policy already has a different blind cost binding")
    policy._blind_original_select = policy._select
    policy._blind_step_by_action = {}
    policy._select = types.MethodType(_recording_select, policy)
    policy._fit_cost_budget = types.MethodType(_blind_fit_cost_budget, policy)
    policy._blind_reference_cost_rule_version = RULE_VERSION
    return policy
