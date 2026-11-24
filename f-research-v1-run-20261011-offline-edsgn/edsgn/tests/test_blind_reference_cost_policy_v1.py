from dataclasses import asdict
from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from edsgn.scheduling import Policy
from edsgn.engine import policy_digest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "edsgn/staging/blind_reference_cost_policy_v1.py"
spec = importlib.util.spec_from_file_location("blind_reference_cost_policy_v1", MODULE)
blind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blind)


def _cost_table():
    points = [[1, 1.0], [4, 4.0], [8, 8.0]]
    return {"graph": points, "state": points, "base_ms": 0.25,
            "batch_size": 1, "units": "toy measured envelope"}


def _plan(policy, step, q, rs, r, quality=None):
    n = len(q)
    runtime = SimpleNamespace(graph_age=torch.zeros((1, n), dtype=torch.long), step=step)
    monitor = {
        "graph_age": torch.zeros((1, n)),
        "state_age": torch.zeros((1, n)),
        "q": torch.as_tensor(q, dtype=torch.float32).reshape(1, n),
        "r_smooth": torch.as_tensor(rs, dtype=torch.float32).reshape(1, n),
        "r": torch.as_tensor(r, dtype=torch.float32).reshape(1, n),
        "quality_change": torch.zeros((1, n)) if quality is None else torch.as_tensor(quality).reshape(1, n),
    }
    return policy.plan(monitor, runtime)


@pytest.mark.parametrize("kind", ["periodic", "random"])
def test_swap_adaptive_scores_has_no_effect_on_blind_control(kind):
    kwargs = dict(kind=kind, graph_fraction=.75, state_fraction=.75,
                  max_graph_age=8, max_state_age=8, audit_interval=0,
                  seed=7, force_initialization=False, cost_table=_cost_table(),
                  runtime_budget_ms=4.25)
    p = blind.apply_blind_reference_cost(Policy(**kwargs))
    a = _plan(p, 2, [0, 1, 2, 3, 4, 5], [9, 8, 7, 6, 5, 4], [1, 2, 3, 4, 5, 6])
    b = _plan(p, 2, [5, 4, 3, 2, 1, 0], [4, 5, 6, 7, 8, 9], [6, 5, 4, 3, 2, 1])
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[2]["estimated_cost"] == b[2]["estimated_cost"]


def test_original_score_cost_fitter_changes_under_score_swap_demonstrating_confound():
    p = Policy(kind="periodic", graph_fraction=1, state_fraction=1,
               cost_table=_cost_table(), runtime_budget_ms=3.25)
    offered = torch.ones((1, 4), dtype=torch.bool)
    mandatory = torch.zeros_like(offered)
    graph_hi = torch.tensor([[9., 8., 1., 0.]])
    state_hi = torch.tensor([[0., 1., 8., 9.]])
    a = p._fit_cost_budget(offered, offered, mandatory, mandatory, graph_hi, state_hi)
    b = p._fit_cost_budget(offered, offered, mandatory, mandatory, state_hi, graph_hi)
    assert not torch.equal(a[0], b[0]) or not torch.equal(a[1], b[1])


def test_mandatory_actions_and_b1_envelope_survive_optional_budget_fallback():
    p = blind.apply_blind_reference_cost(Policy(kind="periodic", graph_fraction=.25,
        state_fraction=.25, max_graph_age=1, max_state_age=1, audit_interval=0,
        force_initialization=False, cost_table=_cost_table(), runtime_budget_ms=.5))
    n = 4
    runtime = SimpleNamespace(graph_age=torch.full((1,n), 5), step=3)
    monitor = {"graph_age": torch.full((1,n), 5), "state_age": torch.full((1,n), 5),
        "q": torch.zeros((1,n)), "r_smooth": torch.zeros((1,n)), "r": torch.zeros((1,n)),
        "quality_change": torch.zeros((1,n))}
    g, s, counters = p.plan(monitor, runtime)
    assert g.all() and s.all()  # mandatory overrun is retained
    assert counters["estimated_mandatory_cost"] > p.runtime_budget_ms
    assert counters["runtime_budget_overrun_ms"] > 0


def test_equal_normalized_rank_ties_graph_first_then_uses_same_cost_envelope():
    table={"graph":[[1,2.0],[4,8.0]],"state":[[1,1.0],[4,4.0]],
           "base_ms":.2,"batch_size":1}
    p=blind.apply_blind_reference_cost(Policy(kind="periodic",graph_fraction=.25,
        state_fraction=.25,max_graph_age=20,max_state_age=20,audit_interval=0,
        force_initialization=False,cost_table=table,runtime_budget_ms=1.2))
    n=4;z=torch.zeros((1,n));offers=torch.ones((1,n),dtype=torch.bool)
    mandatory=torch.zeros_like(offers)
    p._blind_step_by_action.update(graph=0,state=0)
    g,s=p._fit_cost_budget(offers,offers,mandatory,mandatory,z,z)
    # Both first candidates have normalized rank 0; graph is considered first,
    # but its measured incremental cost exceeds the base-subtracted envelope.
    assert not g.any()
    assert s.sum().item()==1


def test_periodic_rotation_and_random_repeatability_with_independent_action_salts():
    periodic = blind.apply_blind_reference_cost(Policy(kind="periodic", graph_fraction=.5,
        state_fraction=.5, max_graph_age=20, max_state_age=20, audit_interval=0,
        force_initialization=False, cost_table=_cost_table(), runtime_budget_ms=10))
    a0 = _plan(periodic, 0, [0]*10, [0]*10, [0]*10)
    a1 = _plan(periodic, 1, [0]*10, [0]*10, [0]*10)
    assert not torch.equal(a0[0], a1[0]) or not torch.equal(a0[1], a1[1])
    random = blind.apply_blind_reference_cost(Policy(kind="random", graph_fraction=.5,
        state_fraction=.5, max_graph_age=20, max_state_age=20, audit_interval=0,
        seed=13, force_initialization=False, cost_table=_cost_table(), runtime_budget_ms=10))
    r1 = _plan(random, 4, [0]*20, [0]*20, [0]*20)
    r2 = _plan(random, 4, [9]*20, [8]*20, [7]*20)
    assert torch.equal(r1[0], r2[0]) and torch.equal(r1[1], r2[1])
    assert not torch.equal(r1[0], r1[1])


def test_helper_is_idempotent_and_does_not_change_dataclass_or_digest_binding():
    p = Policy(kind="random", graph_fraction=.4, state_fraction=.6, seed=9,
               cost_table=_cost_table(), runtime_budget_ms=4)
    before = asdict(p)
    digest_before = policy_digest(p)
    assert blind.apply_blind_reference_cost(p) is p
    assert blind.apply_blind_reference_cost(p) is p
    assert asdict(p) == before
    assert policy_digest(p) == digest_before
    extra = blind.freeze_extra_fields()
    assert blind.validate_freeze_extra(extra)["rule_version"] == blind.RULE_VERSION
    for bad in ({}, {**extra, blind.FREEZE_RULE_KEY: "other"},
                {**extra, blind.FREEZE_SOURCE_KEY: "0"*64}):
        with pytest.raises(ValueError):
            blind.validate_freeze_extra(bad)


def test_adaptive_and_full_policies_are_untouched_and_rejected():
    for kind in ("adaptive", "full"):
        p = Policy(kind=kind)
        before = asdict(p)
        with pytest.raises(ValueError, match="periodic/random"):
            blind.apply_blind_reference_cost(p)
        assert asdict(p) == before
