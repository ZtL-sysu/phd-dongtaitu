import copy

import numpy as np
import pytest
import torch

from edsgn.model import ForecastModel, ModelConfig
from edsgn.scheduling import Policy


def _monitor(batch, nodes, *, graph_age=None, state_age=None, q=None, r=None, change=None):
    dev = torch.device("cpu")
    z = lambda: torch.zeros(batch, nodes, device=dev)
    return {
        "graph_age": z() if graph_age is None else torch.tensor(graph_age, dtype=torch.float32),
        "state_age": z() if state_age is None else torch.tensor(state_age, dtype=torch.float32),
        "q": z() if q is None else torch.tensor(q, dtype=torch.float32),
        "r": z() if r is None else torch.tensor(r, dtype=torch.float32),
        "r_smooth": z() if r is None else torch.tensor(r, dtype=torch.float32),
        "quality_change": z() if change is None else torch.tensor(change, dtype=torch.float32),
    }


def _runtime(batch, nodes, step=2):
    model = ForecastModel(ModelConfig(n_nodes=nodes, n_features=1, hmax=1,
                                     d_total=8, k=min(1, nodes-1),
                                     candidate_cap=min(1, nodes-1)))
    runtime = model.init_runtime(batch)
    runtime.step = step
    return runtime


def test_random_policy_has_batch_slot_invariant_priority_and_predictions():
    torch.set_num_threads(2)
    torch.manual_seed(413)
    n, length = 5, 4
    model = ForecastModel(ModelConfig(n_nodes=n, n_features=1, history=length,
                                      hmax=2, variant="V4", d_total=16,
                                      k=2, candidate_cap=4)).eval()
    a = torch.randn(1, length, n, 1)
    b = torch.randn(1, length, n, 1) + .3
    ma = torch.rand_like(a) > .2
    mb = torch.rand_like(b) > .25
    ma[:, :, 0] = True; mb[:, :, 1] = True
    policy = Policy(kind="random", graph_fraction=.4, state_fraction=.4,
                    max_graph_age=50, max_state_age=50, audit_interval=0,
                    seed=71, force_initialization=False)

    solo = model(a, mask=ma, policy=copy.deepcopy(policy), return_trace=True)
    joined = model(torch.cat([a, b]), mask=torch.cat([ma, mb]),
                   policy=copy.deepcopy(policy), return_trace=True)
    reversed_batch = model(torch.cat([b, a]), mask=torch.cat([mb, ma]),
                           policy=copy.deepcopy(policy), return_trace=True)
    assert torch.allclose(solo["predictions"][0], joined["predictions"][0], atol=1e-6, rtol=1e-5)
    assert torch.allclose(solo["predictions"][0], reversed_batch["predictions"][1], atol=1e-6, rtol=1e-5)
    for t, solo_row in enumerate(solo["trace"]):
        assert solo_row["graph_ids"] == [i for i in joined["trace"][t]["graph_ids"] if i < n]
        assert solo_row["state_ids"] == [i for i in joined["trace"][t]["state_ids"] if i < n]
        rev_row = reversed_batch["trace"][t]
        assert solo_row["graph_ids"] == [i-n for i in rev_row["graph_ids"] if i >= n]
        assert solo_row["state_ids"] == [i-n for i in rev_row["state_ids"] if i >= n]


def test_cost_table_scope_and_runtime_budget_fail_closed():
    valid = {"batch_size": 1, "graph": [[1, 1.], [4, 5.]], "state": [[1, 1.], [4, 5.]]}
    policy = Policy(kind="adaptive", cost_table=valid)
    g, s, _ = policy.plan(_monitor(1, 4), _runtime(1, 4))
    assert g.shape == s.shape == (1, 4)

    with pytest.raises(ValueError, match="batch_size=1"):
        Policy(kind="adaptive", cost_table=valid).plan(_monitor(2, 4), _runtime(2, 4))
    with pytest.raises(ValueError, match="explicit batch_size=1 calibrated cost_table"):
        Policy(kind="adaptive", runtime_budget_ms=3.).plan(_monitor(1, 4), _runtime(1, 4))
    with pytest.raises(ValueError, match="explicit batch_size=1 calibrated cost_table"):
        Policy(kind="adaptive", cost_table={"graph": [[1, 1.]], "state": [[1, 1.]]},
               runtime_budget_ms=3.).plan(_monitor(1, 4), _runtime(1, 4))
    with pytest.raises(ValueError, match="batch_size=1"):
        Policy(kind="adaptive", cost_table=valid, runtime_budget_ms=3.).plan(
            _monitor(2, 4), _runtime(2, 4))

    # Quota-only independent batched windows remain supported.
    g, s, _ = Policy(kind="random", graph_fraction=.5, state_fraction=.5,
                     audit_interval=0).plan(_monitor(2, 4), _runtime(2, 4))
    assert g.shape == s.shape == (2, 4)


def _old_cpu_budget_reference(policy, g, s, mg, ms, sg, ss):
    budget = max(0., policy.runtime_budget_ms - float(policy.cost_table.get("base_ms", 0.)))
    offered_graph, offered_state = g.clone(), s.clone()
    selected_graph, selected_state = mg.clone(), ms.clone()
    optional = []
    for action, mask, must, scores, offered in (
        ("graph", selected_graph, mg, sg, offered_graph),
        ("state", selected_state, ms, ss, offered_state),
    ):
        eligible = (~must) & offered
        for b, n in eligible.nonzero().tolist():
            group = policy.incremental_cost(action, max(1, int(np.ceil(mask.shape[1] / 4))))
            optional.append((float(scores[b, n]) / max(group, 1e-9), action, b, n))
    optional.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    for _, action, b, n in optional:
        target = selected_graph if action == "graph" else selected_state
        target[b, n] = True
        cost = policy.incremental_cost("graph", int(selected_graph.sum())) + policy.incremental_cost("state", int(selected_state.sum()))
        if cost > budget:
            target[b, n] = False
    return selected_graph, selected_state


@pytest.mark.parametrize("case_id,table,budget", [
    (0, {"batch_size": 1, "base_ms": .2, "graph": [[1, .5], [2, 1.4], [4, 7.]],
         "state": [[1, .3], [2, .7], [4, 3.2]]}, 5.4),
    (1, {"batch_size": 1, "base_ms": 3., "graph": [[1, 1.], [4, 10.]],
         "state": [[1, .5], [4, 5.]]}, 1.),  # mandatory work alone exceeds budget
    (2, {"batch_size": 1, "base_ms": 0., "graph": [[1, 1.], [4, 10.]],
         "state": [[1, .5], [4, 5.]]}, 0.),
    (3, {"batch_size": 1, "base_ms": .15, "graph": [[1, .1], [4, 2.]],
         "state": [[1, .2], [4, 4.]]}, 3.1),
])
def test_cpu_cost_selection_matches_original_candidate_reference(case_id, table, budget):
    torch.set_num_threads(2)
    batch, nodes = 1, 8
    masks = [
        ([[1, 1, 1, 1, 0, 1, 0, 1]], [[1, 1, 0, 1, 1, 1, 1, 0]],
         [[1, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 1, 0, 0, 0]]),
        ([[0, 1, 0, 1, 1, 0, 1, 0]], [[1, 0, 1, 0, 1, 0, 1, 1]],
         [[0, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 1]]),
        ([[1, 0, 1, 0, 1, 1, 0, 1]], [[0, 1, 1, 1, 0, 1, 0, 1]],
         [[1, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 1, 0, 0, 1, 0, 0]]),
        ([[1, 1, 1, 1, 1, 1, 1, 1]], [[1, 1, 1, 1, 1, 1, 1, 1]],
         [[1, 1, 1, 1, 0, 0, 0, 0]], [[0, 0, 0, 0, 1, 1, 1, 1]]),
    ][case_id]
    offered_g, offered_s, must_g, must_s = [torch.tensor(x, dtype=torch.bool) for x in masks]
    graph_scores = torch.tensor([[.3, .7, .2, .9, .8, .1, .6, .5]])
    state_scores = torch.tensor([[.5, .4, .95, .25, .1, .8, .65, .7]])
    graph_scores = torch.roll(graph_scores, shifts=case_id, dims=1)
    state_scores = torch.roll(state_scores, shifts=-case_id, dims=1)
    policy = Policy(kind="adaptive", cost_table=table, runtime_budget_ms=budget)
    expected_g, expected_s = _old_cpu_budget_reference(policy, offered_g, offered_s,
                                                        must_g, must_s, graph_scores, state_scores)
    actual_g, actual_s = policy._fit_cost_budget(offered_g, offered_s, must_g, must_s,
                                                  graph_scores, state_scores)
    assert torch.equal(actual_g, expected_g)
    assert torch.equal(actual_s, expected_s)
