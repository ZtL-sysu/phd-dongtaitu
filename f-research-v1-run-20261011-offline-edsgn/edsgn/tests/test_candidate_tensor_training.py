import copy

import numpy as np
import pytest
import torch

from edsgn.model import ForecastModel, ModelConfig
from edsgn.scheduling import Policy


VARIANTS = ("V1", "V2", "V3", "V4", "V9")


def _config(variant, tensor):
    return ModelConfig(
        n_nodes=7, n_features=2, hmax=3, history=6, variant=variant,
        d_total=16, k=2, candidate_cap=4, exploration_interval=2,
        summary_length=3, encoder_width=8, training_prior_cache=False,
        training_candidate_tensor=tensor,
    )


def _events(kind):
    packs = []
    for b, start in enumerate((100, 200)):
        user = (b + 1) % 7
        rows = np.asarray([
            [start + 0, user, (b + 3) % 7],
            [start + 2, user, (b + 5) % 7],
            [start + 4, user, (b + 4) % 7],
        ], dtype=np.int64)
        packs.append({"kind": kind, "records": rows,
                      "time_indices": np.arange(start, start + 6, dtype=np.int64)})
    return packs


def _state_fields(runtime):
    return ("neighbors", "weights", "neighbor_valid", "initialized", "u",
            "spike_pending", "pending_log_g", "graph_age", "state_age",
            "last_input", "last_input_valid", "previous_pred_1",
            "previous_pred_valid", "q", "r_smooth", "levels",
            "candidate_pool", "exploration_phase")


def _trace_without_wallclock(trace):
    return [{key: value for key, value in row.items() if key != "component_host_ms"}
            for row in trace]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("event_kind", ("email", "mooc"))
def test_training_tensor_candidate_path_matches_reference_full_forward_and_grads(variant, event_kind):
    torch.manual_seed(427)
    reference = ForecastModel(_config(variant, False)).cpu().train()
    tensorized = ForecastModel(_config(variant, True)).cpu().train()
    tensorized.load_state_dict(copy.deepcopy(reference.state_dict()))
    x0 = torch.randn(2, 6, 7, 2)
    mask = torch.rand_like(x0) > .12
    age = torch.rand_like(x0) * 3
    raw = x0 * 1.7 + .2
    policy = Policy()

    expected = reference(x0.clone().requires_grad_(True), mask, age, raw,
                         policy=policy, event_priors=_events(event_kind),
                         return_trace=True, return_intermediate=True)
    actual = tensorized(x0.clone().requires_grad_(True), mask, age, raw,
                        policy=policy, event_priors=_events(event_kind),
                        return_trace=True, return_intermediate=True)

    assert torch.equal(actual["predictions"], expected["predictions"])
    assert torch.equal(actual["graph_loss"], expected["graph_loss"])
    assert torch.equal(actual["intermediate"][-1], expected["intermediate"][-1])
    assert actual["counters"] == expected["counters"]
    assert _trace_without_wallclock(actual["trace"]) == _trace_without_wallclock(expected["trace"])
    assert actual["counters"]["candidate_prior_merge_checks"] > 0
    for name in _state_fields(actual["runtime"]):
        assert torch.equal(getattr(actual["runtime"], name), getattr(expected["runtime"], name)), name
    assert actual["runtime"].contacts == expected["runtime"].contacts
    assert actual["runtime"].user_history == expected["runtime"].user_history

    for output in (expected, actual):
        loss = output["predictions"].square().mean() + output["graph_loss"]
        loss.backward()
    p_ref = dict(reference.named_parameters())
    p_new = dict(tensorized.named_parameters())
    assert p_ref.keys() == p_new.keys()
    for name in p_ref:
        ga, gb = p_ref[name].grad, p_new[name].grad
        assert (ga is None) == (gb is None), name
        if ga is not None:
            assert torch.isfinite(ga).all(), name
            assert torch.isfinite(gb).all(), name
            assert torch.equal(ga, gb), name


def test_eval_and_no_grad_do_not_enter_candidate_helper(monkeypatch):
    import edsgn.candidate_batch as candidate_batch

    def forbidden(*args, **kwargs):
        raise AssertionError("training-only candidate helper was called")

    monkeypatch.setattr(candidate_batch, "tensor_query", forbidden)
    monkeypatch.setattr(candidate_batch, "numpy_contacts", forbidden)
    model = ForecastModel(_config("V4", True)).cpu()
    runtime = model.init_runtime(1)
    runtime.contacts = {0: {1: 4, 2: 2}}
    ids = torch.tensor([0])
    model.eval()
    model._candidates(ids, runtime)
    model.train()
    with torch.no_grad():
        model._candidates(ids, runtime)


def test_opt_in_training_contact_case_enters_tensor_helper(monkeypatch):
    import edsgn.candidate_batch as candidate_batch

    called = []
    original = candidate_batch.tensor_query

    def observed(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate_batch, "tensor_query", observed)
    model = ForecastModel(_config("V4", True)).cpu().train()
    runtime = model.init_runtime(1)
    runtime.contacts = {0: {1: 4, 2: 2}}
    ids = torch.tensor([0])
    with torch.enable_grad():
        model._candidates(ids, runtime)
    assert called


def test_training_candidate_tensor_is_opt_in_by_default():
    config = ModelConfig(n_nodes=7, n_features=2, hmax=1, d_total=16)
    assert config.training_candidate_tensor is False


def test_training_candidate_helper_guarded_for_v0_missing_contacts_and_k0(monkeypatch):
    import edsgn.candidate_batch as candidate_batch

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate helper called outside eligible training branch")

    monkeypatch.setattr(candidate_batch, "tensor_query", forbidden)
    monkeypatch.setattr(candidate_batch, "numpy_contacts", forbidden)
    for variant, n, k, cap, contacts in (
        ("V4", 7, 2, 4, {}),
        ("V0", 7, 2, 4, {0: {1: 3}}),
        ("V4", 1, 0, 0, {0: {}}),
    ):
        model = ForecastModel(ModelConfig(n_nodes=n, n_features=2, hmax=1,
                                          d_total=16, k=k, candidate_cap=cap,
                                          variant=variant,
                                          training_candidate_tensor=True)).cpu().train()
        runtime = model.init_runtime(1)
        runtime.contacts = contacts
        ids = torch.tensor([0])
        model._candidates(ids, runtime)
