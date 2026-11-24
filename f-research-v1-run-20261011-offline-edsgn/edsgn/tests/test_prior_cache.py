from collections import Counter

import numpy as np
import pytest
import torch

from edsgn.model import ForecastModel, ModelConfig
from edsgn.prior_cache import TrainingPriorCache, build_prior_plan, prior_content_key


def make_model(*, cache=True, cap_bytes=1 << 20, n=5):
    return ForecastModel(ModelConfig(
        n_nodes=n, n_features=1, hmax=2, history=4, variant="V1",
        d_total=16, k=2, candidate_cap=min(3, n - 1),
        training_prior_cache=cache, training_prior_cache_bytes=cap_bytes,
    ))


def event_pack(kind, offset=0, future_dst=3):
    # Deliberate duplicates, same-bin order, row ties and K-driven evictions.
    records = np.asarray([
        [10, 0, 1], [10, 0, 2], [10, 0, 3], [10, 0, 4],
        [11, 0, 2], [11, 0, 1], [11, 0, 2], [12, 1, 4],
        [12, 0, 4], [13, 0, future_dst],
    ], dtype=np.int64)
    records[:, 1:] = (records[:, 1:] + offset) % 5
    records[-1, 2] = future_dst
    return {"kind": kind, "records": records, "time_indices": np.arange(10, 14, dtype=np.int64)}


def run_direct(model, packs, *, cached):
    model.train()
    runtime = model.init_runtime(len(packs), n_nodes=model.config.n_nodes, device=torch.device("cpu"))
    for step in range(4):
        runtime.step = step
        model._prior_events(packs, runtime)
    return runtime


def normalized_contacts(runtime):
    return {int(k): dict(v) for k, v in runtime.contacts.items()}


def test_cached_step_deltas_match_raw_email_and_mooc_and_second_runtime_hits():
    for kind in ("email", "mooc"):
        cached = make_model(cache=True)
        raw = make_model(cache=False)
        raw.load_state_dict(cached.state_dict())
        pack = event_pack(kind)
        cold = run_direct(cached, [pack], cached=True)
        reference = run_direct(raw, [pack], cached=False)
        assert normalized_contacts(cold) == normalized_contacts(reference)
        assert cold.user_history == reference.user_history
        assert cold.counters["prior_events_applied"] == reference.counters["prior_events_applied"] == len(pack["records"])
        assert cold.counters["prior_pairs_applied"] == reference.counters["prior_pairs_applied"]
        assert cold.counters["prior_events_processed"] == reference.counters["prior_events_processed"]
        assert cold.counters["prior_pairs_generated"] == reference.counters["prior_pairs_generated"]

        warm = run_direct(cached, [pack], cached=True)
        assert normalized_contacts(warm) == normalized_contacts(reference)
        assert warm.user_history == reference.user_history
        assert warm.counters["prior_cache_hits"] == 1
        assert warm.counters["prior_events_processed"] == 0
        assert warm.counters["prior_pairs_generated"] == 0
        assert warm.counters["prior_events_cache_replayed"] == len(pack["records"])
        assert warm.counters["prior_pairs_cache_replayed"] == reference.counters["prior_pairs_applied"]
        assert warm.counters["prior_events_applied"] == reference.counters["prior_events_applied"]


def test_forward_predictions_graphloss_and_gradients_match_for_both_kinds():
    torch.set_num_threads(2)
    torch.manual_seed(81)
    for kind in ("email", "mooc"):
        cached = make_model(cache=True, n=4)
        raw = make_model(cache=False, n=4)
        raw.load_state_dict(cached.state_dict())
        x = torch.randn(2, 4, 4, 1)
        event_records = np.asarray([[100,0,1],[101,0,2],[101,0,3],[102,1,3],[103,2,0]], dtype=np.int64)
        packs = [
            {"kind": kind, "records": event_records.copy(), "time_indices": np.arange(100,104,dtype=np.int64)},
            {"kind": kind, "records": event_records[::-1].copy(), "time_indices": np.arange(100,104,dtype=np.int64)},
        ]
        out_cached = cached(x, event_priors=packs, return_intermediate=True)
        out_raw = raw(x, event_priors=packs, return_intermediate=True)
        assert torch.allclose(out_cached["predictions"], out_raw["predictions"], atol=1e-7, rtol=0)
        assert torch.allclose(out_cached["graph_loss"], out_raw["graph_loss"], atol=1e-7, rtol=0)
        loss_cached = out_cached["predictions"].square().mean() + out_cached["graph_loss"]
        loss_raw = out_raw["predictions"].square().mean() + out_raw["graph_loss"]
        loss_cached.backward(); loss_raw.backward()
        grads_c = dict(cached.named_parameters()); grads_r = dict(raw.named_parameters())
        assert grads_c.keys() == grads_r.keys()
        for name in grads_c:
            assert (grads_c[name].grad is None) == (grads_r[name].grad is None), name
            if grads_c[name].grad is not None:
                assert torch.allclose(grads_c[name].grad, grads_r[name].grad, atol=2e-6, rtol=0), name
        assert out_cached["runtime"].counters["prior_cache_misses"] == 2
        assert out_cached["runtime"].counters["prior_cache_build_events"] == 2 * len(event_records)


def test_batch_reordering_reuses_content_keys_without_slot_leakage():
    cached = make_model(cache=True)
    raw = make_model(cache=False)
    raw.load_state_dict(cached.state_dict())
    a, b = event_pack("email", offset=0), event_pack("email", offset=1)
    warm = run_direct(cached, [a, b], cached=True)
    reversed_hit = run_direct(cached, [b, a], cached=True)
    reversed_raw = run_direct(raw, [b, a], cached=False)
    assert normalized_contacts(reversed_hit) == normalized_contacts(reversed_raw)
    assert reversed_hit.user_history == reversed_raw.user_history
    assert reversed_hit.counters["prior_cache_hits"] == 2
    assert warm.counters["prior_cache_misses"] == 2


def test_future_event_changes_key_but_cannot_change_earlier_prefix():
    model = make_model(cache=True, n=4)
    x = torch.randn(1,4,4,1)
    first = event_pack("email", future_dst=1)
    later = event_pack("email", future_dst=3)
    p1 = {"kind":"email", "records":np.asarray([[100,0,1],[103,0,1]],dtype=np.int64), "time_indices":np.arange(100,104,dtype=np.int64)}
    p2 = {"kind":"email", "records":np.asarray([[100,0,1],[103,0,3]],dtype=np.int64), "time_indices":np.arange(100,104,dtype=np.int64)}
    assert prior_content_key(p1,4,3) != prior_content_key(p2,4,3)
    out1=model(x,event_priors=[p1],return_intermediate=True)
    out2=model(x,event_priors=[p2],return_intermediate=True)
    assert torch.equal(out1["intermediate"][0],out2["intermediate"][0])
    assert torch.equal(out1["intermediate"][1],out2["intermediate"][1])
    assert out2["runtime"].counters["prior_cache_misses"] == 1
    assert out2["runtime"].counters["prior_cache_hits"] == 0


def test_eval_and_no_grad_bypass_training_cache_and_byte_cap_is_enforced():
    model = make_model(cache=True, cap_bytes=1 << 20)
    x=torch.randn(1,4,5,1); pack=event_pack("mooc")
    model.train(); model(x,event_priors=[pack])
    model.eval()
    eval_out=model(x,event_priors=[pack])
    assert eval_out["runtime"].counters["prior_cache_hits"] == 0
    assert eval_out["runtime"].counters["prior_cache_misses"] == 0
    assert eval_out["runtime"].counters["prior_events_processed"] == len(pack["records"])
    model.train()
    with torch.no_grad():
        no_grad_out=model(x,event_priors=[pack])
    assert no_grad_out["runtime"].counters["prior_cache_hits"] == 0
    assert no_grad_out["runtime"].counters["prior_cache_misses"] == 0
    assert no_grad_out["runtime"].counters["prior_events_processed"] == len(pack["records"])

    bounded=make_model(cache=True,cap_bytes=256)
    bounded_rt=run_direct(bounded,[pack],cached=True)
    assert bounded._training_prior_cache.bytes_used <= 256
    assert bounded_rt.counters["prior_cache_oversize"] == 1
    assert bounded_rt.counters["prior_cache_hits"] == 0
    assert bounded_rt.counters["prior_cache_bypasses"] == 4
    assert bounded_rt.counters["prior_events_processed"] == len(pack["records"])


def test_lru_respects_hard_estimated_byte_cap_and_key_binds_dimensions():
    pack=event_pack("email")
    plan=build_prior_plan(pack,5,3,1<<20)
    assert plan is not None
    cache=TrainingPriorCache(plan.estimated_bytes)
    key=prior_content_key(pack,5,3)
    assert cache.put(key,plan)
    assert cache.bytes_used <= cache.max_bytes
    assert cache.get(key) is plan
    assert prior_content_key(pack,6,3) != key
    assert prior_content_key(pack,5,2) != key


def test_single_node_empty_graph_cache_flag_preserves_raw_forward():
    config = dict(n_nodes=1, n_features=1, hmax=2, history=4,
                  variant="V4", d_total=16, k=2, candidate_cap=3)
    cached = ForecastModel(ModelConfig(**config, training_prior_cache=True))
    raw = ForecastModel(ModelConfig(**config, training_prior_cache=False))
    raw.load_state_dict(cached.state_dict())
    x = torch.randn(1, 4, 1, 1)
    pack = {"kind": "synthetic", "records": np.empty((0, 3), dtype=np.int64),
            "time_indices": np.arange(4, dtype=np.int64)}
    first = cached(x, event_priors=[pack])
    second = raw(x, event_priors=[pack])
    assert torch.equal(first["predictions"], second["predictions"])
    assert first["runtime"].counters["prior_cache_misses"] == 0

