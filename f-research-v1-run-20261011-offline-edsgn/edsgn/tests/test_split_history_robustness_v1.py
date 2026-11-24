"""CPU toy tests for prospective split-history time-block visibility."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "staging"))
sys.path.insert(0, str(ROOT / "src"))

from edsgn.data import PreparedData
from edsgn.robustness import RobustnessPreparedData, canonical_recipe, _compute_age
import split_history_robustness_v1 as split_history


def make_data(*, count_data=False, empty_val=False, seed=17):
    t, n, f = 48, 3, 1
    raw = (np.arange(t * n * f, dtype=np.float32).reshape(t, n, f) + 1)
    if count_data:
        raw = np.floor(raw)
    target = raw * 10 + 3
    mask = np.ones((t, n, f), dtype=bool)
    splits = {"train": (0, 12), "validation": (12, 20) if not empty_val else (0, 0), "test": (20, 40)}
    ds = PreparedData(values=raw.copy(), targets=target.copy(), input_mask=mask.copy(),
                      target_mask=mask.copy(), obs_age=np.zeros_like(raw),
                      interval_end=np.arange(t, dtype=np.float64), node_ids=np.arange(n),
                      splits=splits, mean=np.zeros(f), std=np.ones(f), scale=np.ones(f),
                      count_data=count_data, horizons=(1, 2, 3), history=4, hmax=3,
                      events=np.asarray([[20, 0, 1], [27, 1, 0]], dtype=np.int64),
                      event_kind="email", metadata={"seed": seed}, partition="chronological")
    ds._raw_values = raw.copy()
    return ds


def recipe(rate=.5, *, kind="time_block", seed=2027, noise=0, jitter=0):
    return canonical_recipe({"schema_version": 1, "seed": seed,
                             "missing": {"rate": rate, "kind": kind},
                             "count_noise_max_abs": noise,
                             "event_time_jitter_bins": jitter})


def test_time_block_hits_the_complete_test_history_union_and_is_shared():
    data = make_data()
    spec = recipe(.5)
    first = split_history.build_robustness_view(data, spec, split="test", trajectory_index=2)
    second = split_history.build_robustness_view(data, spec, split="test", trajectory_index=2)
    placement = first.split_history
    assert (placement["origin_first"], placement["origin_last"]) == (19, 36)
    assert (placement["history_union_start_inclusive"], placement["history_union_end_exclusive"]) == (16, 37)
    assert placement["block_length"] == round(21 * .5)
    assert placement["history_union_start_inclusive"] <= placement["block_start_inclusive"]
    assert placement["block_end_exclusive"] <= placement["history_union_end_exclusive"]
    assert np.array_equal(first._input_mask, second._input_mask)
    assert int((data.input_mask & ~first._input_mask).sum()) > 0
    assert first.perturbation_summary["hidden_input_positions"] > 0
    for origin in data.origins("test"):
        clean_target = data.target(int(origin))
        altered_target = first.target(int(origin))
        assert np.array_equal(clean_target["targets"], altered_target["targets"])
        assert np.array_equal(clean_target["target_mask"], altered_target["target_mask"])
        window = first.window(int(origin))
        assert np.array_equal(window["input_mask"], second.window(int(origin))["input_mask"])
        assert np.array_equal(window["obs_age"], first._obs_age[int(origin)-3:int(origin)+1])
    assert np.array_equal(first._obs_age, _compute_age(first._input_mask))


def test_rate_zero_and_fixed_seed_are_explicit_and_do_not_draw_outside_union():
    data = make_data()
    zero = split_history.SplitHistoryRobustnessPreparedData(data, recipe(0), split="test")
    assert zero.split_history["block_start_inclusive"] is None
    assert zero.split_history["block_end_exclusive"] is None
    assert np.array_equal(zero._input_mask, data.input_mask)
    a = split_history.SplitHistoryRobustnessPreparedData(data, recipe(.3), split="test", trajectory_index=5)
    b = split_history.SplitHistoryRobustnessPreparedData(data, recipe(.3), split="test", trajectory_index=5)
    c = split_history.SplitHistoryRobustnessPreparedData(data, recipe(.3), split="test", trajectory_index=6)
    assert a.split_history["block_start_inclusive"] == b.split_history["block_start_inclusive"]
    assert a.split_history["block_start_inclusive"] != c.split_history["block_start_inclusive"]
    with pytest.raises(ValueError, match="fixed at 2027"):
        split_history.SplitHistoryRobustnessPreparedData(data, recipe(.3, seed=2028), split="test")


def test_count_noise_and_explicit_event_jitter_keep_existing_semantics():
    data = make_data(count_data=True)
    perturbed = split_history.SplitHistoryRobustnessPreparedData(
        data, recipe(.5, noise=2, jitter=1), split="test", trajectory_index=1)
    no_missing = RobustnessPreparedData(data, recipe(0, noise=2, jitter=1), trajectory_index=1)
    assert np.array_equal(perturbed._noisy_raw, no_missing._noisy_raw)
    assert np.array_equal(perturbed.window(30)["events"], no_missing.window(30)["events"])
    assert np.array_equal(perturbed.target(30)["targets"], data.target(30)["targets"])
    assert perturbed.perturbation_summary["event_time_jitter_bins"] == 1
    random_view = split_history.build_robustness_view(data, recipe(.3, kind="random"), split="test")
    group_view = split_history.build_robustness_view(data, recipe(.3, kind="related_group"), split="test")
    assert type(random_view) is RobustnessPreparedData
    assert type(group_view) is RobustnessPreparedData


def test_freeze_metadata_binds_wrapper_domain_recipe_split_and_origin_range(tmp_path, monkeypatch):
    data = make_data()
    runroot = tmp_path / "run"
    runroot.mkdir()
    wrapper = runroot / "wrapper.py"
    domain = runroot / "domain.json"
    wrapper.write_text("wrapper-source-v1", encoding="utf-8")
    domain.write_text('{"dataset":"toy-domain-v1"}', encoding="utf-8")
    monkeypatch.setattr(split_history, "ROOT", runroot)
    spec = recipe(.1)
    frozen = split_history.split_history_freeze_extras(
        wrapper_path=wrapper, domain_manifest_path=domain, data=data,
        recipe=spec, split="test", trajectory_index=3)
    validated = split_history.validate_split_history_freeze(
        frozen, wrapper_path=wrapper, domain_manifest_path=domain, data=data,
        recipe=spec, split="test", trajectory_index=3)
    assert validated["selected_split"] == "test"
    assert validated["history_union_start_inclusive"] == 16
    for key, value in (("recipe_sha256", "0"*64), ("origin_first", 20),
                       ("selected_split", "validation"), ("domain_manifest_sha256", "1"*64)):
        tampered = {"split_history_time_block": {**frozen["split_history_time_block"], key: value}}
        with pytest.raises(RuntimeError, match="metadata differs"):
            split_history.validate_split_history_freeze(
                tampered, wrapper_path=wrapper, domain_manifest_path=domain, data=data,
                recipe=spec, split="test", trajectory_index=3)
    with pytest.raises(RuntimeError, match="lacks split-history"):
        split_history.validate_split_history_freeze(
            {}, wrapper_path=wrapper, domain_manifest_path=domain, data=data,
            recipe=spec, split="test", trajectory_index=3)
    wrapper.write_text("changed-wrapper", encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata differs"):
        split_history.validate_split_history_freeze(
            frozen, wrapper_path=wrapper, domain_manifest_path=domain, data=data,
            recipe=spec, split="test", trajectory_index=3)
    wrapper.write_text("wrapper-source-v1", encoding="utf-8")
    domain.write_text('{"dataset":"tampered-domain"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata differs"):
        split_history.validate_split_history_freeze(
            frozen, wrapper_path=wrapper, domain_manifest_path=domain, data=data,
            recipe=spec, split="test", trajectory_index=3)


def test_invalid_split_empty_origin_range_and_missing_manifest_fail_closed(tmp_path, monkeypatch):
    data = make_data(empty_val=True)
    with pytest.raises(ValueError, match="no complete origins"):
        split_history.split_history_range(data, "val")
    with pytest.raises(ValueError, match="Unsupported declared split"):
        split_history.split_history_range(data, "testish")
    monkeypatch.setattr(split_history, "ROOT", tmp_path)
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("source", encoding="utf-8")
    with pytest.raises(ValueError, match="existing wrapper source and domain manifest"):
        split_history.split_history_freeze_extras(
            wrapper_path=wrapper, domain_manifest_path=tmp_path / "missing.json",
            data=data, recipe=recipe(.1), split="test")
    with pytest.raises(ValueError, match="existing wrapper source and domain manifest"):
        split_history.split_history_freeze_extras(
            wrapper_path=tmp_path / "missing.py", domain_manifest_path=wrapper,
            data=data, recipe=recipe(.1), split="test")


def test_collection_freeze_binds_every_trajectory_placement(tmp_path, monkeypatch):
    data0, data1 = make_data(seed=17), make_data(seed=18)
    runroot = tmp_path / "run"
    runroot.mkdir()
    wrapper = runroot / "wrapper.py"
    domain = runroot / "domain.json"
    wrapper.write_text("wrapper-source", encoding="utf-8")
    domain.write_text('{"domain":"synthetic-test-collection"}', encoding="utf-8")
    monkeypatch.setattr(split_history, "ROOT", runroot)
    spec = recipe(.3)
    frozen = split_history.split_history_freeze_bundle_extras(
        wrapper_path=wrapper, domain_manifest_path=domain,
        datasets=[data0, data1], recipe=spec, split="test")
    validated = split_history.validate_split_history_freeze_bundle(
        frozen, wrapper_path=wrapper, domain_manifest_path=domain,
        datasets=[data0, data1], recipe=spec, split="test")
    assert validated["trajectory_count"] == 2
    assert [item["trajectory_index"] for item in validated["trajectories"]] == [0, 1]
    assert [item["source_trajectory_seed"] for item in validated["trajectories"]] == [17, 18]
    tampered = {"split_history_time_block_bundle": {
        **validated, "trajectories": [validated["trajectories"][0]]}}
    with pytest.raises(RuntimeError, match="collection metadata differs"):
        split_history.validate_split_history_freeze_bundle(
            tampered, wrapper_path=wrapper, domain_manifest_path=domain,
            datasets=[data0, data1], recipe=spec, split="test")
    with pytest.raises(RuntimeError, match="lacks split-history"):
        split_history.validate_split_history_freeze_bundle(
            {}, wrapper_path=wrapper, domain_manifest_path=domain,
            datasets=[data0, data1], recipe=spec, split="test")
