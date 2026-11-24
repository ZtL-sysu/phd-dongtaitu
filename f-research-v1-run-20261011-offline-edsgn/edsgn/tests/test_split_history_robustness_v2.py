"""CPU-only tests for strict split-history robustness v2."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RUN = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RUN / "edsgn" / "staging"), str(RUN / "edsgn" / "src")]
from edsgn.data import PreparedData
from edsgn import robustness
import split_history_robustness_v2 as split_history


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
                      event_kind="email", metadata={"seed": seed, "source": "toy-email"}, partition="chronological")
    ds._raw_values = raw.copy()
    return ds


def recipe(rate=.5, *, kind="time_block", seed=2027, noise=0, jitter=0):
    return robustness.canonical_recipe({"schema_version": 1, "seed": seed,
        "missing": {"rate": rate, "kind": kind}, "count_noise_max_abs": noise,
        "event_time_jitter_bins": jitter})


def prepare_freeze_files(tmp_path, monkeypatch, datasets, split="test"):
    runroot = tmp_path / "run"; runroot.mkdir()
    wrapper = runroot / "split_history_robustness_v2.py"
    source = runroot / "robustness.py"
    wrapper.write_text("wrapper-bytes-v1", encoding="utf-8")
    source.write_text("dependency-bytes-v1", encoding="utf-8")
    monkeypatch.setattr(split_history, "ROOT", runroot)
    monkeypatch.setattr(split_history, "__file__", str(wrapper))
    monkeypatch.setattr(robustness, "__file__", str(source))
    rows = [_ for _ in datasets]
    placements = [_time for _time in [split_history._time_block_bounds(d, recipe(.3), split, i)
                                      for i, d in enumerate(rows)]]
    expected = split_history._expected_collection_domain_manifest(rows, placements)
    domain = runroot / "domain.json"
    domain.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    return runroot, wrapper, source, domain


def test_split_range_uses_origins_and_shape_not_target_content():
    data = make_data()
    before = split_history.split_history_range(data, "test")
    data.targets[:] = np.nan
    data.target_mask[:] = False
    after = split_history.split_history_range(data, "test")
    assert before == after
    assert before["history_union_start_inclusive"] == 16
    assert before["history_union_end_exclusive"] == 37
    validation = split_history.split_history_range(data, "validation")
    assert validation["split"] == "val"
    assert validation["origin_count"] == len(data.origins("val"))


def test_weighted_window_missingness_pre_post_incremental_and_union_are_exact():
    data = make_data()
    data.input_mask[16:20, 0, 0] = False
    view = split_history.SplitHistoryRobustnessPreparedData(data, recipe(.5), split="test", trajectory_index=2)
    p = view.split_history
    origins = data.origins("test")
    hidden_slots = 0
    all_slots = 0
    original_observed = 0
    blocked_time_slots = 0
    for origin in origins:
        start = int(origin) - data.history + 1
        all_slots += int(data.input_mask[start:int(origin)+1].size)
        original_observed += int(data.input_mask[start:int(origin)+1].sum())
        hidden_slots += int((data.input_mask[start:int(origin)+1] & ~view._input_mask[start:int(origin)+1]).sum())
        block_start, block_end = p["block_start_inclusive"], p["block_end_exclusive"]
        if block_start is not None:
            blocked_time_slots += max(0, min(int(origin)+1, block_end) - max(start, block_start)) * data.values.shape[1] * data.values.shape[2]
    assert p["window_history_total_positions"] == all_slots
    assert p["window_history_original_observed_positions"] == original_observed
    assert p["window_history_newly_hidden_observed_positions"] == hidden_slots
    assert p["window_history_incremental_missing_fraction_of_original_observed"] == pytest.approx(hidden_slots / original_observed)
    assert p["window_history_missing_fraction_before"] == pytest.approx(1 - original_observed / all_slots)
    assert p["window_history_missing_fraction_after"] == pytest.approx(1 - (original_observed-hidden_slots) / all_slots)
    assert p["window_history_blocked_position_fraction"] == pytest.approx(blocked_time_slots / all_slots)
    assert p["realized_history_union_rate"] == pytest.approx(p["block_length"] / 21)
    assert p["union_missing_fraction_after"] > p["union_missing_fraction_before"]
    assert p["trajectory_input_mask_sha256"] == view.perturbation_summary["trajectory_input_mask_sha256"]
    assert np.array_equal(view._obs_age, robustness._compute_age(view._input_mask))
    outside = np.ones(view._input_mask.shape[0], dtype=bool)
    outside[p["block_start_inclusive"]:p["block_end_exclusive"]] = False
    assert np.array_equal(view._input_mask[outside], data.input_mask[outside])


def test_mask_dependent_summary_recomputed_after_block_and_zero_fill():
    data = make_data(count_data=True)
    data._raw_values[:] = 0
    data.values[:] = 0
    data.mean[:] = .5
    data.std[:] = 2.0
    view = split_history.SplitHistoryRobustnessPreparedData(data, recipe(.5, noise=1), split="test", trajectory_index=1)
    visible = int((data.input_mask & view._input_mask).sum())
    assert view.perturbation_summary["clean_true_zero_observed_positions"] == visible
    assert view.perturbation_summary["noisy_input_zero_positions_observed"] == int(((view._noisy_raw == 0) & view._input_mask).sum())
    assert view.perturbation_summary["count_noise_applied_positions"] == visible
    expected_values = (np.log1p(view._noisy_raw) - data.mean.reshape(1, 1, -1)) / data.std.reshape(1, 1, -1)
    assert np.allclose(view._values[view._input_mask], expected_values[view._input_mask])
    origin = int(data.origins("test")[len(data.origins("test")) // 2])
    win = view.window(origin)
    start = origin - data.history + 1
    assert np.all(win["raw_observations"][~win["input_mask"]] == 0)
    assert np.array_equal(win["obs_age"], view._obs_age[start:origin+1])
    assert np.array_equal(view.target(origin)["targets"], data.target(origin)["targets"])
    assert np.array_equal(view.target(origin)["target_mask"], data.target(origin)["target_mask"])


def test_time_block_is_shared_by_construction_and_other_recipes_stay_delegated():
    data = make_data()
    spec = recipe(.3)
    a = split_history.build_robustness_view(data, spec, split="test", trajectory_index=3)
    b = split_history.build_robustness_view(data, spec, split="test", trajectory_index=3)
    assert np.array_equal(a._input_mask, b._input_mask)
    assert split_history.build_robustness_view(data, recipe(.3, kind="random"), split="test").__class__ is robustness.RobustnessPreparedData
    assert split_history.build_robustness_view(data, recipe(.3, kind="related_group"), split="test").__class__ is robustness.RobustnessPreparedData
    assert split_history.SplitHistoryRobustnessPreparedData(data, recipe(0), split="test").split_history["block_start_inclusive"] is None


def test_noise_jitter_initialization_survives_mask_replacement():
    data = make_data(count_data=True)
    spec = recipe(.5, noise=2, jitter=1)
    v2 = split_history.SplitHistoryRobustnessPreparedData(data, spec, split="test", trajectory_index=1)
    base = robustness.RobustnessPreparedData(data, recipe(0, noise=2, jitter=1), trajectory_index=1)
    assert np.array_equal(v2._noisy_raw, base._noisy_raw)
    assert np.array_equal(v2.window(30)["events"], base.window(30)["events"])
    assert v2.perturbation_summary["event_time_jitter_bins"] == 1
    assert np.array_equal(v2.target(30)["targets"], data.target(30)["targets"])


def test_freeze_requires_executing_wrapper_dependency_and_exact_domain(tmp_path, monkeypatch):
    data = make_data()
    runroot, wrapper, source, domain = prepare_freeze_files(tmp_path, monkeypatch, [data])
    # An unrelated in-root file cannot stand in for the currently executing module.
    fake_wrapper = runroot / "fake.py"; fake_wrapper.write_text("other", encoding="utf-8")
    with pytest.raises(ValueError, match="executing module"):
        split_history.split_history_freeze_extras(wrapper_path=fake_wrapper, domain_manifest_path=domain,
            data=data, recipe=recipe(.3), split="test")
    frozen = split_history.split_history_freeze_extras(wrapper_path=wrapper, domain_manifest_path=domain,
        data=data, recipe=recipe(.3), split="test")
    assert frozen["split_history_time_block"]["robustness_source_sha256"]
    valid = split_history.validate_split_history_freeze({"split_history_time_block": frozen["split_history_time_block"]},
        wrapper_path=wrapper, domain_manifest_path=domain, data=data, recipe=recipe(.3), split="test")
    assert valid["selected_split"] == "test"
    for tamper in ("schema", "dataset", "trajectories", "numeric_type"):
        doc = json.loads(domain.read_text(encoding="utf-8"))
        if tamper == "schema": doc["schema"] = "unknown"
        elif tamper == "dataset": doc["dataset"]["shape_tnf"][0] += 1
        elif tamper == "trajectories": doc["trajectories"][0]["domain_bounds_end_exclusive"] -= 1
        else: doc["trajectories"][0]["trajectory_index"] = 0.0
        domain.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError, match="Domain manifest"):
            split_history.split_history_freeze_extras(wrapper_path=wrapper, domain_manifest_path=domain,
                data=data, recipe=recipe(.3), split="test")
    domain.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        split_history.split_history_freeze_extras(wrapper_path=wrapper, domain_manifest_path=domain,
            data=data, recipe=recipe(.3), split="test")
    domain.write_text(json.dumps(split_history._expected_domain_manifest(
        data, split_history._time_block_bounds(data, recipe(.3), "test", 0))), encoding="utf-8")
    source.write_text("dependency-bytes-v2", encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata differs"):
        split_history.validate_split_history_freeze({"split_history_time_block": frozen["split_history_time_block"]},
            wrapper_path=wrapper, domain_manifest_path=domain, data=data, recipe=recipe(.3), split="test")


def test_domain_bundle_binds_every_trajectory_and_rejects_mismatch(tmp_path, monkeypatch):
    data0, data1 = make_data(seed=17), make_data(seed=18)
    runroot, wrapper, source, domain = prepare_freeze_files(tmp_path, monkeypatch, [data0, data1])
    frozen = split_history.split_history_freeze_bundle_extras(wrapper_path=wrapper, domain_manifest_path=domain,
        datasets=[data0, data1], recipe=recipe(.3), split="test")
    valid = split_history.validate_split_history_freeze_bundle(
        {"split_history_time_block_bundle": frozen["split_history_time_block_bundle"]},
        wrapper_path=wrapper, domain_manifest_path=domain, datasets=[data0, data1], recipe=recipe(.3), split="test")
    assert valid["trajectory_count"] == 2
    doc = json.loads(domain.read_text(encoding="utf-8")); doc["trajectories"].pop()
    domain.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="Domain manifest"):
        split_history.split_history_freeze_bundle_extras(wrapper_path=wrapper, domain_manifest_path=domain,
            datasets=[data0, data1], recipe=recipe(.3), split="test")


@pytest.mark.parametrize("bad", [True, -1, 1.2, "2"])
def test_trajectory_index_rejects_bool_negative_and_noninteger(bad):
    with pytest.raises(ValueError, match="trajectory_index"):
        split_history._time_block_bounds(make_data(), recipe(.3), "test", bad)


def test_invalid_origins_and_domain_escape_fail_closed(tmp_path, monkeypatch):
    data = make_data(empty_val=True)
    with pytest.raises(ValueError, match="no complete origins"):
        split_history.split_history_range(data, "val")
    with pytest.raises(ValueError, match="Unsupported declared split"):
        split_history.split_history_range(data, "trainish")
    monkeypatch.setattr(split_history, "ROOT", tmp_path / "run")
    split_history.ROOT.mkdir()
    with pytest.raises(ValueError, match="inside the experiment run root"):
        split_history._contained(tmp_path / "outside.json")

