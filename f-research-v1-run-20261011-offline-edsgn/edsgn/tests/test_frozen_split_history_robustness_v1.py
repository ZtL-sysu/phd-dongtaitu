import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import frozen_split_history_robustness_v1 as gated
import split_history_robustness_v2 as split_history
from edsgn import engine, robustness
from edsgn.data import PreparedData
from edsgn.scheduling import Policy


def toy_prepared(seed=31, *, partition="test", node_ids=(7, 9), history=3, hmax=2,
                 count_data=False):
    """Use the production PreparedData contract and engine-compatible schema."""
    raw = np.arange(24, dtype=np.float32).reshape(12, 2, 1) / 10.0
    mask = np.ones_like(raw, dtype=bool)
    mask[4, 0, 0] = False
    values = np.where(mask, raw, 0).astype(np.float32)
    targets = np.arange(24, dtype=np.float32).reshape(12, 2, 1) + 50
    target_mask = np.ones_like(mask, dtype=bool)
    from edsgn.robustness import _compute_age
    return PreparedData(values=values, targets=targets, input_mask=mask,
        target_mask=target_mask, obs_age=_compute_age(mask), interval_end=np.arange(12),
        node_ids=np.asarray(node_ids), splits={"train": (0, 6), "test": (3, 10),
        "validation": (3, 10)}, mean=np.asarray([0.0]), std=np.asarray([1.0]),
        scale=np.ones((2, 1), dtype=np.float32), count_data=count_data, horizons=(1, 2),
        history=history, hmax=hmax, events=np.asarray([[2, 7, 9], [8, 7, 9], [10, 7, 9]]),
        event_kind="synthetic", metadata={"seed": seed, "source": "toy-source",
        "raw_values": raw}, partition=partition)


def recipe(kind="time_block", rate=0.5, seed=2027):
    return {"schema_version": 1, "seed": seed,
            "missing": {"kind": kind, "rate": rate},
            "count_noise_max_abs": 0, "event_time_jitter_bins": 0}


def state_file(path, *, p2="passed", p3="passed", p3a="passed", p4="running"):
    path.write_text(json.dumps({"stages": [
        {"id": "P2", "status": p2}, {"id": "P3", "status": p3},
        {"id": "P3A", "status": p3a}, {"id": "P4", "status": p4}]}), encoding="utf-8")
    return path


def test_domain_manifest_and_time_block_use_declared_split_without_targets(tmp_path):
    data = toy_prepared()
    clean_targets_before = data.targets.copy()
    target_method = data.target
    data.target = lambda _origin: pytest.fail("domain manifest must not read forecast targets")
    doc = gated.domain_manifest_from_metadata([data], "test")
    data.target = target_method
    traj = doc["trajectories"][0]
    assert traj["origin_count"] == len(data.origins("test"))
    assert traj["history_union_start_inclusive"] == int(data.origins("test")[0]) - data.history + 1
    assert traj["history_union_end_exclusive"] == int(data.origins("test")[-1]) + 1
    path = gated.write_domain_manifest(tmp_path / "domain.json", [data], "test", root=tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == doc
    with pytest.raises(FileExistsError):
        gated.write_domain_manifest(path, [data], "test", root=tmp_path)

    wrapped = split_history.build_robustness_view(data, recipe(), split="test", trajectory_index=0)
    assert wrapped.perturbation_summary["split_history_placement"]["origin_count"] == len(data.origins("test"))
    pairs, qc = gated._check_prepared_views([data], [wrapped], "test", recipe())
    assert len(pairs) == len(data.origins("test"))
    assert qc[0]["age_windows_checked"] == len(pairs)
    assert np.array_equal(data.targets, clean_targets_before)
    assert np.array_equal(engine.get_target(wrapped, int(pairs[0][1]))[0],
                          engine.get_target(data, int(pairs[0][1]))[0])
    assert qc[0]["time_block_union_realized_rate"] == 0.5


def test_freeze_extra_binds_source_domain_recipe_policy_and_normalization(tmp_path, monkeypatch):
    root = gated.ROOT
    scratch = root / "logs" / f".c06-toy-{tmp_path.name}"
    scratch.mkdir(parents=True)
    try:
        data = toy_prepared()
        target_dir = scratch / "dataset"
        target_dir.mkdir()
        (target_dir / "test_seed1.npz").write_bytes(b"toy metadata fingerprint bytes")
        checkpoint = scratch / "checkpoint.pt"
        checkpoint.write_bytes(b"toy frozen weights bytes")
        domain = gated.write_domain_manifest(scratch / "domain.json", [data], "test", root=root)
        fake_sources = {rel: ("a" * 64) for rel in gated._SOURCE_PATHS}
        monkeypatch.setattr(gated, "_source_hashes", lambda _root=root: fake_sources.copy())
        policy = Policy()
        ckpt = {"statistics": {"norm_mean": [0.0], "norm_std": [1.0]},
                "task": {"n_nodes": 2, "n_features": 1, "history": 3,
                         "hmax": 2, "count_data": False}}
        extras = gated.build_freeze_extras(checkpoint=checkpoint, ckpt=ckpt, policy=policy,
            datasets=[data], dataset_path=target_dir, domain_manifest_path=domain,
            recipe=recipe(), split="test", root=root)
        freeze = {"test_accessed_before_freeze": False, "extra": extras}
        validated = gated.validate_frozen_authorization(checkpoint=checkpoint, model=object(), ckpt=ckpt,
            policy=policy, datasets=[data], dataset_path=target_dir, domain_manifest_path=domain,
            recipe=recipe(), split="test", freeze=freeze, root=root)
        assert validated["c06_authorization"]["batch_size"] == 1
        assert validated["c06_authorization"]["dtype"] == "float32"
        assert validated["c06_authorization"]["policy_sha256"] == engine.policy_digest(policy)
        with pytest.raises(RuntimeError, match="binding differs"):
            gated.validate_frozen_authorization(checkpoint=checkpoint, model=object(), ckpt=ckpt,
                policy=Policy(kind="off"), datasets=[data], dataset_path=target_dir,
                domain_manifest_path=domain, recipe=recipe(), split="test", freeze=freeze, root=root)
        with pytest.raises(RuntimeError, match="does not authorize"):
            gated.validate_frozen_authorization(checkpoint=checkpoint, model=object(), ckpt=ckpt,
                policy=policy, datasets=[data], dataset_path=target_dir, domain_manifest_path=domain,
                recipe=recipe(rate=0.3), split="test", freeze=freeze, root=root)
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


@pytest.mark.parametrize("stages,split,allowed", [
    ({"p2": "passed", "p3": "passed", "p3a": "passed", "p4": "running"}, "test", True),
    ({"p2": "passed", "p3": "passed", "p3a": "running", "p4": "running"}, "val", True),
    ({"p2": "pending", "p3": "passed", "p3a": "passed", "p4": "running"}, "test", False),
    ({"p2": "passed", "p3": "running", "p3a": "passed", "p4": "running"}, "test", False),
    ({"p2": "passed", "p3": "passed", "p3a": "pending", "p4": "running"}, "test", False),
    ({"p2": "passed", "p3": "passed", "p3a": "passed", "p4": "pending"}, "test", False),
    ({"p2": "passed", "p3": "passed", "p3a": "pending", "p4": "running"}, "val", False),
])
def test_phase_gate_matrix(tmp_path, stages, split, allowed):
    state = state_file(tmp_path / "state.json", p2=stages["p2"], p3=stages["p3"], p3a=stages["p3a"], p4=stages["p4"])
    if allowed:
        assert gated.require_phase_gate(split, state)["P4"] == "running"
    else:
        with pytest.raises(RuntimeError):
            gated.require_phase_gate(split, state)


def test_partial_lock_acquisition_preserves_foreign_lease(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    foreign = logs / "offline-phase-wall" / "campaign.lock"
    foreign.parent.mkdir()
    foreign.write_text("foreign-lease", encoding="utf-8")
    with pytest.raises(FileExistsError):
        gated._acquire_leases(tmp_path, "test-id")
    assert not (logs / "model-resource.lock").exists()
    assert foreign.read_text(encoding="utf-8") == "foreign-lease"


def test_gate_failure_writes_failed_receipt_without_lock_or_model_load(tmp_path):
    state = state_file(tmp_path / "state.json", p3a="pending")
    checkpoint = tmp_path / "checkpoint.pt"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    domain = tmp_path / "domain.json"
    domain.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"not loaded")
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="P3A passed"):
        gated.evaluate_frozen(checkpoint=checkpoint, dataset_path=dataset,
            domain_manifest_path=domain, recipe=recipe(), output_dir=output,
            split="test", root=tmp_path, state_path=state,
            process_scan=lambda: pytest.fail("process scanner must not run before gate"),
            budget_fn=lambda *args: pytest.fail("budget/model path must not run before gate"))
    receipts = list((tmp_path / "logs/offline-phase-wall").glob("c06-history-*.json"))
    assert len(receipts) == 1
    doc = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert doc["status"] == "failed" and doc["test_accessed"] is False
    assert not (tmp_path / "logs/model-resource.lock").exists()


def test_budget_guard_fails_closed_on_incomplete_aggregate(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "aggregate_budget", SimpleNamespace(aggregate=lambda *a, **k: {
        "coverage_complete": False, "coverage_gaps": ["missing-ledger"], "spent_seconds": 10}))
    monkeypatch.setitem(sys.modules, "baseline_campaign", SimpleNamespace(WALL_BUDGET_S=72 * 3600))
    receipt_path = tmp_path / "logs" / "active.json"
    receipt = {"status": "running", "output_dir": str(tmp_path / "out")}
    with pytest.raises(RuntimeError, match="coverage incomplete"):
        gated._budget_guard(tmp_path, tmp_path / "out", receipt_path, receipt, 0.0)


def test_existing_output_refuses_before_receipt_creation(tmp_path):
    state = state_file(tmp_path / "state.json")
    checkpoint = tmp_path / "checkpoint.pt"
    domain = tmp_path / "domain.json"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    checkpoint.write_bytes(b"toy")
    domain.write_text("{}", encoding="utf-8")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        gated.evaluate_frozen(checkpoint=checkpoint, dataset_path=dataset,
            domain_manifest_path=domain, recipe=recipe(), output_dir=output,
            split="test", root=tmp_path, state_path=state)
    assert not list((tmp_path / "logs").rglob("c06-history-*.json"))


def test_checkpoint_task_uses_real_payload_schema_and_rejects_collection_mismatch():
    train = toy_prepared(partition="train")
    test = toy_prepared(partition="test")
    ckpt = {"task": {"n_nodes": 2, "n_features": 1, "history": 3,
                     "hmax": 2, "count_data": False},
            "config": {"dataset_path": "train-only-path"}}
    # This mirrors engine.train's payload: task metadata is nested in ckpt,
    # while PreparedData exposes dimensions through values.shape.
    qc = gated._validate_collection_task(ckpt, [test], [train])
    assert qc["collection_shapes"] == [[12, 2, 1]]
    assert "history" not in ckpt["config"] and "hmax" not in ckpt["config"]
    with pytest.raises(RuntimeError, match="task mismatch"):
        gated._validate_collection_task({"task": {**ckpt["task"], "history": 4}}, [test], [train])
    with pytest.raises(RuntimeError, match="node ordering"):
        reordered = toy_prepared(partition="test", node_ids=(9, 7))
        gated._validate_collection_task(ckpt, [reordered], [train])
