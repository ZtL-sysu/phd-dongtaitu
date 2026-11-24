import json
import numpy as np
import pytest
import torch
from torch import nn

from edsgn import engine
from edsgn.data import PreparedData
from edsgn.robustness import (
    RobustnessPreparedData,
    canonical_recipe,
    recipe_digest,
    robustness_freeze_extras,
    robustness_implementation_sha256,
    evaluate_frozen_robustness,
    standard_missingness_recipes,
    validate_robustness_freeze,
)


def _toy_data():
    T, N, F = 12, 4, 1
    raw = np.zeros((T, N, F), dtype=np.float32)
    for t in range(T):
        raw[t, t % N, 0] = float(t + 1)
    mask = np.ones_like(raw, dtype=bool)
    events = np.array([
        [0, 0, 1], [3, 1, 2], [4, 2, 3], [5, 0, 2],
        [6, 2, 1], [7, 3, 0], [8, 1, 3], [10, 0, 3],
    ], dtype=np.int64)
    return PreparedData(
        values=np.log1p(raw), targets=raw, input_mask=mask, target_mask=np.ones_like(mask),
        obs_age=np.zeros_like(raw), interval_end=np.arange(1, T + 1), node_ids=np.arange(N),
        splits={"train": (0, 6), "validation": (6, 10), "test": (10, 12)},
        mean=np.array([0.3]), std=np.array([0.8]), scale=np.ones((N, F)),
        count_data=True, horizons=(1, 2), history=3, hmax=2, events=events,
        event_kind="email", metadata={"raw_values": raw}, partition="chronological",
    )


def _recipe(*, rate=0.0, kind="random", noise=0, jitter=0, seed=17):
    return {"schema_version": 1, "seed": seed,
            "missing": {"rate": rate, "kind": kind},
            "count_noise_max_abs": noise, "event_time_jitter_bins": jitter}


class _EchoRawModel(nn.Module):
    def forward(self, x, *, mask, age, raw, policy=None, event_priors=None, return_trace=False):
        B, _, N, F = raw.shape
        pred = raw[:, -1:].expand(B, 2, N, F).clone()
        return {"predictions": pred, "graph_loss": pred.sum() * 0,
                "counters": {"state_calls": 0, "readout_calls": 1}, "trace": []}


def test_recipe_is_explicit_and_rejects_unsupported_conditions():
    recipe = canonical_recipe(_recipe(rate=0.3, kind="time_block", jitter=1))
    assert recipe["missing"] == {"rate": 0.3, "kind": "time_block"}
    assert recipe_digest(recipe) == recipe_digest(dict(recipe))
    with pytest.raises(ValueError, match="missing rate"):
        canonical_recipe(_recipe(rate=0.2))
    with pytest.raises(ValueError, match="unsupported missingness kind"):
        canonical_recipe(_recipe(kind="node_classification"))
    with pytest.raises(ValueError, match="0 or ±1"):
        canonical_recipe(_recipe(jitter=2))
    grid = standard_missingness_recipes()
    assert len(grid) == 12
    assert {r["missing"]["rate"] for r in grid} == {0.0, 0.1, 0.3, 0.5}
    assert {r["missing"]["kind"] for r in grid} == {"random", "time_block", "related_group"}
    assert {r["seed"] for r in grid} == {2027}


def test_same_recipe_gives_identical_trajectory_mask_across_methods():
    data = _toy_data()
    recipe = _recipe(rate=0.3, kind="random")
    first = RobustnessPreparedData(data, recipe, trajectory_index=0)
    second = RobustnessPreparedData(data, recipe, trajectory_index=0)
    assert np.array_equal(first._input_mask, second._input_mask)
    assert first.perturbation_summary["trajectory_input_mask_sha256"] == second.perturbation_summary["trajectory_input_mask_sha256"]
    assert first.window(6)["input_mask"].shape == (3, 4, 1)


def test_true_zero_and_missing_are_distinguished_targets_stay_clean_and_engine_works():
    data = _toy_data()
    recipe = _recipe(rate=0.0)
    wrapped = RobustnessPreparedData(data, recipe)
    clean = data.window(6)
    perturbed = wrapped.window(6)
    zero_positions = clean["raw_observations"] == 0
    assert zero_positions.any()
    assert clean["input_mask"][zero_positions].all()  # real zero remains observed

    override = np.ones((data.history, *data.values.shape[1:]), dtype=bool)
    zero_index = tuple(np.argwhere(zero_positions)[0])
    override[zero_index] = False
    hidden = wrapped.window(6, mask_override=override)
    assert hidden["raw_observations"][zero_index] == 0
    assert hidden["values"][zero_index] == 0
    assert not bool(hidden["input_mask"][zero_index])
    assert "targets" not in perturbed

    batch_result = engine.evaluate_batches(_EchoRawModel(), [wrapped], "val", torch.device("cpu"), batch_size=2)
    expected_targets = np.stack([data.target(int(origin))["targets"] for _, origin in batch_result["pairs"]])
    expected_masks = np.stack([data.target(int(origin))["target_mask"] for _, origin in batch_result["pairs"]])
    assert np.array_equal(batch_result["targets"], expected_targets)
    assert np.array_equal(batch_result["target_mask"], expected_masks)


def test_time_block_is_frozen_for_whole_trajectory_and_counts_are_bounded():
    data = _toy_data()
    block = RobustnessPreparedData(data, _recipe(rate=0.3, kind="time_block"))
    hidden_times = np.flatnonzero(~block._input_mask.any(axis=(1, 2)))
    assert len(hidden_times) == round(len(data.values) * 0.3)
    assert hidden_times[-1] - hidden_times[0] + 1 == len(hidden_times)

    noisy = RobustnessPreparedData(data, _recipe(noise=2, seed=101))
    assert np.all(noisy._noisy_raw >= 0)
    assert np.max(np.abs(noisy._noisy_raw - data.raw_values)) <= 2
    assert noisy.target(6)["targets"].tolist() == data.target(6)["targets"].tolist()


def test_related_group_uses_train_only_data_and_jitter_excludes_future_before_shift():
    data = _toy_data()
    related = RobustnessPreparedData(data, _recipe(rate=0.5, kind="related_group"))
    group = related.perturbation_summary["train_only_related_group_node_order"]
    assert group is not None and len(group) == len(data.node_ids)
    assert related._input_mask[:, group[:2], :].sum() == 0
    # Original event at bin 7 has unique endpoints (3,0): it cannot be shifted
    # into the origin-6 input window because true-future filtering precedes jitter.
    jittered = RobustnessPreparedData(data, _recipe(jitter=1, seed=0))
    window = jittered.window(6)
    assert not np.any(np.all(window["events"][:, 1:] == np.array([3, 0]), axis=1))
    assert np.all(window["events"][:, 0] <= 6)
    assert np.all(window["events"][:, 0] >= 4)


def test_freeze_requires_matching_allowed_recipe_digest(monkeypatch):
    recipe = _recipe(rate=0.1, kind="random")
    freeze = {"extra": robustness_freeze_extras(recipe)}
    monkeypatch.setattr(engine, "validate_freeze", lambda *args: freeze)
    assert validate_robustness_freeze("ckpt", object(), {}, None, recipe) == freeze
    # A well-formed but unrelated digest reaches the authorization check;
    # malformed digests are rejected earlier by the schema validator.
    freeze["extra"]["allowed_robustness_recipe_digest"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not authorize"):
        validate_robustness_freeze("ckpt", object(), {}, None, recipe)


def test_freeze_fails_closed_when_implementation_source_is_missing_or_tampered(tmp_path, monkeypatch):
    import edsgn.robustness as robustness

    recipe = _recipe()
    freeze = {"extra": robustness_freeze_extras(recipe)}
    monkeypatch.setattr(engine, "validate_freeze", lambda *args: freeze)
    assert freeze["extra"]["allowed_robustness_implementation_sha256"] == robustness_implementation_sha256()
    del freeze["extra"]["allowed_robustness_implementation_sha256"]
    with pytest.raises(RuntimeError, match="implementation digest"):
        validate_robustness_freeze("ckpt", object(), {}, None, recipe)

    freeze["extra"] = robustness_freeze_extras(recipe)
    changed = tmp_path / "robustness.py"
    changed.write_text("# changed perturbation implementation\n", encoding="utf-8")
    monkeypatch.setattr(robustness, "__file__", str(changed))
    with pytest.raises(RuntimeError, match="implementation SHA-256"):
        validate_robustness_freeze("ckpt", object(), {}, None, recipe)


def test_toy_frozen_evaluation_writes_engine_schema_and_hash_bound_rows(tmp_path, monkeypatch):
    data = _toy_data()
    recipe = _recipe(rate=0.1, kind="random")
    model = _EchoRawModel()
    ckpt = {"config": {"dataset_path": "toy", "variant": "V0", "seed": 4}}
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"toy-checkpoint")
    (tmp_path / "test-freeze.json").write_text("{}", encoding="utf-8")
    freeze = {"extra": robustness_freeze_extras(recipe)}
    monkeypatch.setattr(engine, "load_checkpoint", lambda *args: (model, ckpt))
    monkeypatch.setattr(engine, "datasets_at", lambda *args: [data])
    monkeypatch.setattr(engine, "validate_freeze", lambda *args: freeze)

    output = tmp_path / "robust-run"
    evaluate_frozen_robustness(checkpoint, "test", output, recipe, device_name="cpu", batch_size=1)
    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"predictions", "targets", "target_mask", "origins", "seed_id"}
        assert int(archive["seed_id"].item()) == 4
        assert np.array_equal(archive["targets"], np.stack([data.target(9)["targets"]]))
    manifest = json.loads((output / "perturbation-manifest.json").read_text(encoding="utf-8"))
    assert manifest["weight_hash_unchanged"] is True
    assert manifest["input_recipe_sha256"] == recipe_digest(recipe)
    assert manifest["allowed_robustness_implementation_sha256"] == robustness_implementation_sha256()
    assert all(row["input_recipe_sha256"] == recipe_digest(recipe) for row in manifest["accuracy_rows"])
    assert all(row["model_weights_before_sha256"] == row["model_weights_after_sha256"] for row in manifest["counter_rows"])

