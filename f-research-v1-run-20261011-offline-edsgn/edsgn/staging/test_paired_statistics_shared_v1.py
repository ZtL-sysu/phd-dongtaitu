import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "edsgn" / "staging" / "paired_statistics_shared_v1.py"
spec = importlib.util.spec_from_file_location("paired_statistics_shared_v1", MODULE_PATH)
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def _write(path, seed, pred, targets, mask, pairs):
    np.savez(path, predictions=pred, targets=targets, target_mask=mask,
             origins=pairs, seed_id=np.asarray(seed))


def _config(tmp_path, *, mutate=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pairs = np.array([[0, 10], [0, 11], [0, 12], [1, 20], [1, 21], [1, 22]])
    targets = np.zeros((6, 2, 2, 1), dtype=np.float64)
    mask = np.ones_like(targets, dtype=bool)
    mask[1, :, 1, 0] = False
    if mutate == "target":
        targets[0, 0, 0, 0] = 1
    archive_paths = {"A": {"left": [], "right": []}, "B": {"left": [], "right": []}}
    for contrast in archive_paths:
        for side in ("left", "right"):
            for seed in range(5):
                prediction = np.zeros_like(targets)
                prediction[:, :, :, 0] = (seed + 1) * (1 if side == "left" else 2)
                if mutate == "heterogeneous" and side == "left":
                    prediction[1, :, 0, 0] = 9
                path = tmp_path / f"{contrast}-{side}-{seed}.npz"
                _write(path, seed, prediction, targets, mask, pairs)
                archive_paths[contrast][side].append({"path": str(path), "prediction_sha256": "placeholder",
                                                       "input_binding_path": "fake", "input_binding_sha256": "fake"})
    config = {"schema_version": 1, "split": "validation", "bootstrap_reps": 2000,
              "random_seed": 1234, "history_length": 2, "max_horizon": 2,
              "validation_acf_dependence_windows": 1, "block_length_windows": 2,
              "requested_horizons": [1, 2], "seed_ids": [0, 1, 2, 3, 4],
              "contrasts": [{"name": n, "family": "primary", **archive_paths[n]} for n in archive_paths]}
    return config, pairs


def _gate(config_sha="cfg"):
    return {"status": "p3a-passed", "analysis_config_sha256": config_sha,
            "split": "validation", "block_length_windows": 2}


def _mock_bound(monkeypatch):
    def load(path_value, spec, base):
        path = Path(path_value)
        item = paired.st._load_prediction(path)
        return item, {"prediction_path": str(path), "prediction_sha256": "toy",
                      "binding_path": "toy-at-launch-binding", "binding_sha256": "toy",
                      "binding_verification": {"status": "verified"}}
    monkeypatch.setattr(paired, "_load_bound", load)
    monkeypatch.setattr(paired, "_reverify_sources", lambda provenance: None)


def test_shared_schedule_and_raw_pooled_metrics(monkeypatch, tmp_path):
    _mock_bound(monkeypatch)
    config, pairs = _config(tmp_path, mutate="heterogeneous")
    result = paired.analyze_config(config, base_dir=tmp_path, config_sha="cfg", gate=_gate())
    draws, expected_sha = paired._common_draws(pairs, 2, 2000, 1234)
    assert result["shared_draw_schedule_sha256"] == expected_sha
    assert result["shared_draw_schedule_shape"] == list(draws.shape) == [2000, 6]
    assert result["shared_across_contrasts_horizons_seeds_methods"] is True
    # The exact point estimate uses aggregate SAE/count per seed, not an
    # unweighted mean of per-window scores under this heterogeneous mask.
    l = result["contrasts"]["A"]["by_horizon"]["1"]["mae"]
    left_scores, right_scores = [], []
    for seed in range(5):
        left_item = paired.st._load_prediction(tmp_path / f"A-left-{seed}.npz")
        right_item = paired.st._load_prediction(tmp_path / f"A-right-{seed}.npz")
        valid = left_item["mask"][:, 0]
        count = valid.sum()
        left_scores.append(np.abs(left_item["predictions"][:, 0] - left_item["targets"][:, 0])[valid].sum() / count)
        right_scores.append(np.abs(right_item["predictions"][:, 0] - right_item["targets"][:, 0])[valid].sum() / count)
    expected = float(np.mean(np.asarray(left_scores) - np.asarray(right_scores)))
    assert l["observed_mean_difference_left_minus_right"] == pytest.approx(expected)
    assert l["random_seed"] == 1234 and l["bootstrap_reps"] == 2000
    assert result["contrasts"]["A"]["by_horizon"]["1"]["rmse"]["random_seed"] == 1234
    assert result["contrasts"]["B"]["by_horizon"]["2"]["mae"]["random_seed"] == 1234
    assert result["contrasts"]["A"]["requested_horizons_equal_weight_pooled_mae"]["metric_aggregation"].startswith("per seed and horizon")


def test_fail_closed_gate_block_and_family_alignment(monkeypatch, tmp_path):
    _mock_bound(monkeypatch)
    config, _ = _config(tmp_path)
    with pytest.raises(ValueError, match="P3A receipt"):
        paired.analyze_config(config, base_dir=tmp_path, config_sha="cfg", gate={**_gate(), "status": "pending"})
    with pytest.raises(ValueError, match="below frozen minimum"):
        bad = dict(config, block_length_windows=1)
        paired.analyze_config(bad, base_dir=tmp_path, config_sha="cfg", gate={**_gate(), "block_length_windows": 1})
    # Make one contrast use a different set of prediction files with a changed
    # target; cross-contrast family alignment must reject it.
    bad, _ = _config(tmp_path / "other", mutate="target")
    config = dict(config)
    config["contrasts"] = [dict(config["contrasts"][0]), dict(bad["contrasts"][1])]
    for c in config["contrasts"]:
        for side in ("left", "right"):
            for item in c[side]:
                item["path"] = str(Path(item["path"]).resolve())
    with pytest.raises(ValueError, match="share exact origins, targets, and masks"):
        paired.analyze_config(config, base_dir=tmp_path, config_sha="cfg", gate=_gate())


def test_block_must_fit_each_trajectory(monkeypatch, tmp_path):
    _mock_bound(monkeypatch)
    config, _ = _config(tmp_path)
    config["block_length_windows"] = 4
    with pytest.raises(ValueError, match="exceeds shortest trajectory"):
        paired.analyze_config(config, base_dir=tmp_path, config_sha="cfg", gate={**_gate(), "block_length_windows": 4})
