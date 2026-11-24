import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bind_training_inputs
import audit_bound_baseline_numeric_v2 as auditmod


def make_run(tmp_path, monkeypatch):
    dataset = tmp_path / "toy.npz"
    T, N, F = 12, 2, 2
    targets = np.arange(T * N * F, dtype=np.float32).reshape(T, N, F)
    target_mask = np.ones_like(targets, dtype=bool)
    target_mask[7, 1, 1] = False
    meta = {"history": 2, "hmax": 3, "horizons": [1, 3],
            "splits": {"train": [0, 6], "validation": [6, 10], "test": [10, 12]},
            "partition": "chronological"}
    np.savez_compressed(dataset, targets=targets, target_mask=target_mask,
                        metadata_json=np.asarray(json.dumps(meta)))
    run = tmp_path / "run"
    run.mkdir()
    config = {"variant": "GRU", "seed": 7, "dataset_path": str(dataset),
              "history": 2, "hmax": 3, "epochs": 2, "patience": 2,
              "output_dir": str(run)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (run / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    ckpt = run / "checkpoint.pt"
    ckpt.write_bytes(b"toy checkpoint bytes; never deserialized")
    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    # Validation origins: max(L-1, start-1)=5 through end-H-1=6.
    origins = np.asarray([[0, 5], [0, 6]], dtype=np.int64)
    tids = origins[:, 1, None] + np.arange(1, 4)[None, :]
    saved_targets = targets[tids]
    saved_mask = target_mask[tids]
    pred = saved_targets + 1.0
    pred[~saved_mask] = 100.0  # masked values must not affect metrics.
    np.savez_compressed(run / "validation_predictions.npz", predictions=pred,
                        targets=saved_targets, target_mask=saved_mask,
                        origins=origins, seed_id=np.asarray(7))
    rows = []
    for h in (1, 3):
        for f in range(F):
            valid = saved_mask[:, h - 1, :, f]
            residual = pred[:, h - 1, :, f][valid] - saved_targets[:, h - 1, :, f][valid]
            rows.append({"horizon": h, "feature": f,
                         "mae": float(np.abs(residual).mean()),
                         "rmse": float(np.sqrt(np.square(residual).mean())),
                         "n_scored": int(valid.sum()), "n_windows": len(pred), "stratum": "all"})
    with (run / "metrics_by_horizon_val.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (run / "manifest.json").write_text(json.dumps({
        "scope": "full_protocol", "dataset_path": str(dataset),
        "selection": "required-horizon average validation MAE; no test access",
        "train_windows": 6, "val_windows": 2,
    }), encoding="utf-8")
    (run / "training_summary.json").write_text(json.dumps({
        "scope": "full_protocol", "test_accessed": False, "stop_loss_reached": False,
        "epochs_executed": 2, "best_val_mae": 1.0, "checkpoint_sha256": ckpt_sha,
    }), encoding="utf-8")
    with (run / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_selection_mae", "train_windows"])
        writer.writeheader(); writer.writerows([
            {"epoch": 1, "val_selection_mae": 3.0, "train_windows": 6},
            {"epoch": 2, "val_selection_mae": 1.0, "train_windows": 6},
        ])
    (run / "input-binding-at-launch.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(bind_training_inputs, "verify_binding",
                        lambda p: (calls.append(str(p)) or {"status": "verified", "dataset_file_count": 1}))
    return config_path, run, dataset, targets, target_mask, calls


def test_recomputes_required_and_descriptive_all_offsets_separately(tmp_path, monkeypatch):
    config, run, _, _, _, calls = make_run(tmp_path, monkeypatch)
    out = auditmod.audit(config, run)
    assert out["status"] == "passed"
    assert out["required_horizons"] == [1, 3]
    assert out["required_horizon_mean_selection_mae"] == 1.0
    assert out["required_horizon_pooled_sum_count"]["n_valid_cells"] == sum(
        row["n_scored"] for row in out["required_horizon_feature_rows"])
    descriptive = out["all_hmax_offsets_descriptive_only"]
    assert descriptive["label"] == "descriptive_all_offsets_not_selection_metric"
    assert len(descriptive["per_horizon_feature_rows"]) == 6
    assert descriptive["pooled_sum_count"]["n_valid_cells"] > out["required_horizon_pooled_sum_count"]["n_valid_cells"]
    assert out["training"]["best_epoch_from_training_log"] == 2
    assert out["training"]["summary_best_val_mae_matches_log_minimum_and_saved_predictions"] is True
    assert calls and out["test_arrays_or_test_windows_read"] is False


def test_missing_required_horizon_metric_row_fails_closed(tmp_path, monkeypatch):
    config, run, *_ = make_run(tmp_path, monkeypatch)
    path = run / "metrics_by_horizon_val.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(auditmod.AuditError, match="metric rows missing/extra"):
        auditmod.audit(config, run)


def test_saved_validation_targets_must_match_source_at_each_origin(tmp_path, monkeypatch):
    config, run, *_ = make_run(tmp_path, monkeypatch)
    path = run / "validation_predictions.npz"
    with np.load(path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    data["targets"][0, 0, 0, 0] += 2
    np.savez_compressed(path, **data)
    with pytest.raises(auditmod.AuditError, match="Saved target differs"):
        auditmod.audit(config, run)


def test_omitted_validation_origin_is_rejected(tmp_path, monkeypatch):
    config, run, *_ = make_run(tmp_path, monkeypatch)
    path = run / "validation_predictions.npz"
    with np.load(path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    for key in ("predictions", "targets", "target_mask", "origins"):
        data[key] = data[key][:1]
    np.savez_compressed(path, **data)
    with pytest.raises(auditmod.AuditError, match="complete ordered validation origins"):
        auditmod.audit(config, run)


def test_training_summary_selection_must_match_minimum_logged_epoch(tmp_path, monkeypatch):
    config, run, *_ = make_run(tmp_path, monkeypatch)
    path = run / "training_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")); summary["best_val_mae"] = 9.0
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(auditmod.AuditError, match="Minimum logged"):
        auditmod.audit(config, run)


def test_binding_and_checkpoint_hash_are_mandatory(tmp_path, monkeypatch):
    config, run, *_ = make_run(tmp_path, monkeypatch)
    (run / "input-binding-at-launch.json").unlink()
    with pytest.raises(auditmod.AuditError, match="Required run artifacts missing"):
        auditmod.audit(config, run)
