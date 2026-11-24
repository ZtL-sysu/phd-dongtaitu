import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_email_episode_simple.py"
spec = importlib.util.spec_from_file_location("p3_email_episode_simple_under_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def fixture_root(tmp_path, *, p2="passed", p3="running"):
    root = tmp_path / "run"
    (root / "logs").mkdir(parents=True)
    (root / "pipeline-state.json").write_text(json.dumps({"stages": [
        {"id": "P2", "status": p2}, {"id": "P3", "status": p3},
    ]}), encoding="utf-8")
    dataset = root / module.DATASET_REL
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"mock-npz-bytes; never parsed")
    manifest = {
        "dataset": "email-episode-v1",
        "episode": {"legal_origins": {"train": 334, "validation": 46}},
        "decision_roles": {"model_and_preprocessing_fit": "train [0,368) only"},
    }
    (dataset.parent / "data-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def fake_binding_api(calls):
    def prepare(config_path, *, output_dir):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        assert config["split"] == "val" and config["fit_partition"] == "train"
        assert config["expected_validation_origins"] == 46
        assert config["runner_source"]["sha256"] == module.sha256_file(module.__file__)
        calls.append(("prepare", Path(config_path), Path(output_dir)))
        binding = Path(output_dir) / "mock-binding.json"
        binding.write_text("{}", encoding="utf-8")
        return binding

    def verify(path):
        calls.append(("verify", Path(path)))
        return {"status": "verified", "dataset_file_count": 2, "code_file_count": 1}
    return prepare, verify


def fake_baseline(calls):
    def run(dataset_path, output_dir, split):
        assert Path(dataset_path).name == "email-episode.npz"
        assert split == "val"
        out = Path(output_dir)
        calls.append(("baseline", split))
        (out / "simple-baseline-manifest.json").write_text(json.dumps({"test_accessed": False}), encoding="utf-8")
        np.save(out / "linear_coefficients.npy", np.zeros((2, 2), dtype=np.float32))
        for method in ("persistence", "linear"):
            np.savez(out / f"{method}_val_predictions.npz",
                     origins=np.stack([np.zeros(46, dtype=np.int64), np.arange(46)], axis=1),
                     predictions=np.zeros((46, 1, 1, 1)), targets=np.zeros((46, 1, 1, 1)),
                     target_mask=np.ones((46, 1, 1, 1), dtype=bool))
            (out / f"{method}_val_metrics.csv").write_text("toy\n", encoding="utf-8")
    return run


def test_script_is_stage_gated_before_baseline_or_binding(tmp_path):
    root = fixture_root(tmp_path, p2="running", p3="pending")
    calls = []
    with pytest.raises(RuntimeError, match="P2=passed/P3=running"):
        module.run(root, baseline_fn=fake_baseline(calls), binding_api=fake_binding_api(calls))
    assert calls == []
    assert not (root / module.OUTPUT_REL).exists()


def test_mocked_success_binds_and_runs_train_val_only(tmp_path):
    root = fixture_root(tmp_path)
    calls = []
    ticks = iter([10., 13.25])
    row = module.run(root, baseline_fn=fake_baseline(calls), binding_api=fake_binding_api(calls), clock=lambda: next(ticks))
    assert [call[0] for call in calls] == ["prepare", "baseline", "verify"]
    assert row == {"dataset": "email-episode", "output": str(root / module.OUTPUT_REL),
                   "wall_seconds": 3.25,
                   "scope": "full_training_fit/full_validation_46_origins; test not accessed",
                   "test_accessed": False}
    summary_path = root / module.SUMMARY_REL
    assert json.loads(summary_path.read_text(encoding="utf-8")) == [row]
    assert set(row) == {"dataset", "output", "wall_seconds", "scope", "test_accessed"}


def test_existing_prediction_or_summary_refuses_before_binding(tmp_path):
    root = fixture_root(tmp_path)
    output = root / module.OUTPUT_REL
    output.mkdir(parents=True)
    (output / "linear_val_predictions.npz").write_bytes(b"old evidence")
    calls = []
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        module.run(root, baseline_fn=fake_baseline(calls), binding_api=fake_binding_api(calls))
    assert calls == []


def test_existing_summary_is_immutable(tmp_path):
    root = fixture_root(tmp_path)
    summary = root / module.SUMMARY_REL
    summary.write_text("[]", encoding="utf-8")
    calls = []
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        module.run(root, baseline_fn=fake_baseline(calls), binding_api=fake_binding_api(calls))
    assert calls == []


def test_manifest_must_match_episode_origin_contract(tmp_path):
    root = fixture_root(tmp_path)
    sidecar = root / module.DATASET_REL.parent / "data-manifest.json"
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["episode"]["legal_origins"]["validation"] = 45
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    with pytest.raises(ValueError, match="origin contract mismatch"):
        module.run(root, baseline_fn=fake_baseline(calls), binding_api=fake_binding_api(calls))
    assert calls == []
