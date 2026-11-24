import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "staging"))

from frozen_validation_calibration_v2 import (
    _canonical_stage_gate, _epsilon_contract, _finalize_phase, _safe_val_collection,
    _verify_hash_snapshot, _require_fp32_payload, metadata_test_support, _queue,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_canonical_gate_requires_verdict_and_hash_bound_evidence(tmp_path):
    evidence = tmp_path / "accepted.md"; evidence.write_text("approved", encoding="utf-8")
    state = {"stages": [
        {"id":"P2","status":"passed","gate_verdict":"pass"},
        {"id":"P3","status":"passed","gate_verdict":"pass"},
        {"id":"P3A","status":"passed","gate_verdict":"pass",
         "inputs":[],"evidence":[{"path":"accepted.md","sha256":_sha(evidence)}]},
        {"id":"P4","status":"running"},
    ]}
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state), encoding="utf-8")
    assert _canonical_stage_gate(tmp_path)["stages"][2]["gate_verdict"] == "pass"
    state["stages"][2]["gate_verdict"] = "fail"
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="P3A"):
        _canonical_stage_gate(tmp_path)
    state["stages"][2]["gate_verdict"] = "pass"
    state["stages"][2]["evidence"][0]["sha256"] = "0" * 64
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale"):
        _canonical_stage_gate(tmp_path)


def test_epsilon_acceptance_binds_dataset_checkpoint_config_and_queue(tmp_path):
    baseline = tmp_path / "baseline.json"; baseline.write_text("{}", encoding="utf-8")
    doc = {"accepted": True, "accepted_by":"root", "scope":"validation-only-epsilon",
           "split":"val", "epsilon":0.02, "metric":"pooled_raw_mae",
           "required_horizons":[1,3,7], "baseline_evidence_sha256":_sha(baseline),
           "baseline_evidence_path":str(baseline), "application_relevance_rationale":"toy rationale",
           "dataset_path":str(tmp_path/"dataset"), "checkpoint_sha256":"cp",
           "config_sha256":"cfg", "queue_sha256":"queue",
           "test_support_by_trajectory":{"100":20}}
    assert _epsilon_contract(doc, dataset=doc["dataset_path"], checkpoint_sha="cp",
                             config_sha="cfg", queue_sha="queue")["epsilon"] == 0.02
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _epsilon_contract(doc, dataset=doc["dataset_path"], checkpoint_sha="wrong",
                          config_sha="cfg", queue_sha="queue")


def test_candidate_queue_allows_at_most_twelve_registered_pilots():
    assert len(_queue({"scheduler_pilots":[{"id":f"s{i:02d}"} for i in range(11)]})) == 11
    with pytest.raises(ValueError, match="1 through 12"):
        _queue({"scheduler_pilots":[{"id":f"s{i:02d}"} for i in range(13)]})


def test_single_npz_rejected_to_avoid_loading_test_targets(tmp_path):
    source = tmp_path / "combined.npz"; source.write_bytes(b"not opened")
    with pytest.raises(RuntimeError, match="test targets/masks"):
        _safe_val_collection(source)


def test_test_support_uses_metadata_member_without_opening_target_arrays(tmp_path, monkeypatch):
    source = tmp_path / "test_seed200.npz"
    meta = {"partition":"test","history":2,"hmax":3,
            "metadata":{"seed":200,"steps":20},"splits":{"test":[0,20]}}
    npz = __import__("numpy").savez_compressed
    npz(source, metadata_json=__import__("numpy").asarray(json.dumps(meta)),
        targets=__import__("numpy").arange(1000), target_mask=__import__("numpy").ones(1000))
    import frozen_validation_calibration_v2 as module
    real_load = module.np.load
    touched = []
    class ArchiveProxy:
        def __init__(self, archive): self.archive, self.files = archive, archive.files
        def __enter__(self): return self
        def __exit__(self, *args): self.archive.close()
        def __getitem__(self, key):
            touched.append(key)
            assert key == "metadata_json"
            return self.archive[key]
    monkeypatch.setattr(module.np, "load", lambda *a, **k: ArchiveProxy(real_load(*a, **k)))
    assert metadata_test_support(tmp_path) == {"200": 16}
    assert touched == ["metadata_json"]


def test_foreign_lock_preserved_and_72h_budget_fail_closed(tmp_path, monkeypatch):
    import simple_baseline_freeze_v1 as helpers
    monkeypatch.setattr(helpers, "_owned_model_workers", lambda root: [])
    monkeypatch.setattr(helpers, "_budget_report", lambda root, out: {"coverage_complete":True,"spent_seconds":0})
    foreign = tmp_path / "logs/model-resource.lock"; foreign.parent.mkdir(parents=True)
    foreign.write_text('{"owner":"foreign"}', encoding="utf-8")
    with pytest.raises(Exception):
        helpers._acquire_launch_guard(tmp_path, tmp_path/"out", "calibration", "foreign-test",
                                      0.0, 1.0, ["toy"])
    assert foreign.exists() and json.loads(foreign.read_text())["owner"] == "foreign"
    foreign.unlink()
    monkeypatch.setattr(helpers, "_budget_report", lambda root, out: {"coverage_complete":True,"spent_seconds":72*3600})
    with pytest.raises(Exception, match="72-hour"):
        helpers._acquire_launch_guard(tmp_path, tmp_path/"out", "calibration", "budget-test",
                                      0.0, 1.0, ["toy"])
    assert not (tmp_path / "logs/model-resource.lock").exists()
    assert not (tmp_path / "logs/offline-phase-wall/campaign.lock").exists()
    monkeypatch.setattr(helpers, "_budget_report",
                        lambda root, out: (_ for _ in ()).throw(RuntimeError("coverage incomplete")))
    with pytest.raises(Exception, match="coverage"):
        helpers._acquire_launch_guard(tmp_path, tmp_path/"out", "calibration", "coverage-test",
                                      0.0, 1.0, ["toy"])
    assert not (tmp_path / "logs/model-resource.lock").exists()


def test_failure_receipt_and_lease_cleanup(tmp_path):
    receipt_dir = tmp_path / "logs/offline-phase-wall"; receipt_dir.mkdir(parents=True)
    lock = tmp_path / "logs/model-resource.lock"; lock.write_text("{}", encoding="utf-8")
    campaign = receipt_dir / "campaign.lock"; campaign.write_text("{}", encoding="utf-8")
    class Guard:
        token = "token"
        released = False
        def release(self):
            self.released = True
            lock.unlink(missing_ok=True); campaign.unlink(missing_ok=True)
    guard = Guard()
    helpers = SimpleNamespace(_phase_receipt=__import__("simple_baseline_freeze_v1")._phase_receipt)
    with pytest.raises(ValueError):
        try:
            raise ValueError("toy operation failed")
        except Exception as exc:
            _finalize_phase(helpers, tmp_path, tmp_path/"out", "job-1", 1.0, 1.0,
                            "failed", ["toy"], exc, guard, {"input_sha256_before":{}})
            raise
    receipt = json.loads((receipt_dir/"job-1.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed" and receipt["exception_type"] == "ValueError"
    assert guard.released is True and not lock.exists() and not campaign.exists()


def test_hash_snapshot_detects_source_mutation_and_fp32_is_explicit(tmp_path):
    source = tmp_path / "source"; source.write_text("a", encoding="utf-8")
    before = {str(source): _sha(source)}
    source.write_text("b", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bytes changed"):
        _verify_hash_snapshot(before, {str(source): _sha(source)})
    import torch
    _require_fp32_payload({"state_dict":{"p":torch.ones(2,dtype=torch.float32)}})
    with pytest.raises(ValueError, match="FP32"):
        _require_fp32_payload({"state_dict":{"p":torch.ones(2,dtype=torch.float16)}})
