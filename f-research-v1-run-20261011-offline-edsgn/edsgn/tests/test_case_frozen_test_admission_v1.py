"""CPU metadata-only tests for case_v3 pretest admission primitives."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "staging"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import case_frozen_test_admission_v1 as admission


def put(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def ref(path: Path):
    root = next(parent for parent in path.parents if parent.name.startswith("test_"))
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def domain_fixture(root):
    collection = root / "data/synthetic-stress/test"
    trajectory = put(collection / "test_seed001.npz", b"opaque test archive bytes")
    domain = put(root / "protocols/domain.json", {
        "schema": "edsgn.split-history-domain.v1",
        "dataset": {"source_name": "noise_only", "event_kind": "synthetic", "partition": "test",
                    "shape_tnf": [20, 4, 1], "history_length": 3, "hmax": 2},
        "trajectories": [{"trajectory_index": 0, "split": "test", "origin_first": 2,
                           "origin_last": 17, "origin_count": 16}],
    })
    domain_ref = ref(domain)
    descriptor = put(root / "protocols/descriptor.json", {
        "schema": "edsgn.c06-domain-preflight.v1", "split": "test",
        "domain_manifest_path": domain.relative_to(root).as_posix(),
        "domain_manifest_sha256": domain_ref["sha256"],
        "ordered_collection": [{"trajectory_index": 0, "path": trajectory.relative_to(root).as_posix(),
                                 "archive_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                                 "shape_tnf": [20, 4, 1]}],
        "source_sha256": {},
    })
    return domain, descriptor, trajectory


def test_case_domain_hashes_opaque_archive_without_npz_decode(tmp_path, monkeypatch):
    domain, descriptor, archive = domain_fixture(tmp_path)
    monkeypatch.setattr(admission, "_json", admission._json)
    consumed = {}
    got_domain, got_descriptor, collection = admission._validate_case_domain(
        tmp_path, "synthetic", ref(domain), ref(descriptor), consumed)
    assert got_domain["dataset"]["partition"] == "test"
    assert got_descriptor["ordered_collection"][0]["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert collection == archive.parent
    assert consumed[archive.relative_to(tmp_path).as_posix()] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_case_domain_rejects_extra_or_reordered_test_archive(tmp_path):
    domain, descriptor, _ = domain_fixture(tmp_path)
    put(domain.parents[1] / "data/synthetic-stress/test/test_seed002.npz", b"extra opaque bytes")
    with pytest.raises(admission.AdmissionError, match="exact ordered test"):
        admission._validate_case_domain(tmp_path, "synthetic", ref(domain), ref(descriptor), {})


def test_case_domain_rejects_descriptor_hash_change(tmp_path):
    domain, descriptor, archive = domain_fixture(tmp_path)
    doc = json.loads(descriptor.read_text())
    doc["ordered_collection"][0]["archive_sha256"] = "0" * 64
    put(descriptor, doc)
    with pytest.raises(admission.AdmissionError, match="C06 row"):
        admission._validate_case_domain(tmp_path, "synthetic", ref(domain), ref(descriptor), {})


def test_validation_selection_requires_root_acceptance_and_full_origin_record(tmp_path):
    checkpoint = put(tmp_path / "runs/train/checkpoint.pt", b"opaque checkpoint")
    origin = put(tmp_path / "reviews/validation-origins.json", {
        "split": "validation", "origins_complete_and_ordered": True, "full_validation_origin_count": 8,
        "checkpoint_sha256": admission._sha(checkpoint), "dataset_fingerprint": "f" * 64,
        "source_manifest_sha256": "a" * 64, "node_order_sha256": "b" * 64,
    })
    selection = put(tmp_path / "reviews/selection.json", {
        "schema_version": 1, "scope": admission.SELECTION_SCOPE, "status": "accepted",
        "accepted_by": "root", "test_feedback_used": False, "dataset_id": "synthetic",
        "training_dataset_path": "data/synthetic", "dataset_fingerprint": "f" * 64,
        "source_manifest_sha256": "a" * 64, "node_order_sha256": "b" * 64,
        "variant": "V4", "model_seed": 0, "split": "validation",
        "checkpoint_sha256": admission._sha(checkpoint), "full_validation_origin_count": 8,
        "validation_origin_record": ref(origin), "selection_rule": {"focal": "largest_training_activity_score_then_node_id",
          "neighbor": "largest_mean_valid_neighbor_weight_in_full_validation_then_node_id"},
        "focal_node": "n0", "neighbor_node": "n1",
    })
    selection_ref = ref(selection)
    config_source = put(tmp_path / "configs/train.json", b"toy config")
    training_binding = put(tmp_path / "runs/train/input-binding.json", {"dataset_sources": [], "code_sources": [],
        "config_source": {"path": ref(config_source)["path"], "sha256": ref(config_source)["sha256"]}})
    source_sha = "a" * 64
    acceptance = put(tmp_path / "reviews/root-selection.json", {
        "schema_version": 1, "scope": "case_signal_selection_root_acceptance",
        "status": "accepted", "accepted_by": "root", "verdict": "pass",
        "test_feedback_used": False,
        "evidence_sha256": {selection_ref["path"]: selection_ref["sha256"],
            ref(origin)["path"]: ref(origin)["sha256"], ref(training_binding)["path"]: ref(training_binding)["sha256"],
            ref(tmp_path / "configs/train.json")["path"]: ref(tmp_path / "configs/train.json")["sha256"],
            ref(checkpoint)["path"]: ref(checkpoint)["sha256"]},
    })
    consumed = {}
    got = admission._validate_selection(tmp_path, "synthetic", selection_ref, ref(acceptance),
        {"dataset_path": "data/synthetic", "output_dir": "runs/train"}, admission._sha(checkpoint), source_sha,
        training_binding, {}, consumed)
    assert got["focal_node"] == "n0"
    assert len(consumed) == 6


def test_selection_rejects_test_feedback_and_checkpoint_mismatch(tmp_path):
    checkpoint = put(tmp_path / "checkpoint.pt", b"opaque")
    origin = put(tmp_path / "origin.json", {"split": "validation", "complete_ordered": True,
        "full_validation_origin_count": 1, "origins_complete_and_ordered": True,
        "checkpoint_sha256": admission._sha(checkpoint),
        "dataset_fingerprint": "f"*64, "source_manifest_sha256": "a"*64, "node_order_sha256": "b"*64})
    selection = put(tmp_path / "selection.json", {"schema_version": 1,
        "scope": admission.SELECTION_SCOPE, "status": "accepted", "accepted_by": "root",
        "test_feedback_used": True, "dataset_id": "synthetic", "training_dataset_path": "data/synthetic",
        "dataset_fingerprint": "f"*64, "source_manifest_sha256": "a"*64, "node_order_sha256": "b"*64,
        "variant": "V4", "model_seed": 0, "split": "validation", "checkpoint_sha256": admission._sha(checkpoint),
        "full_validation_origin_count": 1, "validation_origin_record": ref(origin),
        "selection_rule": "fixed", "focal_node": 0, "neighbor_node": 1})
    acc = put(tmp_path / "accept.json", {"schema_version": 1, "scope": "case_signal_selection_root_acceptance",
        "status": "accepted", "accepted_by": "root", "verdict": "pass", "test_feedback_used": False,
        "evidence_sha256": {ref(selection)["path"]: ref(selection)["sha256"], ref(origin)["path"]: ref(origin)["sha256"]}})
    with pytest.raises(admission.AdmissionError, match="validation-only"):
        admission._validate_selection(tmp_path, "synthetic", ref(selection), ref(acc),
            {"dataset_path": "data/synthetic"}, admission._sha(checkpoint), "a"*64,
            checkpoint, {}, {})


def test_public_entry_rejects_without_actual_root_accepted_binding(tmp_path):
    binding = put(tmp_path / "admission.json", {"schema_version": 1})
    job_path = put(tmp_path / "job.json", {})
    with pytest.raises(admission.AdmissionError, match="exact schema"):
        admission.validate_admission(tmp_path, ref(binding), {}, job_path)
