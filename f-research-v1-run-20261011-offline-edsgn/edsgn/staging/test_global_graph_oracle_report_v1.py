import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import global_graph_oracle_report_v1 as report
import global_frozen_test_completion_v1 as completion_module
from global_frozen_test_completion_v1 import verify_completion


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    elif isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}


def fixture(root):
    job_id = "graph-toy"
    dataset = root / "data/syn"
    trajectory = dataset / "test_seed200.npz"
    meta = {"seed": 200, "n_nodes": 2, "case": "identity", "switch_index": 3,
            "gradual_length": 128}
    outer = {"partition": "test", "event_kind": "synthetic", "metadata": meta,
             "history": 2, "hmax": 7, "horizons": [1, 3, 7], "splits": {"test": [3, 15]}}
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory, metadata_json=np.asarray(json.dumps(outer)),
                        raw_values=np.zeros((16, 2, 1), dtype=np.float32))
    oracle_dir = dataset / "oracle"
    oracle_dir.mkdir()
    oracle = oracle_dir / "test_seed200.oracle.npz"
    before = np.asarray([[0, 1], [0, 0]], dtype=np.float32)
    after = np.asarray([[0, 0], [1, 0]], dtype=np.float32)
    np.savez_compressed(oracle, P_before=before, P_after=after,
                        switch_index=np.asarray(3, dtype=np.int64),
                        metadata_json=np.asarray(json.dumps({**meta, "oracle_role": "evaluation_only"})))
    authority_path = _write(root / "protocols/oracle-authority.json", {
        "schema_version": 1, "scope": "prospective_graph_oracle_reporting_authority",
        "status": "frozen_before_test", "jobs": [{"job_id": job_id,
            "dataset_root": "data/syn", "regime_id": "identity",
            "sources": [{"dataset_index": 0, "trajectory_seed_id": 200,
                         "trajectory": _ref(root, trajectory), "oracle": _ref(root, oracle)}]}]})
    source = _write(root / "edsgn/staging/frozen_graph_state_diagnostic_v8.py", "toy source")
    report_source = _write(root / report.REPORTER_SOURCE_REL, Path(report.__file__).read_bytes())
    completion_source = _write(root / report.COMPLETION_SOURCE_REL, Path(completion_module.__file__).read_bytes())
    config = _write(root / "configs/toy.json", {"seed": 0})
    checkpoint = _write(root / "checkpoints/toy.pt", b"opaque")
    authority_ref = _ref(root, authority_path)
    binding_path = "runs/binding.json"
    job_path = "runs/jobs/graph-toy.json"
    output_dir = "runs/output/graph-toy"
    report_rel = output_dir + "/frozen-graph-state-diagnostic.json"
    principal = [report_rel, output_dir + "/predictions.npz", output_dir + "/counter-rows.json",
                 output_dir + "/trace-rows.jsonl.gz", output_dir + "/metrics_by_horizon.csv"]
    queue_path = _write(root / "protocols/global-queue.json", {
        "schema_version": 1, "scope": "prospective_complete_test_job_set", "status": "frozen_before_test",
        "queue_id": "q1", "jobs": [{"job_id": job_id, "job_spec_path": job_path,
            "pretest_execution_binding_paths": [binding_path], "driver": "graph_v8", "output_dir": output_dir,
            "expected_origin_count": 6, "coverage_field": "full_test_origin_count",
            "expected_report_scope": "frozen_test_replay_oracle_free", "expected_phase": "evaluation",
            "report_path": report_rel, "phase_receipt_alias_path": "runs/evidence/graph-toy.json",
            "principal_files": principal,
            "pretest_refs": [_ref(root, p) for p in (source, report_source, completion_source, config, checkpoint,
                                                        authority_path, trajectory, oracle)]}]})
    queue_ref = _ref(root, queue_path)
    binding = _write(root / binding_path, {"schema_version": 2, "global_test_queue": queue_ref,
        "plan_path": config.relative_to(root).as_posix(), "plan_sha256": _sha(config),
        "runs": [{"config": _ref(root, config), "checkpoint": _ref(root, checkpoint)}]})
    job = _write(root / job_path, {"schema_version": 2, "job_id": job_id, "driver": "graph_v8",
        "argv": ["--checkpoint", checkpoint.relative_to(root).as_posix(), "--dataset-path", dataset.relative_to(root).as_posix(),
                 "--output", output_dir, "--policy", "frozen", "--device", "cpu"],
        "sources": [_ref(root, source), _ref(root, report_source), _ref(root, completion_source),
                    authority_ref, _ref(root, trajectory), _ref(root, oracle)],
        "pretest_execution_bindings": [_ref(root, binding)], "phase_receipt_alias": "runs/evidence/graph-toy.json"})
    out = root / output_dir
    out.mkdir(parents=True)
    origins = np.asarray([[0, i] for i in range(2, 8)], dtype=np.int64)
    np.savez_compressed(out / "predictions.npz", predictions=np.zeros((6, 7, 2, 1)),
        targets=np.zeros((6, 7, 2, 1)), target_mask=np.ones((6, 7, 2, 1), dtype=bool),
        origins=origins, seed_id=np.asarray(0), trajectory_seed_id=np.asarray([200] * 6))
    _write(out / "counter-rows.json", "[]")
    _write(out / "metrics_by_horizon.csv", "horizon,mae\n1,0\n")
    trace = out / "trace-rows.jsonl.gz"
    with gzip.open(trace, "wt", encoding="utf-8") as stream:
        for origin in range(2, 8):
            for step in range(2):
                stream.write(json.dumps({"dataset_index": 0, "origin": origin, "history_step": step,
                    "input_time_index": origin - 1 + step,
                    "graph_rows": [{"node_id": 0, "new_support_ids": [1]}]}) + "\n")
    fingerprint = {p.relative_to(dataset).as_posix(): _sha(p) for p in sorted(dataset.rglob("*")) if p.is_file()}
    report_path = _write(out / "frozen-graph-state-diagnostic.json", {
        "scope": "frozen_test_replay_oracle_free", "split": "test", "full_test_origin_count": 6,
        "trace_rows_file": trace.name, "trace_row_count": 12,
        "test_collection_fingerprint": fingerprint, "test_collection_fingerprint_final": fingerprint,
        "oracle_accessed": False, "oracle_npz_decoded": False,
        "oracle_analysis_status": "deferred_until_global_completion"})
    outputs = [{"path": p.relative_to(root).as_posix(), "sha256": _sha(p)} for p in sorted(out.iterdir())]
    child_id = "random-child-uuid"
    original = _write(root / "logs/offline-phase-wall/random-child-uuid.json", {
        "scope": "offline-experiment-phase", "status": "completed", "job_id": child_id,
        "phase": "evaluation", "output_dir": str(out.resolve()), "origin_count": 6})
    alias = _write(root / "runs/evidence/graph-toy.json", original.read_bytes())
    snapshot = {p.relative_to(root).as_posix(): _sha(p) for p in
                (queue_path, source, report_source, completion_source, config, checkpoint, authority_path,
                 trajectory, oracle, binding, job)}
    dispatch = _write(root / "logs/verified-frozen-dispatch/graph-toy.json", {
        "schema_version": 2, "job_id": job_id, "driver": "graph_v8",
        "status": "accepted-output-pending-scientific-review", "child_returncode": 0,
        "before": {"files": snapshot}, "after": {"files": snapshot},
        "validator_before": {"returncode": 0}, "validator_after": {"returncode": 0},
        "child_phase_receipt": {"path": original.relative_to(root).as_posix(), "sha256": _sha(original),
            "job_id": child_id, "phase": "evaluation", "status": "completed", "output_dir": str(out.resolve()),
            "principal_artifacts": {Path(x).name: _sha(root / x) for x in principal[:3]},
            "receipt_doc": json.loads(original.read_text()), "phase_receipt_original_path": original.relative_to(root).as_posix(),
            "phase_receipt_original_sha256": _sha(original), "phase_receipt_alias_path": alias.relative_to(root).as_posix(),
            "phase_receipt_alias_sha256": _sha(alias)},
        "phase_receipt_original_path": original.relative_to(root).as_posix(),
        "phase_receipt_original_sha256": _sha(original), "phase_receipt_alias_path": alias.relative_to(root).as_posix(),
        "phase_receipt_alias_sha256": _sha(alias)})
    completion = _write(root / "protocols/completion.json", {"schema_version": 1,
        "scope": "global_frozen_test_completion", "status": "accepted_by_root", "accepted_by": "root", "verdict": "pass",
        "queue": queue_ref, "jobs": [{"job_id": job_id, "job_spec": _ref(root, job),
            "dispatcher_receipt": _ref(root, dispatch), "phase_receipt_original": _ref(root, original),
            "phase_receipt_alias": _ref(root, alias), "driver_report": _ref(root, report_path), "output_files": outputs}]})
    return root, _ref(root, completion), queue_ref, authority_ref


def test_oracle_npz_is_not_opened_before_global_completion(tmp_path, monkeypatch):
    root, cr, qr, ar = fixture(tmp_path)
    calls = []
    def blocked(*args, **kwargs):
        calls.append(args)
        raise report.CompletionError("completion not accepted")
    monkeypatch.setattr(report, "verify_completion", blocked)
    monkeypatch.setattr(report.np, "load", lambda *a, **k: pytest.fail("NPZ loaded before closure"))
    with pytest.raises(report.CompletionError):
        report.analyze_global_oracle_report(root, cr, qr, ar, "runs/oracle-report")
    assert len(calls) == 1


def test_boolean_trajectory_seed_id_rejected_before_any_npz_decode(tmp_path, monkeypatch):
    root, cr, qr, ar = fixture(tmp_path)
    authority_path = root / ar["path"]
    authority = json.loads(authority_path.read_text())
    authority["jobs"][0]["sources"][0]["trajectory_seed_id"] = True
    _write(authority_path, authority)
    ar = _ref(root, authority_path)

    job_path = root / "runs/jobs/graph-toy.json"
    job = json.loads(job_path.read_text())
    for row in job["sources"]:
        if row["path"] == ar["path"]:
            row["sha256"] = ar["sha256"]
    _write(job_path, job)
    queue_path = root / qr["path"]
    queue = json.loads(queue_path.read_text())
    for row in queue["jobs"][0]["pretest_refs"]:
        if row["path"] == ar["path"]:
            row["sha256"] = ar["sha256"]
    _write(queue_path, queue)
    qr = _ref(root, queue_path)

    monkeypatch.setattr(report, "verify_completion", lambda *args: {})
    calls = []
    monkeypatch.setattr(report.np, "load", lambda *args, **kwargs: calls.append(args) or pytest.fail("NPZ decoded before authority schema rejection"))
    with pytest.raises(report.OracleReportError, match="trajectory_seed_id must be a nonnegative integer"):
        report.analyze_global_oracle_report(root, cr, qr, ar, "runs/oracle-report")
    assert calls == []


def test_full_toy_completion_then_sampled_alignment_and_fences(tmp_path):
    root, cr, qr, ar = fixture(tmp_path)
    assert verify_completion(root, cr, qr)
    result = report.analyze_global_oracle_report(root, cr, qr, ar, "runs/oracle-report")
    assert result["job_count"] == 1 and result["scope"].startswith("global_postcompletion")
    report_doc = json.loads((root / result["report_path"]).read_text())
    job = report_doc["jobs"][0]
    assert job["full_origin_count"] == 6 and job["full_trace_row_count"] == 12
    assert job["sampled_alignment_row_count"] == 12
    assert job["alignment_scope"].endswith("not full-window edge recovery")
    rows_path = root / job["alignment_path"]
    with gzip.open(rows_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert rows[0]["intersection_count"] == 1
    assert rows[0]["relative_to_switch"] == -2
    assert len(result["output_files"]) == 2


def test_abrupt_and_gradual_transition_boundaries():
    p0 = np.asarray([[0, 1], [0, 0]], dtype=float)
    p1 = np.asarray([[0, 0], [1, 0]], dtype=float)
    abrupt = {"case": "identity", "switch_index": 4}
    assert np.array_equal(report._expected_transition(abrupt, p0, p1, 3)[0], p0)
    assert np.array_equal(report._expected_transition(abrupt, p0, p1, 4)[0], p1)
    gradual = {"case": "gradual", "switch_index": 4, "gradual_length": 2}
    got0, a0 = report._expected_transition(gradual, p0, p1, 3)
    got1, a1 = report._expected_transition(gradual, p0, p1, 4)
    got2, a2 = report._expected_transition(gradual, p0, p1, 5)
    assert a0 == 0 and np.array_equal(got0, p0)
    assert a1 == 0.5 and np.allclose(got1, 0.5 * p0 + 0.5 * p1)
    assert a2 == 1 and np.array_equal(got2, p1)


def test_exclusive_output_and_authority_mapping_failure(tmp_path):
    root, cr, qr, ar = fixture(tmp_path)
    output = root / "runs/oracle-report"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        report.analyze_global_oracle_report(root, cr, qr, ar, "runs/oracle-report")
    output.rmdir()
    authority = json.loads((root / ar["path"]).read_text())
    authority["jobs"][0]["sources"][0]["dataset_index"] = 1
    _write(root / ar["path"], authority)
    ar["sha256"] = _sha(root / ar["path"])
    with pytest.raises(Exception):
        report.analyze_global_oracle_report(root, cr, qr, ar, "runs/oracle-report")


def test_source_pin_missing_fails_before_npz_open(tmp_path, monkeypatch):
    root, cr, qr, ar = fixture(tmp_path)
    job_path = root / "runs/jobs/graph-toy.json"
    job = json.loads(job_path.read_text())
    job["sources"] = [x for x in job["sources"] if x["path"] != report.REPORTER_SOURCE_REL]
    _write(job_path, job)
    monkeypatch.setattr(report, "verify_completion", lambda *args: {})
    monkeypatch.setattr(report.np, "load", lambda *a, **k: pytest.fail("NPZ loaded before source-pin rejection"))
    with pytest.raises(report.OracleReportError, match="source hash is not pretest-pinned"):
        report.analyze_global_oracle_report(root, cr, qr, ar, "runs/oracle-report")


def test_source_tamper_fails_before_global_decode(tmp_path, monkeypatch):
    root, cr, qr, ar = fixture(tmp_path)
    source = root / report.REPORTER_SOURCE_REL
    source.write_bytes(source.read_bytes() + b"tamper")
    monkeypatch.setattr(report, "verify_completion", lambda *args: pytest.fail("completion should not run on changed helper"))
    monkeypatch.setattr(report.np, "load", lambda *a, **k: pytest.fail("NPZ loaded after source tamper"))
    with pytest.raises(report.OracleReportError, match="RUN source bytes differ"):
        report.analyze_global_oracle_report(root, cr, qr, ar, "runs/oracle-report")


def test_unlisted_collection_member_added_during_decode_fails_final_fence(tmp_path, monkeypatch):
    root, cr, qr, ar = fixture(tmp_path)
    original = report._read_oracle
    def add_member_after_decode(*args, **kwargs):
        value = original(*args, **kwargs)
        (root / "data/syn/unlisted-after-start.txt").write_text("late member", encoding="utf-8")
        return value
    monkeypatch.setattr(report, "_read_oracle", add_member_after_decode)
    with pytest.raises(report.OracleReportError, match="collection membership/hash changed"):
        report.analyze_global_oracle_report(root, cr, qr, ar, "runs/oracle-report")


@pytest.mark.parametrize("artifact", ["alignment", "report"])
def test_final_output_mutation_during_closure_fails_fixed_hash_fence(tmp_path, monkeypatch, artifact):
    root, cr, qr, ar = fixture(tmp_path)
    original = report.verify_completion
    calls = 0
    def mutate_after_report(*args):
        nonlocal calls
        calls += 1
        result = original(*args)
        if calls == 3:
            target = (root / "runs/oracle-report/graph-toy-structural-alignment.jsonl.gz"
                      if artifact == "alignment" else root / "runs/oracle-report/global-graph-oracle-report.json")
            assert target.is_file()
            target.write_bytes(target.read_bytes() + b"post-write mutation")
        return result
    monkeypatch.setattr(report, "verify_completion", mutate_after_report)
    with pytest.raises(report.OracleReportError, match="output membership/hash fence"):
        report.analyze_global_oracle_report(root, cr, qr, ar, "runs/oracle-report")
    assert calls == 3
