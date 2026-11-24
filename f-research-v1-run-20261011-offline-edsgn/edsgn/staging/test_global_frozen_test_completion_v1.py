import hashlib
import json
from pathlib import Path

import pytest

import global_frozen_test_completion_v1 as completion
from global_frozen_test_completion_v1 import CompletionError, verify_completion, verify_queued_job


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    root = tmp_path
    out = root / "runs/test/job-a"
    report_rel = "runs/test/job-a/frozen-graph-state-diagnostic.json"
    principal = [report_rel, "runs/test/job-a/predictions.npz", "runs/test/job-a/counter-rows.json",
                 "runs/test/job-a/trace-rows.jsonl.gz", "runs/test/job-a/metrics_by_horizon.csv"]
    source = _write(root / "edsgn/staging/frozen_graph_state_diagnostic_v8.py", "# frozen toy source\n")
    pre = _write(root / "runs/pre/config.json", {"seed": 0})
    checkpoint = _write(root / "runs/pre/model.pt", b"opaque checkpoint bytes")
    binding_path = "runs/pre/pretest-binding.json"
    job_path = "runs/jobs/job-a.json"
    output_dir = "runs/test/job-a"
    alias_path = "runs/evidence/job-a-phase.json"

    # Acyclic construction: independent refs -> frozen queue -> binding -> job.
    queue = _write(root / "protocols/global-queue.json", {
        "schema_version": 1, "scope": "prospective_complete_test_job_set",
        "status": "frozen_before_test", "queue_id": "frozen-q1",
        "jobs": [{"job_id": "job-a", "job_spec_path": job_path,
                  "pretest_execution_binding_paths": [binding_path], "driver": "graph_v8",
                  "output_dir": output_dir, "expected_origin_count": 2,
                  "coverage_field": "full_test_origin_count",
                  "expected_report_scope": "frozen_test_replay_oracle_free",
                  "expected_phase": "evaluation", "report_path": report_rel,
                  "phase_receipt_alias_path": alias_path,
                  "principal_files": principal,
                  "pretest_refs": [{"path": p.relative_to(root).as_posix(), "sha256": _sha(p)}
                                   for p in (source, pre, checkpoint)]}]})
    qr = {"path": queue.relative_to(root).as_posix(), "sha256": _sha(queue)}
    source_ref = {"path": source.relative_to(root).as_posix(), "sha256": _sha(source)}
    pre_ref = {"path": pre.relative_to(root).as_posix(), "sha256": _sha(pre)}
    cp_ref = {"path": checkpoint.relative_to(root).as_posix(), "sha256": _sha(checkpoint)}
    binding = _write(root / binding_path, {"schema_version": 2, "global_test_queue": qr,
                                           "plan_path": pre_ref["path"], "plan_sha256": pre_ref["sha256"],
                                           "runs": [{"config": pre_ref, "checkpoint": cp_ref}]})
    binding_ref = {"path": binding.relative_to(root).as_posix(), "sha256": _sha(binding)}
    job = _write(root / job_path, {
        "schema_version": 2, "job_id": "job-a", "driver": "graph_v8",
        "argv": ["--checkpoint", "runs/pre/model.pt", "--dataset-path", "runs/data/test",
                 "--output", output_dir, "--policy", "frozen", "--device", "cpu"],
        "sources": [source_ref], "pretest_execution_bindings": [binding_ref],
        "phase_receipt_alias": alias_path})
    jobref = {"path": job.relative_to(root).as_posix(), "sha256": _sha(job)}
    _write(root / "runs/data/test/metadata.json", {"split": "test"})
    assert set(verify_queued_job(root, qr, job_path)) >= {qr["path"], source_ref["path"], binding_ref["path"]}

    # Files match graph8's actual principal output layout; NPZ remains opaque.
    _write(out / "frozen-graph-state-diagnostic.json", {
        "scope": "frozen_test_replay_oracle_free", "split": "test",
        "full_test_origin_count": 2, "oracle_accessed": False,
        "oracle_npz_decoded": False,
        "oracle_analysis_status": "deferred_until_global_completion"})
    _write(out / "predictions.npz", b"opaque prediction artifact")
    _write(out / "counter-rows.json", "[]")
    _write(out / "trace-rows.jsonl.gz", b"opaque gzip artifact")
    _write(out / "metrics_by_horizon.csv", "horizon,mae\n1,0.2\n")

    child_id = "h2-matched-retention-9f0c057d-actual-random-child-id"
    original = _write(root / "logs/offline-phase-wall" / f"{child_id}.json", {
        "scope": "offline-experiment-phase", "status": "completed", "job_id": child_id,
        "output_dir": str(out.resolve()), "phase": "evaluation", "origin_count": 2})
    alias = _write(root / alias_path, original.read_bytes())
    principal_names = [Path(x).name for x in principal]
    dispatch = _write(root / "logs/verified-frozen-dispatch/job-a.json", {
        "schema_version": 2, "job_id": "job-a", "driver": "graph_v8",
        "status": "accepted-output-pending-scientific-review", "child_returncode": 0,
        "before": {"files": {qr["path"]: qr["sha256"], source_ref["path"]: source_ref["sha256"],
                              pre.relative_to(root).as_posix(): _sha(pre), checkpoint.relative_to(root).as_posix(): _sha(checkpoint),
                              binding_ref["path"]: binding_ref["sha256"], jobref["path"]: jobref["sha256"]}},
        "after": {"files": {qr["path"]: qr["sha256"], source_ref["path"]: source_ref["sha256"],
                             pre.relative_to(root).as_posix(): _sha(pre), checkpoint.relative_to(root).as_posix(): _sha(checkpoint),
                             binding_ref["path"]: binding_ref["sha256"], jobref["path"]: jobref["sha256"]}},
        "validator_before": {"returncode": 0}, "validator_after": {"returncode": 0},
        "child_phase_receipt": {"path": original.relative_to(root).as_posix(), "sha256": _sha(original),
                                 "job_id": child_id, "phase": "evaluation", "status": "completed",
                                 "output_dir": str(out.resolve()),
                                 "principal_artifacts": {name: _sha(out / name) for name in principal_names[:3]},
                                 "receipt_doc": json.loads(original.read_text()),
                                 "phase_receipt_original_path": original.relative_to(root).as_posix(),
                                 "phase_receipt_original_sha256": _sha(original),
                                 "phase_receipt_alias_path": alias.relative_to(root).as_posix(),
                                 "phase_receipt_alias_sha256": _sha(alias)},
        "phase_receipt_original_path": original.relative_to(root).as_posix(),
        "phase_receipt_original_sha256": _sha(original),
        "phase_receipt_alias_path": alias.relative_to(root).as_posix(),
        "phase_receipt_alias_sha256": _sha(alias)})
    report = out / Path(report_rel).name
    output_files = [{"path": p.relative_to(root).as_posix(), "sha256": _sha(p)}
                    for p in sorted(out.iterdir()) if p.is_file()]
    completion = _write(root / "protocols/completion.json", {
        "schema_version": 1, "scope": "global_frozen_test_completion",
        "status": "accepted_by_root", "accepted_by": "root", "verdict": "pass",
        "queue": qr,
        "jobs": [{"job_id": "job-a", "job_spec": jobref,
                  "dispatcher_receipt": {"path": dispatch.relative_to(root).as_posix(), "sha256": _sha(dispatch)},
                  "phase_receipt_original": {"path": original.relative_to(root).as_posix(), "sha256": _sha(original)},
                  "phase_receipt_alias": {"path": alias.relative_to(root).as_posix(), "sha256": _sha(alias)},
                  "driver_report": {"path": report.relative_to(root).as_posix(), "sha256": _sha(report)},
                  "output_files": output_files}]})
    cr = {"path": completion.relative_to(root).as_posix(), "sha256": _sha(completion)}
    return root, cr, qr, job_path


def test_real_dispatch_layout_and_acyclic_queue_closure(tmp_path):
    root, cr, qr, job_path = fixture(tmp_path)
    consumed = verify_completion(root, cr, qr)
    assert "protocols/global-queue.json" in consumed
    assert "runs/test/job-a/predictions.npz" in consumed
    assert any(path.startswith("logs/offline-phase-wall/") for path in consumed)
    assert "runs/pre/pretest-binding.json" in consumed
    assert set(verify_queued_job(root, qr, job_path)) >= {qr["path"], "runs/jobs/job-a.json"}


@pytest.mark.parametrize("mutation", ["missing-job", "oracle", "count", "alias", "output-extra", "dispatch-snapshot", "wrong-reported-contract", "random-child-id", "wrong-absolute-output"])
def test_rejects_incomplete_or_inconsistent_campaign(tmp_path, mutation):
    root, cr, qr, _ = fixture(tmp_path)
    cpath = root / cr["path"]
    c = json.loads(cpath.read_text())
    row = c["jobs"][0]
    if mutation == "missing-job":
        c["jobs"] = []
    elif mutation in {"oracle", "count"}:
        p = root / "runs/test/job-a/frozen-graph-state-diagnostic.json"
        doc = json.loads(p.read_text())
        doc["oracle_accessed" if mutation == "oracle" else "full_test_origin_count"] = True if mutation == "oracle" else 3
        _write(p, doc)
        row["driver_report"]["sha256"] = _sha(p)
        row["output_files"] = [{"path": x["path"], "sha256": _sha(root / x["path"])} for x in row["output_files"]]
    elif mutation == "alias":
        _write(root / row["phase_receipt_alias"]["path"], b"different receipt")
        row["phase_receipt_alias"]["sha256"] = _sha(root / row["phase_receipt_alias"]["path"])
    elif mutation == "output-extra":
        _write(root / "runs/test/job-a/unlisted.txt", "extra")
    elif mutation == "dispatch-snapshot":
        p = root / row["dispatcher_receipt"]["path"]
        doc = json.loads(p.read_text()); doc["after"]["files"] = {}
        _write(p, doc); row["dispatcher_receipt"]["sha256"] = _sha(p)
    elif mutation in {"random-child-id", "wrong-absolute-output"}:
        p = root / row["dispatcher_receipt"]["path"]
        doc = json.loads(p.read_text())
        child = doc["child_phase_receipt"]
        child["job_id"] = "dispatcher-job-id-not-child-id" if mutation == "random-child-id" else child["job_id"]
        if mutation == "wrong-absolute-output":
            child["output_dir"] = str(root / "different/output")
        _write(p, doc); row["dispatcher_receipt"]["sha256"] = _sha(p)
    elif mutation == "wrong-reported-contract":
        qpath = root / qr["path"]
        q = json.loads(qpath.read_text()); q["jobs"][0]["coverage_field"] = "some_other_count"
        _write(qpath, q)
        qr["sha256"] = _sha(qpath)
        c["queue"]["sha256"] = qr["sha256"]
    _write(cpath, c); cr["sha256"] = _sha(cpath)
    with pytest.raises(CompletionError):
        verify_completion(root, cr, qr)


def test_verify_queued_job_rejects_unregistered_path_and_binding_queue_tamper(tmp_path):
    root, _, qr, job_path = fixture(tmp_path)
    with pytest.raises(CompletionError):
        verify_queued_job(root, qr, "runs/jobs/other.json")
    binding = root / "runs/pre/pretest-binding.json"
    doc = json.loads(binding.read_text()); doc["global_test_queue"]["sha256"] = "0" * 64
    _write(binding, doc)
    with pytest.raises(CompletionError):
        verify_queued_job(root, qr, job_path)


def test_rejects_queue_bytes_changed_after_pretest_binding(tmp_path):
    root, cr, qr, _ = fixture(tmp_path)
    qpath = root / qr["path"]
    qpath.write_text(qpath.read_text() + " ", encoding="utf-8")
    with pytest.raises(CompletionError):
        verify_completion(root, cr, qr)


def _two_row_queue(root, mutation=None):
    root, _, qr, _ = fixture(root)
    qpath = root / qr["path"]
    q = json.loads(qpath.read_text())
    first = q["jobs"][0]
    second = json.loads(json.dumps(first))
    second["job_id"] = "job-b"
    second["job_spec_path"] = "runs/jobs/job-b.json"
    second["output_dir"] = "runs/test/job-b"
    second["report_path"] = second["output_dir"] + "/frozen-graph-state-diagnostic.json"
    second["phase_receipt_alias_path"] = "runs/evidence/job-b-phase.json"
    second["principal_files"] = [second["output_dir"] + "/" + Path(x).name for x in first["principal_files"]]
    # Sharing a pretest binding is intentionally legal for a multi-side plan.
    if mutation == "duplicate-job-spec":
        second["job_spec_path"] = first["job_spec_path"]
    elif mutation == "duplicate-output":
        second["output_dir"] = first["output_dir"]
        second["report_path"] = second["output_dir"] + "/frozen-graph-state-diagnostic.json"
        second["principal_files"] = [second["output_dir"] + "/" + Path(x).name for x in first["principal_files"]]
    elif mutation == "ancestor-output":
        second["output_dir"] = "runs/test"
        second["report_path"] = "runs/test/frozen-graph-state-diagnostic.json"
        second["principal_files"] = ["runs/test/" + Path(x).name for x in first["principal_files"]]
    elif mutation == "duplicate-report":
        second["report_path"] = first["report_path"]
    elif mutation == "duplicate-principal":
        second["principal_files"][1] = first["principal_files"][1]
    elif mutation == "duplicate-alias":
        second["phase_receipt_alias_path"] = first["phase_receipt_alias_path"]
    elif mutation == "alias-inside-output":
        second["phase_receipt_alias_path"] = second["output_dir"] + "/evidence/phase.json"
    q["jobs"] = [first, second]
    _write(qpath, q)
    return root, qpath


def test_global_queue_paths_are_job_isolated_but_binding_sharing_is_allowed(tmp_path):
    root, qpath = _two_row_queue(tmp_path)
    _, rows = completion._load_queue(root, qpath)
    assert set(rows) == {"job-a", "job-b"}
    assert rows["job-a"]["pretest_execution_binding_paths"] == rows["job-b"]["pretest_execution_binding_paths"]


@pytest.mark.parametrize("mutation", ["duplicate-job-spec", "duplicate-output", "ancestor-output",
                                        "duplicate-report", "duplicate-principal", "duplicate-alias",
                                        "alias-inside-output"])
def test_global_queue_rejects_cross_job_path_collisions(tmp_path, mutation):
    root, qpath = _two_row_queue(tmp_path, mutation)
    with pytest.raises(CompletionError):
        completion._load_queue(root, qpath)
