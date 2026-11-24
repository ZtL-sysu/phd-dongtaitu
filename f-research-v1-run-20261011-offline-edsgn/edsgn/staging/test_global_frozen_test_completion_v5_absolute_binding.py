"""CPU-only compatibility checks for unchanged absolute-path launch bindings."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import global_frozen_test_completion_v5 as completion


def write(root, rel, value):
    p = root / rel; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
    return p


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_job(root, mutate=None):
    data = root / "data/collection"
    write(root, "data/collection/train.npz", b"opaque bytes")
    write(root, "data/collection/manifest.json", {"metadata": "only"})
    write(root, "edsgn/src/edsgn/placeholder.py", b"# source")
    config = write(root, "configs/v4.json", {"variant": "V4", "seed": 0,
        "dataset_path": str(data), "output_dir": str(root / "runs/train/v4")})
    binder_path = write(root, "edsgn/scripts/bind_training_inputs.py",
        (Path(__file__).resolve().parents[1] / "scripts/bind_training_inputs.py").read_bytes())
    spec = importlib.util.spec_from_file_location("binding_fixture", binder_path)
    binder = importlib.util.module_from_spec(spec); spec.loader.exec_module(binder)
    binder.ROOT, binder.__file__ = root, str(binder_path)
    binding_path = binder.prepare_binding(config_path=config)
    binding = json.loads(binding_path.read_text())
    if mutate:
        mutate(binding)
    job_source = write(root, "edsgn/staging/case_source.py", b"source")
    output, report, alias, jobrel, bindrel = (
        "runs/test/case", "runs/test/case/case-signal-sensitivity-report.json",
        "runs/evidence/case.json", "runs/jobs/case.json", "runs/pre/binding.json")
    refs = {}
    for key in ("dataset_sources", "code_sources"):
        for item in binding[key]:
            p = Path(item["path"]).resolve()
            if not p.is_relative_to(root.resolve()):
                continue
            refs[p.relative_to(root).as_posix()] = item["sha256"]
    refs[job_source.relative_to(root).as_posix()] = digest(job_source)
    # Queue is frozen before admission binding/job; no cyclic references.
    queue = write(root, "protocols/queue.json", {"schema_version": 1,
        "scope": "prospective_complete_test_job_set", "status": "frozen_before_test",
        "queue_id": "absolute-binding", "jobs": [{"job_id": "case-job", "job_spec_path": jobrel,
        "pretest_execution_binding_paths": [bindrel], "driver": "case_v3", "output_dir": output,
        "expected_origin_count": 2, "coverage_field": "origin_count",
        "expected_report_scope": "frozen_case_signal_sensitivity_diagnostic", "expected_phase": "evaluation",
        "report_path": report, "phase_receipt_alias_path": alias,
        "principal_files": ["runs/test/case/" + n for n in completion.DRIVER_CONTRACTS["case_v3"]["principal"]],
        "pretest_refs": [{"path": p, "sha256": h} for p, h in sorted(refs.items())]}]})
    qref = {"path": queue.relative_to(root).as_posix(), "sha256": digest(queue)}
    binding["global_test_queue"] = qref
    binding_file = write(root, bindrel, binding)
    bref = {"path": bindrel, "sha256": digest(binding_file)}
    job = write(root, jobrel, {"schema_version": 2, "job_id": "case-job", "driver": "case_v3",
        "argv": ["--split", "test", "--output", output],
        "sources": [{"path": job_source.relative_to(root).as_posix(), "sha256": digest(job_source)}],
        "pretest_execution_bindings": [bref], "phase_receipt_alias": alias})
    return root, queue, qref, job, binder, binding_file


def test_actual_binding_absolute_refs_are_canonicalized_for_queue(tmp_path):
    root, queue, qref, job, binder, binding = make_job(tmp_path)
    # Confirm the fixture uses the unmodified binder's actual absolute-path schema.
    doc = json.loads(binding.read_text())
    assert Path(doc["dataset_sources"][0]["path"]).is_absolute()
    result = completion.verify_queued_job(root, qref, "runs/jobs/case.json")
    assert "data/collection/train.npz" in result
    assert "edsgn/src/edsgn/placeholder.py" in result


@pytest.mark.parametrize("badpath", ["C:/outside/secret.json", "../outside.json"])
def test_absolute_or_traversal_source_outside_run_is_rejected(tmp_path, badpath):
    def tamper(binding):
        binding["dataset_sources"][0]["path"] = badpath
    root, _, qref, job, _, _ = make_job(tmp_path, tamper)
    with pytest.raises(completion.CompletionError): completion.verify_queued_job(root, qref, "runs/jobs/case.json")


def test_duplicate_canonical_path_within_one_binding_list_rejected(tmp_path):
    def tamper(binding):
        binding["dataset_sources"].append(dict(binding["dataset_sources"][0]))
    root, _, qref, job, _, _ = make_job(tmp_path, tamper)
    with pytest.raises(completion.CompletionError, match="duplicate canonical path"):
        completion.verify_queued_job(root, qref, "runs/jobs/case.json")


def test_same_canonical_path_across_distinct_lists_with_same_hash_allowed(tmp_path):
    def tamper(binding):
        binding["code_sources"].append(dict(binding["dataset_sources"][0]))
    root, _, qref, job, _, _ = make_job(tmp_path, tamper)
    result = completion.verify_queued_job(root, qref, "runs/jobs/case.json")
    assert "data/collection/train.npz" in result


def test_conflicting_hash_for_same_canonical_path_across_lists_rejected(tmp_path):
    def tamper(binding):
        row = dict(binding["dataset_sources"][0]); row["sha256"] = "0" * 64
        binding["code_sources"].append(row)
    root, _, qref, job, _, _ = make_job(tmp_path, tamper)
    with pytest.raises(completion.CompletionError): completion.verify_queued_job(root, qref, "runs/jobs/case.json")
