"""Metadata-only temporary-RUN fixtures for ordinary frozen-test admission."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import ordinary_frozen_test_admission_v1 as admission
import bind_training_inputs as binder
import global_frozen_test_completion_v1 as completion


RUN = Path(__file__).resolve().parents[2]


def put(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    elif isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")
    return path


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(root: Path, path: Path):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}


def copied_source(root: Path, rel: str):
    return put(root / rel, (RUN / rel).read_bytes())


def make_state(root: Path, matrix_ref):
    # Copy the actual validator and all byte dependencies it imports/checks.
    for rel in (admission.PIPELINE_VALIDATOR_REL, admission.PIPELINE_DEFINITION_REL,
                *admission.PIPELINE_SIBLINGS):
        copied_source(root, rel)
    contract = put(root / "protocols/test-contract.txt", b"contract fixture")
    contract_ref = ref(root, contract)
    definition = json.loads((root / admission.PIPELINE_DEFINITION_REL).read_text())
    ids = definition["stages"]
    passed = ids[:ids.index("P3A") + 1]
    stages = []
    history = []
    for sid in ids:
        status = "passed" if sid in passed else "pending"
        if sid == "P4":
            status = "running"
        inputs = [contract_ref] if status == "passed" else []
        evidence = [matrix_ref] if sid == "P3A" else ([contract_ref] if status == "passed" else [])
        stages.append({"id": sid, "status": status, "contract_version": "v1",
                       "gate_verdict": "pass" if status == "passed" else "pending",
                       "inputs": inputs, "evidence": evidence})
        if status == "passed":
            history.append({"stage": sid, "action": "pass"})
    state = {"schema_version": 1, "pipeline": "f-research-v1", "contract_version": "v1",
             "contract": [contract_ref], "history": history, "stages": stages}
    return put(root / "pipeline-state.json", state)


def complete_fixture(tmp_path, *, copy_route=False):
    root = tmp_path
    # Runtime/source tree for the unchanged binder; toy files are opaque bytes.
    package = root / "edsgn/src/edsgn"
    for path in (RUN / "edsgn/src/edsgn").glob("*.py"):
        put(package / path.name, path.read_bytes())
    binder_copy = copied_source(root, "edsgn/scripts/bind_training_inputs.py")
    for rel in (admission.SOURCE_REL, admission.GLOBAL_COMPLETION_REL,
                admission.DISPATCHER_REL, admission.DRIVER_SOURCES["graph_v8"]):
        if (RUN / rel).is_file():
            copied_source(root, rel)
        else:
            put(root / rel, b"pinned dispatcher fixture")

    dataset = root / "datasets/email"
    put(dataset / "metadata.json", {"split": "metadata-only-fixture"})
    put(dataset / "opaque.npz", b"opaque NPZ bytes; never decoded")
    out = root / "runs/train/email/STID/seed0"
    config = put(root / "configs/stid.json", {
        "variant": "STID", "seed": 0, "dataset_path": str(dataset),
        "output_dir": str(out), "training": {"epochs": 1},
    })
    old_root, old_file = binder.ROOT, binder.__file__
    binder.ROOT, binder.__file__ = root, str(binder_copy)
    try:
        binding_path = binder.prepare_binding(config_path=config)
        blueprint_path = root / "protocols/input-blueprint.json"
        blueprint_ref = admission.write_static_input_blueprint(root, ref(root, config),
                                                                "protocols/input-blueprint.json",
                                                                "email", "STID")
        preparation = {"mode": "static_config", "config": ref(root, config),
                       "input_blueprint": blueprint_ref,
                       "expected_training_binding_path": binding_path.relative_to(root).as_posix()}

        driver = admission.DRIVER_SOURCES["graph_v8"]
        global_source = admission.GLOBAL_COMPLETION_REL
        dispatch_source = admission.DISPATCHER_REL
        checkpoint = put(out / ("checkpoint.pt" if copy_route else "best_model.pt"), b"opaque checkpoint")
        if copy_route:
            execution_checkpoint = root / "runs/frozen-replay-checkpoints-v1/email/stid-seed0/model.pt"
            execution_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            execution_checkpoint.write_bytes(checkpoint.read_bytes())
        else:
            execution_checkpoint = checkpoint
        freeze_dir = execution_checkpoint.parent
        matrix = {"schema_version": 1, "scope": "prospective_offline_test_job_admission_matrix",
                  "status": "accepted_before_training_test",
                  "source_refs": [ref(root, root / rel) for rel in
                                  (admission.SOURCE_REL, global_source, dispatch_source)],
                  "jobs": [{"job_id": "email-STID-seed0", "driver": "graph_v8", "method_id": "STID",
                            "seed_id": 0, "training_variant": "STID", "training_preparation": preparation,
                            "expected_driver_arguments": ["--checkpoint", str(execution_checkpoint),
                                "--dataset-path", str(dataset), "--output", "runs/test/email/STID/seed0",
                                "--policy", "full", "--device", "cpu"],
                            "output_dir": "runs/test/email/STID/seed0",
                            "phase_receipt_alias_path": "runs/evidence/email-STID-test.json"}]}
        if copy_route:
            matrix["jobs"][0]["checkpoint_route"] = {
                "mode": "byte_identical_copy", "source_path": str(checkpoint),
                "execution_path": str(execution_checkpoint)}
        matrix_path = put(root / "protocols/ordinary-matrix.json", matrix)
        matrix_ref = ref(root, matrix_path)
        make_state(root, matrix_ref)

        full_policy = {"kind": "full", "graph_fraction": 1.0, "state_fraction": 1.0,
            "max_graph_age": 4, "max_state_age": 4, "graph_threshold": 0.0,
            "state_threshold": 0.0, "audit_interval": 8, "seed": 0,
            "force_initialization": True, "cost_table": {}, "runtime_budget_ms": None}
        policy_digest = hashlib.sha256(json.dumps(full_policy, sort_keys=True, separators=(",", ":"),
                                                    ensure_ascii=False).encode()).hexdigest()
        freeze = put(freeze_dir / "test-freeze.json", {"schema_version": 1, "status": "frozen",
            "checkpoint_sha256": sha(execution_checkpoint),
            "policies": {"full": full_policy}, "allowed_policy_digests": {"full": policy_digest}})
        summary = put(out / "training_summary.json", {"scope": "full_protocol", "test_accessed": False,
                                                        "checkpoint_sha256": sha(checkpoint)})
        phase = put(root / "logs/offline-phase-wall/train-email-STID.json", {
            "scope": "offline-experiment-phase", "phase": "training", "status": "completed",
            "returncode": 0, "output_dir": str(out), "binding_path": str(binding_path),
            "inputs": [ref(root, config)], "outputs": [ref(root, summary), ref(root, checkpoint)]})
        numeric = put(root / "reviews/numeric-validation.json", {
            "scope": "full_bound_validation_numeric_audit", "status": "passed",
            "split": "validation", "test_arrays_or_test_windows_read": False,
            "seed": 0, "variant": "STID", "dataset_path": str(dataset),
            "checkpoint_sha256": sha(checkpoint), "n_validation_origins": 2,
            "origins_complete_and_ordered": True,
            "saved_targets_and_masks_match_validation_source": True})
        acceptance = put(root / "reviews/root-acceptance.json", {
            "accepted_by": "root", "verdict": "pass", "scope": "completed-development-seed-validation",
            "pipeline_advanced": False, "scientific_hypothesis_accepted": False,
            "formal_five_seed_result": False, "final_test_result": False,
            "dataset": "email", "variant": "STID", "model_seed": 0,
            "evidence_sha256": {p.relative_to(root).as_posix(): sha(p) for p in
                (config, binding_path, phase, summary, checkpoint, numeric)}})
        job_path = root / "runs/jobs/email-STID-seed0.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path = root / "protocols/global-test-queue.json"
        independent_paths = [config, blueprint_path, binding_path, phase, summary, checkpoint,
                             execution_checkpoint, freeze, acceptance, numeric, *[
                                 root / x for x in (admission.SOURCE_REL, admission.GLOBAL_COMPLETION_REL,
                                                    admission.DISPATCHER_REL, driver)]]
        # Also bind every file source inventoried by the actual launch binding.
        launch = json.loads(binding_path.read_text())
        independent_paths += [Path(x["path"]) for x in launch["dataset_sources"] + launch["code_sources"]]
        independent_paths = list({p.resolve() for p in independent_paths})
        job_sources = [ref(root, root / rel) for rel in
                       (admission.SOURCE_REL, global_source, dispatch_source, driver)]
        pre_refs = {x["path"]: x["sha256"] for x in
                    [*(ref(root, p) for p in independent_paths), *job_sources, matrix_ref]}
        queue_row = {"job_id": "email-STID-seed0", "job_spec_path": job_path.relative_to(root).as_posix(),
            "pretest_execution_binding_paths": ["runs/evidence/email-STID-admission.json"], "driver": "graph_v8",
            "output_dir": "runs/test/email/STID/seed0", "expected_origin_count": 2,
            "coverage_field": "full_test_origin_count", "expected_report_scope": "frozen_test_replay_oracle_free",
            "expected_phase": "evaluation", "report_path": "runs/test/email/STID/seed0/frozen-graph-state-diagnostic.json",
            "phase_receipt_alias_path": "runs/evidence/email-STID-test.json",
            "principal_files": ["runs/test/email/STID/seed0/" + name for name in
                ("frozen-graph-state-diagnostic.json", "predictions.npz", "counter-rows.json",
                 "trace-rows.jsonl.gz", "metrics_by_horizon.csv")],
            "pretest_refs": [{"path": p, "sha256": h} for p, h in sorted(pre_refs.items())]}
        queue = put(queue_path, {"schema_version": 1, "scope": "prospective_complete_test_job_set",
            "status": "frozen_before_test", "queue_id": "toy-queue-2026", "jobs": [queue_row]})
        queue_ref = ref(root, queue)
        input_refs = [ref(root, p) for p in independent_paths]
        admission_doc = {"schema_version": 1, "scope": "ordinary_pretest_job_admission",
            "status": "accepted_by_root", "test_feedback_used": False, "matrix": matrix_ref,
            "matrix_job_id": "email-STID-seed0", "global_test_queue": queue_ref,
            "config": ref(root, config), "training_binding": ref(root, binding_path),
            "training_phase_receipt": ref(root, phase), "training_summary": ref(root, summary),
            "checkpoint": ref(root, execution_checkpoint), "freeze": ref(root, freeze), "policy_config": None,
            "candidate_selection": None, "root_numeric_acceptance": ref(root, acceptance), "input_refs": input_refs}
        if copy_route:
            admission_doc["training_checkpoint"] = ref(root, checkpoint)
        admission_path = put(root / "runs/evidence/email-STID-admission.json", admission_doc)
        admission_ref = ref(root, admission_path)
        job = {"schema_version": 2, "job_id": "email-STID-seed0", "driver": "graph_v8",
               "argv": matrix["jobs"][0]["expected_driver_arguments"], "sources": job_sources,
               "pretest_execution_bindings": [admission_ref],
               "phase_receipt_alias": "runs/evidence/email-STID-test.json"}
        put(job_path, job)
        return root, admission_ref, job, job_path, matrix, binder_copy
    finally:
        binder.ROOT, binder.__file__ = old_root, old_file


def test_static_blueprint_writes_metadata_only_and_is_bound_to_method_id(tmp_path, monkeypatch):
    root, _, _, _, matrix, binder_copy = complete_fixture(tmp_path)
    row = matrix["jobs"][0]
    consumed = {}
    config_ref = row["training_preparation"]["config"]
    blueprint_ref = row["training_preparation"]["input_blueprint"]
    old_root, old_file = binder.ROOT, binder.__file__
    binder.ROOT, binder.__file__ = root, str(binder_copy)
    try:
        config, expected = admission._validate_static_preparation(root, row, config_ref, blueprint_ref,
            row["training_preparation"]["expected_training_binding_path"], consumed)
        bad = dict(row, method_id="different-method")
        with pytest.raises(admission.AdmissionError, match="comparison_id"):
            admission._validate_static_preparation(root, bad, config_ref, blueprint_ref,
                row["training_preparation"]["expected_training_binding_path"], {})
    finally:
        binder.ROOT, binder.__file__ = old_root, old_file
    assert config["variant"] == "STID" and expected.name == "input-binding-at-launch.json"


def test_official_gate_and_full_admission_use_actual_canonical_temp_ledger(tmp_path):
    root, admission_ref, job, job_path, _, binder_copy = complete_fixture(tmp_path)
    old_root, old_file = binder.ROOT, binder.__file__
    binder.ROOT, binder.__file__ = root, str(binder_copy)
    try:
        consumed = admission.validate_admission(root, admission_ref, job, job_path)
    finally:
        binder.ROOT, binder.__file__ = old_root, old_file
    assert "pipeline-state.json" in consumed
    assert "protocols/global-test-queue.json" in consumed
    assert "runs/train/email/STID/seed0/best_model.pt" in consumed
    assert "runs/jobs/email-STID-seed0.json" in consumed
    assert "datasets/email/opaque.npz" in consumed  # hashed as bytes, never decoded
    binding_doc = json.loads((root / admission_ref["path"]).read_text())
    queued = completion.verify_queued_job(root, binding_doc["global_test_queue"], job_path.relative_to(root).as_posix())
    assert binding_doc["global_test_queue"]["path"] in queued


def test_gate_rejects_p3a_without_the_exact_matrix_hash(tmp_path):
    root, _, _, _, _, _ = complete_fixture(tmp_path)
    state_path = root / "pipeline-state.json"
    state = json.loads(state_path.read_text())
    p3a = next(row for row in state["stages"] if row["id"] == "P3A")
    p3a["evidence"] = [{"path": "protocols/test-contract.txt",
                        "sha256": sha(root / "protocols/test-contract.txt")}]
    put(state_path, state)
    actual_admission = json.loads((root / "runs/evidence/email-STID-admission.json").read_text())
    with pytest.raises(admission.AdmissionError, match="exact admission matrix"):
        admission._official_gate(root, actual_admission["matrix"], {})


def test_frozen_policy_identity_is_checked_without_loading_checkpoint(tmp_path):
    root, _, _, _, _, _ = complete_fixture(tmp_path)
    freeze = root / "runs/train/email/STID/seed0/test-freeze.json"
    admission._validate_frozen_policy(root, "graph_v8", {"--policy": "full"}, freeze, None)
    with pytest.raises(admission.AdmissionError, match="not uniquely authorized"):
        admission._validate_frozen_policy(root, "graph_v8", {"--policy": "periodic"}, freeze, None)


def test_byte_identical_checkpoint_copy_has_distinct_parent_and_adjacent_freeze(tmp_path):
    root, admission_ref, job, job_path, _, binder_copy = complete_fixture(tmp_path, copy_route=True)
    old_root, old_file = binder.ROOT, binder.__file__
    binder.ROOT, binder.__file__ = root, str(binder_copy)
    try:
        consumed = admission.validate_admission(root, admission_ref, job, job_path)
    finally:
        binder.ROOT, binder.__file__ = old_root, old_file
    assert "runs/train/email/STID/seed0/checkpoint.pt" in consumed
    assert "runs/frozen-replay-checkpoints-v1/email/stid-seed0/model.pt" in consumed
    assert "runs/frozen-replay-checkpoints-v1/email/stid-seed0/test-freeze.json" in consumed


def test_checkpoint_copy_rejects_mutated_bytes_and_unregistered_route(tmp_path):
    root, _, _, _, matrix, _ = complete_fixture(tmp_path, copy_route=True)
    row = matrix["jobs"][0]
    cfg = json.loads((root / "configs/stid.json").read_text())
    source = root / "runs/train/email/STID/seed0/checkpoint.pt"
    execution = root / "runs/frozen-replay-checkpoints-v1/email/stid-seed0/model.pt"
    source_ref, execution_ref = ref(root, source), ref(root, execution)
    admission._validate_checkpoint_route(root, row["checkpoint_route"], cfg, execution_ref, source_ref)
    execution.write_bytes(b"altered copy")
    with pytest.raises(admission.AdmissionError, match="not byte-identical"):
        admission._validate_checkpoint_route(root, row["checkpoint_route"], cfg,
                                             ref(root, execution), source_ref)
    execution.write_bytes(source.read_bytes())
    execution_ref = ref(root, execution)
    bad_route = dict(row["checkpoint_route"], execution_path="runs/test/unregistered.pt")
    with pytest.raises(admission.AdmissionError, match="prospective checkpoint_route path"):
        admission._validate_checkpoint_route(root, bad_route, cfg, execution_ref, source_ref)


def test_checkpoint_copy_missing_execution_file_is_admission_error(tmp_path):
    root, _, _, _, matrix, _ = complete_fixture(tmp_path, copy_route=True)
    row = matrix["jobs"][0]
    cfg = json.loads((root / "configs/stid.json").read_text())
    source = root / "runs/train/email/STID/seed0/checkpoint.pt"
    source_ref = ref(root, source)
    missing = root / "runs/frozen-replay-checkpoints-v1/email/stid-seed0/missing.pt"
    missing_ref = {"path": missing.relative_to(root).as_posix(), "sha256": "0" * 64}
    with pytest.raises(admission.AdmissionError, match="execution checkpoint copy path is unavailable"):
        admission._validate_checkpoint_route(root, row["checkpoint_route"], cfg, missing_ref, source_ref)


def test_declared_copy_route_requires_training_checkpoint_ref(tmp_path):
    root, admission_ref, job, job_path, _, binder_copy = complete_fixture(tmp_path, copy_route=True)
    adm_path = root / admission_ref["path"]
    doc = json.loads(adm_path.read_text()); del doc["training_checkpoint"]
    put(adm_path, doc)
    admission_ref = ref(root, adm_path)
    job["pretest_execution_bindings"] = [admission_ref]
    put(job_path, job)
    old_root, old_file = binder.ROOT, binder.__file__
    binder.ROOT, binder.__file__ = root, str(binder_copy)
    try:
        with pytest.raises(admission.AdmissionError, match="training_checkpoint ref must be present"):
            admission.validate_admission(root, admission_ref, job, job_path)
    finally:
        binder.ROOT, binder.__file__ = old_root, old_file


def test_completed_prior_matrix_row_may_have_outputs_while_current_row_is_absent(tmp_path):
    root, _, _, _, matrix, _ = complete_fixture(tmp_path)
    prior = json.loads(json.dumps(matrix["jobs"][0]))
    prior.update({"job_id": "email-GRU-seed0", "method_id": "GRU",
                  "output_dir": "runs/test/email/GRU/seed0",
                  "phase_receipt_alias_path": "runs/evidence/email-GRU-test.json"})
    prior["expected_driver_arguments"] = [
        "--checkpoint", "runs/train/email/GRU/seed0/model.pt", "--dataset-path",
        str(root / "datasets/email"), "--output", prior["output_dir"], "--policy", "full", "--device", "cpu"]
    matrix["jobs"].append(prior)
    matrix_path = put(root / "protocols/ordinary-matrix.json", matrix)
    put(root / "runs/test/email/STID/seed0/old-result.json", "completed prior row")
    put(root / "runs/evidence/email-STID-test.json", "completed prior receipt alias")
    _, parsed = admission._matrix_doc(root, ref(root, matrix_path), {})
    assert len(parsed["jobs"]) == 2


@pytest.mark.parametrize("target", ["output", "alias"])
def test_current_job_rejects_preexisting_output_or_alias(tmp_path, target):
    root, admission_ref, job, job_path, matrix, binder_copy = complete_fixture(tmp_path)
    if target == "output":
        put(root / matrix["jobs"][0]["output_dir"] / "stale.json", "already exists")
    else:
        put(root / matrix["jobs"][0]["phase_receipt_alias_path"], {"status": "stale"})
    old_root, old_file = binder.ROOT, binder.__file__
    binder.ROOT, binder.__file__ = root, str(binder_copy)
    try:
        with pytest.raises(admission.AdmissionError, match="current job output and phase receipt alias"):
            admission.validate_admission(root, admission_ref, job, job_path)
    finally:
        binder.ROOT, binder.__file__ = old_root, old_file


@pytest.mark.parametrize("mutation,match", [
    ("dataset", "actual job argv"),
    ("seed", "missing or hash mismatch"),
    ("stale_job_source", "job source"),
])
def test_admission_fails_closed_on_identity_or_source_mutations(tmp_path, mutation, match):
    root, admission_ref, job, job_path, matrix, binder_copy = complete_fixture(tmp_path)
    if mutation == "dataset":
        job["argv"][3] = "datasets/mooc"
        put(job_path, job)
    elif mutation == "seed":
        cfg = root / "configs/stid.json"
        doc = json.loads(cfg.read_text()); doc["seed"] = 1; put(cfg, doc)
    elif mutation == "stale_job_source":
        job["sources"][0]["sha256"] = "0" * 64
        put(job_path, job)
    old_root, old_file = binder.ROOT, binder.__file__
    binder.ROOT, binder.__file__ = root, str(binder_copy)
    try:
        with pytest.raises(admission.AdmissionError, match=match):
            admission.validate_admission(root, admission_ref, job, job_path)
    finally:
        binder.ROOT, binder.__file__ = old_root, old_file


def test_resolver_rejects_symlink_even_when_target_stays_inside_root(tmp_path):
    real = put(tmp_path / "real.txt", "x")
    link = tmp_path / "inside-link.txt"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable on this host")
    with pytest.raises(admission.AdmissionError, match="symlink"):
        admission._resolve(tmp_path, "inside-link.txt")
