"""End-to-end metadata fixture for case admission using real validators/binder."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

RUN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN / "edsgn/staging"))
import case_frozen_test_admission_v2 as case
import ordinary_frozen_test_admission_v1 as ordinary


def put(root, rel, value):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        p.write_bytes(value)
    else:
        p.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return p


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(root, p):
    return {"path": Path(p).resolve().relative_to(root.resolve()).as_posix(), "sha256": sha(p)}


def copied(root, rel):
    return put(root, rel, (RUN / rel).read_bytes())


def all_refs(obj, excluded=()):
    found = {}
    def walk(x):
        if isinstance(x, dict):
            if set(x) == {"path", "sha256"} and isinstance(x["path"], str):
                if x["path"] not in excluded:
                    found[x["path"]] = x["sha256"]
            else:
                for y in x.values(): walk(y)
        elif isinstance(x, list):
            for y in x: walk(y)
    walk(obj)
    return found


def fixture(root):
    # Copy the runtime and pinned author sources; all archives/checkpoints below
    # are opaque bytes and are only hashed by the unchanged launch binder.
    for p in (RUN / "edsgn/src/edsgn").glob("*.py"):
        put(root, "edsgn/src/edsgn/" + p.name, p.read_bytes())
    for base in (RUN / "sources/SeqSNN",):
        if base.exists():
            for p in base.rglob("*.py"):
                put(root, p.relative_to(RUN).as_posix(), p.read_bytes())
    required_sources = {
        case.SOURCE_REL, case.DRIVER_ADMISSION_REL, case.GLOBAL_REL, case.DRIVER_GLOBAL_REL, case.ORDINARY_REL,
        case.DRIVER_REL, case.C06_REL, case.KERNEL_REL, case.H2_KERNEL_REL,
        case.KERNEL_ACCEPTANCE_REL, "edsgn/staging/simple_baseline_frozen_evaluation_v2.py",
        "edsgn/scripts/aggregate_budget.py", "edsgn/scripts/owned_python_job_inventory_v1.py",
        "reviews/count-loss-training-controller-v5-root-engineering-v1.json",
        "edsgn/scripts/frozen_synthetic_stress.py", "edsgn/src/edsgn/prepare.py",
        "edsgn/scripts/bind_training_inputs.py",
    }
    for rel in required_sources:
        copied(root, rel)
    # Actual canonical validator and its byte dependencies are exercised.
    for rel in (ordinary.PIPELINE_VALIDATOR_REL, ordinary.PIPELINE_DEFINITION_REL,
                *ordinary.PIPELINE_SIBLINGS): copied(root, rel)

    data = "data/processed/synthetic-n64-identity-change0.3-miss0"
    put(root, f"{data}/train_seed0.npz", b"opaque train archive")
    put(root, f"{data}/validation_seed0.npz", b"opaque validation archive")
    test_archives = [put(root, f"{data}/test_seed{seed}.npz", f"opaque test archive {seed}".encode())
                     for seed in (200, 201, 202, 203)]
    test_archive = test_archives[0]
    put(root, f"{data}/metadata.json", {"fixture": "metadata-only"})

    out = "runs/train/synthetic/SeqSNN/V4/seed0"
    config = put(root, "configs/v4-synthetic-seed0.json", {
        "variant": "V4", "seed": 0, "dataset_path": str(root / data),
        "output_dir": str(root / out), "training": {"epochs": 1},
    })
    binder_path = root / "edsgn/scripts/bind_training_inputs.py"
    spec = importlib.util.spec_from_file_location("fixture_binder", binder_path)
    binder = importlib.util.module_from_spec(spec); spec.loader.exec_module(binder)
    binder.ROOT, binder.__file__ = root, str(binder_path)
    training_binding = binder.prepare_binding(config_path=config)
    blueprint_ref = ordinary.write_static_input_blueprint(root, ref(root, config),
        "protocols/v4-input-blueprint.json", "synthetic", "case_v3")
    prep = {"mode": "static_config", "config": ref(root, config),
            "input_blueprint": blueprint_ref,
            "expected_training_binding_path": training_binding.relative_to(root).as_posix()}

    checkpoint = put(root, f"{out}/checkpoint.pt", b"opaque checkpoint bytes")
    policy = {"kind": "full", "graph_fraction": 1.0, "state_fraction": 1.0,
              "max_graph_age": 4, "max_state_age": 4, "graph_threshold": 0.0,
              "state_threshold": 0.0, "audit_interval": 8, "seed": 0,
              "force_initialization": True, "cost_table": {}, "runtime_budget_ms": None}
    pdigest = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    freeze = put(root, f"{out}/test-freeze.json", {"schema_version": 1, "status": "frozen",
        "checkpoint_sha256": sha(checkpoint), "policies": {"full": policy},
        "allowed_policy_digests": {"full": pdigest}})
    summary = put(root, f"{out}/training_summary.json", {"scope": "full_protocol", "test_accessed": False,
        "checkpoint_sha256": sha(checkpoint), "source_manifest_sha256": "a" * 64})
    phase = put(root, "logs/offline-phase-wall/train-synthetic-V4-seed0.json", {
        "scope": "offline-experiment-phase", "phase": "training", "status": "completed", "returncode": 0,
        "output_dir": str(root / out), "binding_path": str(training_binding),
        "inputs": [ref(root, config)], "outputs": [ref(root, summary), ref(root, checkpoint)]})
    numeric = put(root, "reviews/numeric-v4.json", {"scope": "full_bound_validation_numeric_audit",
        "status": "passed", "split": "validation", "test_arrays_or_test_windows_read": False,
        "seed": 0, "variant": "V4", "dataset_path": str(root / data),
        "checkpoint_sha256": sha(checkpoint), "n_validation_origins": 1,
        "origins_complete_and_ordered": True, "saved_targets_and_masks_match_validation_source": True})
    acceptance = put(root, "reviews/root-numeric-v4.json", {
        "accepted_by": "root", "verdict": "pass", "scope": "completed-development-seed-validation",
        "pipeline_advanced": False, "scientific_hypothesis_accepted": False,
        "formal_five_seed_result": False, "final_test_result": False,
        "dataset": "synthetic", "variant": "V4", "model_seed": 0,
        "evidence_sha256": {p.relative_to(root).as_posix(): sha(p) for p in
            (config, training_binding, phase, summary, checkpoint, numeric)}})

    # Descriptor uses true C06 v1 domain schema, with one toy origin interval.
    domain = put(root, "protocols/case/domain.json", {"schema": "edsgn.split-history-domain.v1",
        "dataset": {"source_name": "identity", "event_kind": "synthetic", "partition": "test",
                    "shape_tnf": [40, 64, 1], "history_length": 28, "hmax": 7},
        "trajectories": [{"trajectory_index": i, "source_trajectory_seed": 200 + i, "split": "test",
                          "origin_first": 27, "origin_last": 32, "origin_count": 6,
                          "history_length": 28} for i in range(4)]})
    c06_sources = [case.C06_REL, "edsgn/staging/split_history_robustness_v2.py",
        "edsgn/staging/frozen_split_history_robustness_v1.py", "edsgn/staging/frozen_split_history_robustness_v5.py",
        "edsgn/staging/blind_reference_cost_policy_v1.py", "reviews/blind-reference-cost-policy-root-acceptance-v1.json",
        "edsgn/src/edsgn/data.py", "edsgn/src/edsgn/engine.py", "edsgn/src/edsgn/robustness.py",
        "edsgn/src/edsgn/prepare.py"]
    for rel in c06_sources:
        if not (root / rel).is_file(): copied(root, rel)
    descriptor = put(root, "protocols/case/descriptor.json", {"schema": "edsgn.c06-domain-preflight.v1",
        "split": "test", "ordered_collection": [{"trajectory_index": i,
            "path": p.relative_to(root).as_posix(), "archive_sha256": sha(p),
            "shape_tnf": [40, 64, 1]} for i, p in enumerate(test_archives)],
        "domain_manifest_path": domain.relative_to(root).as_posix(),
        "domain_manifest_sha256": sha(domain), "source_sha256": {rel: sha(root / rel) for rel in c06_sources}})
    # Freeze source digest exactly as the case driver does, using copied runtime bytes.
    sys.path.insert(0, str(root / "edsgn/src"))
    from edsgn import engine
    source_map = {f"edsgn/src/edsgn/{name}": digest for name, digest in engine.source_manifest().items()}
    for rel in (case.DRIVER_ADMISSION_REL, case.DRIVER_GLOBAL_REL, case.DRIVER_REL, case.KERNEL_REL, case.H2_KERNEL_REL,
                case.KERNEL_ACCEPTANCE_REL, "edsgn/staging/simple_baseline_frozen_evaluation_v2.py",
                "edsgn/scripts/aggregate_budget.py", "edsgn/scripts/owned_python_job_inventory_v1.py",
                "reviews/count-loss-training-controller-v5-root-engineering-v1.json",
                "edsgn/scripts/frozen_synthetic_stress.py", "edsgn/src/edsgn/prepare.py"):
        source_map[rel] = sha(root / rel)
    source_digest = hashlib.sha256(json.dumps(source_map, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False).encode()).hexdigest()
    origin = put(root, "reviews/case-validation-origins.json", {"split": "validation",
        "origins_complete_and_ordered": True, "full_validation_origin_count": 1,
        "checkpoint_sha256": sha(checkpoint), "dataset_fingerprint": engine.dataset_fingerprint(str(root / data)),
        "source_manifest_sha256": source_digest, "node_order_sha256": "c" * 64})
    selection = put(root, "protocols/case/selection.json", {"schema_version": 1,
        "scope": case.SELECTION_SCOPE, "status": "accepted", "accepted_by": "root",
        "test_feedback_used": False, "dataset_id": "synthetic", "variant": "V4", "model_seed": 0,
        "split": "validation", "checkpoint_sha256": sha(checkpoint), "source_manifest_sha256": source_digest,
        "training_dataset_path": str(root / data), "dataset_fingerprint": engine.dataset_fingerprint(str(root / data)),
        "node_order_sha256": "c" * 64, "full_validation_origin_count": 1,
        "validation_origin_record": ref(root, origin),
        "selection_rule": {"focal": "largest_training_activity_score_then_node_id",
                            "neighbor": "largest_mean_valid_neighbor_weight_in_full_validation_then_node_id"},
        "focal_node": 0, "neighbor_node": 1})
    launch_doc = json.loads(training_binding.read_text())
    launch_evidence = [Path(x["path"]) for x in
        launch_doc["dataset_sources"] + launch_doc["code_sources"]]
    launch_evidence.append(Path(launch_doc["config_source"]["path"]))
    selection_acceptance = put(root, "reviews/case-selection-root.json", {
        "status": "accepted", "accepted_by": "root", "verdict": "pass", "test_feedback_used": False,
        "evidence_sha256": {p.relative_to(root).as_posix(): sha(p) for p in
            (selection, origin, training_binding, config, checkpoint, *launch_evidence)}})

    matrix_sources = sorted(required_sources - {"edsgn/scripts/bind_training_inputs.py"})
    job_id, output, alias = "synthetic-V4-case-seed0", "runs/case/synthetic/V4/seed0", "runs/evidence/synthetic-case.json"
    matrix_path = root / "protocols/case/admission-matrix.json"
    binding_path = root / "runs/evidence/case-admission.json"
    job_path = root / "runs/jobs/case-synthetic.json"
    argv = ["--checkpoint", checkpoint.relative_to(root).as_posix(), "--dataset-path", str(root / data),
            "--selection", selection.relative_to(root).as_posix(), "--dataset-id", "synthetic",
            "--split", "test", "--output", output, "--managed-job-spec", job_path.relative_to(root).as_posix(),
            "--pretest-admission-binding", binding_path.relative_to(root).as_posix(), "--device", "cuda"]
    matrix = {"schema_version": 1, "scope": "prospective_case_pretest_job_admission_matrix",
        "status": "accepted_before_test", "source_refs": [ref(root, root / x) for x in matrix_sources],
        "jobs": [{"job_id": job_id, "driver": "case_v3", "method_id": "V4", "dataset_id": "synthetic",
            "seed_id": 0, "training_variant": "V4", "training_preparation": prep,
            "expected_driver_arguments": argv, "output_dir": output, "phase_receipt_alias_path": alias,
            "case_selection_path": selection.relative_to(root).as_posix(),
            "case_domain_path": domain.relative_to(root).as_posix(),
            "c06_descriptor_path": descriptor.relative_to(root).as_posix()}]}
    put(root, matrix_path.relative_to(root).as_posix(), matrix)
    matrix_ref = ref(root, matrix_path)

    # Copy official state support and create a canonical P2/P3/P3A-pass, P4-running ledger.
    for rel in (ordinary.PIPELINE_VALIDATOR_REL, ordinary.PIPELINE_DEFINITION_REL, *ordinary.PIPELINE_SIBLINGS):
        copied(root, rel)
    contract = put(root, "protocols/contract-fixture.json", {"contract": "temp"})
    contract_ref = ref(root, contract)
    definition = json.loads((root / ordinary.PIPELINE_DEFINITION_REL).read_text(encoding="utf-8-sig"))
    stages, history = [], []
    for sid in definition["stages"]:
        passed = sid in {"P0", "P1", "P1A", "P1B", "P2", "P3", "P3A"}
        status = "passed" if passed else ("running" if sid == "P4" else "pending")
        stages.append({"id": sid, "status": status, "contract_version": "v1",
            "gate_verdict": "pass" if passed else ("pending" if status == "pending" else "pass"),
            "inputs": [contract_ref] if passed else [],
            "evidence": [matrix_ref] if sid == "P3A" else ([contract_ref] if passed else [])})
        if passed: history.append({"stage": sid, "action": "pass"})
    put(root, "pipeline-state.json", {"schema_version": 1, "pipeline": "f-research-v1",
        "contract_version": "v1", "contract": [contract_ref], "history": history, "stages": stages})

    sources = [ref(root, root / x) for x in matrix_sources]
    job = {"schema_version": 2, "job_id": job_id, "driver": "case_v3", "argv": argv,
           "sources": sources, "pretest_execution_bindings": [], "phase_receipt_alias": alias}
    put(root, job_path.relative_to(root).as_posix(), job)
    # Populate admission's immutable references first; the queue excludes itself
    # from its own pretest refs, avoiding a hash cycle.
    admission = {"schema_version": 1, "scope": "case_pretest_job_admission", "status": "accepted_by_root",
        "accepted_by": "root", "test_feedback_used": False, "matrix": matrix_ref, "matrix_job_id": job_id,
        "global_test_queue": {"path": "protocols/global-test-queue.json", "sha256": "0"*64},
        "config": ref(root, config), "training_binding": ref(root, training_binding),
        "training_phase_receipt": ref(root, phase), "training_summary": ref(root, summary),
        "checkpoint": ref(root, checkpoint), "freeze": ref(root, freeze), "policy_config": None,
        "candidate_selection": None, "root_numeric_acceptance": ref(root, acceptance),
        "case_selection": ref(root, selection), "case_domain": ref(root, domain),
        "c06_descriptor": ref(root, descriptor), "selection_root_acceptance": ref(root, selection_acceptance),
        "input_refs": []}
    # All extant independent refs plus every file from the true launch binding.
    independent = all_refs({k: v for k, v in admission.items() if k != "global_test_queue"})
    launch = json.loads(training_binding.read_text())
    for x in launch["dataset_sources"] + launch["code_sources"]:
        independent[x["path"]] = x["sha256"]
    for rel in (ordinary.PIPELINE_STATE_REL, ordinary.PIPELINE_VALIDATOR_REL,
                ordinary.PIPELINE_DEFINITION_REL, *ordinary.PIPELINE_SIBLINGS):
        independent[rel] = sha(root / rel)
    # Refs in source inventories are guaranteed by the copied package and author tree.
    admission["input_refs"] = [{"path": p, "sha256": h} for p, h in sorted(independent.items())]
    queue_path = root / "protocols/global-test-queue.json"
    q_pre = dict(independent)
    q_pre.update({x["path"]: x["sha256"] for x in sources})
    q_pre.pop("protocols/global-test-queue.json", None)
    qrow = {"job_id": job_id, "job_spec_path": job_path.relative_to(root).as_posix(),
        "pretest_execution_binding_paths": [binding_path.relative_to(root).as_posix()], "driver": "case_v3",
        "output_dir": output, "expected_origin_count": 24, "coverage_field": "origin_count",
        "expected_report_scope": "frozen_case_signal_sensitivity_diagnostic", "expected_phase": "evaluation",
        "report_path": output + "/case-signal-sensitivity-report.json", "phase_receipt_alias_path": alias,
        "principal_files": [output + "/" + name for name in (
            "predictions-teacher.npz", "predictions-relationship_removal.npz",
            "predictions-focal_retention_removal.npz", "metrics-teacher-by-horizon-feature.csv",
            "metrics-relationship_removal-by-horizon-feature.csv", "metrics-focal_retention_removal-by-horizon-feature.csv",
            "case-signal-semantic-proof.jsonl", "evaluation_manifest.json", "case-signal-sensitivity-report.json",
            "output-closure.json")],
        "pretest_refs": [{"path": p, "sha256": h} for p, h in sorted(q_pre.items())]}
    put(root, queue_path.relative_to(root).as_posix(), {"schema_version": 1,
        "scope": "prospective_complete_test_job_set", "status": "frozen_before_test",
        "queue_id": "case-real-layout", "jobs": [qrow]})
    admission["global_test_queue"] = ref(root, queue_path)
    # Job and admission refs are now non-cyclic; update job before hashing it.
    put(root, binding_path.relative_to(root).as_posix(), admission)
    job["pretest_execution_bindings"] = [ref(root, binding_path)]
    put(root, job_path.relative_to(root).as_posix(), job)
    put(root, binding_path.relative_to(root).as_posix(), admission)
    # Update P3A matrix evidence to exact matrix digest (state is not yet hashed by anything).
    state_path = root / "pipeline-state.json"
    state = json.loads(state_path.read_text())
    next(x for x in state["stages"] if x["id"] == "P3A")["evidence"] = [matrix_ref]
    put(root, "pipeline-state.json", state)
    return (root, ref(root, binding_path), job, job_path, binder, matrix, admission)


def test_full_case_admission_real_layout_and_snapshot_coverage(tmp_path):
    started = time.perf_counter()
    root, binding_ref, job, job_path, binder, matrix, admission = fixture(tmp_path)
    old_root = binder.ROOT; binder.ROOT = root
    try:
        consumed = case.validate_admission(root, binding_ref, job, job_path)
    finally:
        binder.ROOT = old_root
    assert consumed["pipeline-state.json"]
    assert consumed["protocols/global-test-queue.json"]
    assert consumed["runs/train/synthetic/SeqSNN/V4/seed0/checkpoint.pt"]
    assert consumed["data/processed/synthetic-n64-identity-change0.3-miss0/test_seed200.npz"]
    assert consumed["reviews/case-validation-origins.json"]
    assert consumed["edsgn/src/edsgn/model.py"]
    assert time.perf_counter() >= started


@pytest.mark.parametrize("mutation", ["policy", "missing_training_binding_queue_ref", "missing_fullval_source",
    "stale_config", "stale_selection", "stale_source", "duplicate_cross_stage_exact", "duplicate_within_stage"])
def test_case_admission_mutations_fail_closed(tmp_path, mutation):
    root, binding_ref, job, job_path, binder, matrix, admission = fixture(tmp_path)
    old_root = binder.ROOT; binder.ROOT = root
    try:
        if mutation == "policy":
            job["argv"][job["argv"].index("--device") + 1] = "cpu"
            put(root, job_path.relative_to(root).as_posix(), job)
            with pytest.raises(case.AdmissionError): case.validate_admission(root, binding_ref, job, job_path)
            return
        if mutation == "stale_config":
            p = root / admission["config"]["path"]; p.write_bytes(p.read_bytes() + b"tamper")
        elif mutation == "stale_selection":
            p = root / admission["case_selection"]["path"]; p.write_bytes(p.read_bytes() + b"tamper")
        elif mutation == "stale_source":
            p = root / case.DRIVER_REL; p.write_bytes(p.read_bytes() + b"tamper")
        elif mutation == "missing_training_binding_queue_ref":
            qpath = root / admission["global_test_queue"]["path"]
            q = json.loads(qpath.read_text()); q["jobs"][0]["pretest_refs"] = [x for x in q["jobs"][0]["pretest_refs"] if x["path"] != admission["training_binding"]["path"]]
            put(root, qpath.relative_to(root).as_posix(), q)
        elif mutation == "missing_fullval_source":
            p = root / "reviews/case-validation-origins.json"; p.unlink()
        elif mutation in {"duplicate_cross_stage_exact", "duplicate_within_stage"}:
            sp = root / "pipeline-state.json"; state = json.loads(sp.read_text())
            p3, p3a = [x for x in state["stages"] if x["id"] in {"P3", "P3A"}]
            if mutation == "duplicate_cross_stage_exact": p3a["inputs"].append(p3a["evidence"][0])
            else: p3a["evidence"].append(p3a["evidence"][0])
            put(root, "pipeline-state.json", state)
        with pytest.raises((case.AdmissionError, ValueError, FileNotFoundError)):
            case.validate_admission(root, binding_ref, job, job_path)
    finally:
        binder.ROOT = old_root
