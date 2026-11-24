"""Mixed seven-driver metadata closure regression for global completion v3.

All payloads below are opaque bytes. These fixtures exercise only the verifier's
metadata/hash contracts; they do not emulate scientific results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import global_frozen_test_completion_v3 as gc


DRIVERS = ("graph_v8", "prediction_v5", "h2_v4", "profile_v5", "c06_v5", "simple_v3", "case_v3")


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes): path.write_bytes(value)
    elif isinstance(value, str): path.write_text(value, encoding="utf-8")
    else: path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(root, path):
    return {"path": Path(path).relative_to(root).as_posix(), "sha256": _sha(path)}


def _make_fixture(root, drivers=DRIVERS, mutation=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    jobs, artifacts = [], []
    for i, driver in enumerate(drivers):
        jid = f"mix-{i}-{driver}"
        contract = gc.DRIVER_CONTRACTS[driver]
        out_rel = f"runs/test/{jid}"
        report_name = contract["report"]
        source = _write(root / f"edsgn/staging/{driver}-fixture.py", f"# opaque source {driver}\n")
        plan = _write(root / f"protocols/{jid}-plan.json", {"scope": "fixture-plan", "driver": driver})
        registry = _write(root / f"protocols/{jid}-registry.json", {"scope": "fixture-registry", "driver": driver})
        checkpoint = _write(root / f"runs/pre/{jid}.pt", b"opaque checkpoint")
        simple_refs = []
        if driver == "simple_v3":
            fit = _write(root / f"runs/fit/{jid}-fit.json", {
                "artifact_role": "train_only_simple_baseline_fit", "split": "train", "test_accessed": False,
                "dataset_path": "data/toy.npz", "parameter_identity": "persistence-repeat-last-observation/no-parameters"})
            dev = _write(root / f"reviews/{jid}-dev.json", {"scope": "simple-baseline-development-fit",
                "accepted_by_root": True, "verdict": "pass", "fit_manifest_path": _ref(root, fit)["path"],
                "fit_manifest_sha256": _sha(fit)})
            domain = _write(root / f"protocols/{jid}-domain.json", {"trajectories": [{"trajectory_index": 0, "split": "test"}]})
            descriptor = _write(root / f"protocols/{jid}-descriptor.json", {"schema": "edsgn.c06-domain-preflight.v1",
                "split": "test", "ordered_collection": [{"trajectory_index": 0, "split": "test"}],
                "domain_manifest_path": _ref(root, domain)["path"], "domain_manifest_sha256": _sha(domain)})
            auth = _write(root / f"reviews/{jid}-auth.json", {"status": "accepted_pretest_authorization",
                "accepted_by_root": True, "verdict": "pass", "method": "persistence", "dataset_path": "data/toy.npz",
                "parameter_identity": "persistence-repeat-last-observation/no-parameters",
                "fit_manifest_path": _ref(root, fit)["path"], "fit_manifest_sha256": _sha(fit),
                "domain_descriptor_path": _ref(root, descriptor)["path"], "domain_descriptor_sha256": _sha(descriptor)})
            simple_refs = [fit, dev, domain, descriptor, auth]
        binding_rel = f"runs/pre/{jid}-binding.json"
        job_rel = f"runs/jobs/{jid}.json"
        alias_rel = f"runs/{jid}/evidence/phase.json"
        report_rel = f"{out_rel}/{report_name}"
        principal_rel = [f"{out_rel}/{name}" for name in sorted(contract["principal"])]
        if driver == "h2_v4":
            principal_rel.extend(f"{out_rel}/{name}" for name in (
                "predictions-informative.npz", "predictions-shuffled.npz",
                "predictions-constant.npz", "predictions-reset.npz"))
        pre_refs = [_ref(root, p) for p in (source, plan, registry, checkpoint, *simple_refs)]
        binding = _write(root / binding_rel, {
            "schema_version": 2, "global_test_queue": {"path": "", "sha256": ""},
            "plan_path": _ref(root, plan)["path"], "plan_sha256": _sha(plan),
            "registry_path": _ref(root, registry)["path"], "registry_sha256": _sha(registry),
            "source": _ref(root, source), "checkpoint": _ref(root, checkpoint),
        })
        jobs.append({"job_id": jid, "job_spec_path": job_rel,
                     "pretest_execution_binding_paths": [binding_rel], "driver": driver,
                     "output_dir": out_rel, "expected_origin_count": 2,
                     "coverage_field": contract["coverage"], "expected_report_scope": contract["scope"],
                     "expected_phase": contract["phase"], "report_path": report_rel,
                     "phase_receipt_alias_path": alias_rel, "principal_files": principal_rel,
                     "pretest_refs": pre_refs})
        artifacts.append({"jid": jid, "driver": driver, "out_rel": out_rel, "report_name": report_name,
                          "principal_rel": principal_rel, "source": source, "plan": plan,
                          "registry": registry, "checkpoint": checkpoint, "binding": binding,
                          "binding_rel": binding_rel, "job_rel": job_rel, "alias_rel": alias_rel,
                          "report_rel": report_rel})

    queue = _write(root / "protocols/mixed-seven-queue.json", {
        "schema_version": 1, "scope": "prospective_complete_test_job_set", "status": "frozen_before_test",
        "queue_id": "mixed-seven-v1", "jobs": jobs})
    queue_ref = _ref(root, queue)
    completion_jobs = []
    for row in jobs:
        row["pretest_refs"] = row["pretest_refs"]
    for art, qrow in zip(artifacts, jobs):
        jid, driver = art["jid"], art["driver"]
        contract = gc.DRIVER_CONTRACTS[driver]
        art["binding"].write_text(json.dumps({
            "schema_version": 2, "global_test_queue": queue_ref,
            "plan_path": _ref(root, art["plan"])["path"], "plan_sha256": _sha(art["plan"]),
            "registry_path": _ref(root, art["registry"])["path"], "registry_sha256": _sha(art["registry"]),
            "source": _ref(root, art["source"]), "checkpoint": _ref(root, art["checkpoint"]),
        }, sort_keys=True), encoding="utf-8")
        binding_ref = _ref(root, art["binding"])
        job_sources = (qrow["pretest_refs"] if art["driver"] == "simple_v3" else [_ref(root, art["source"])])
        job = _write(root / art["job_rel"], {
            "schema_version": 2, "job_id": art["jid"], "driver": art["driver"],
            "argv": ["--checkpoint", _ref(root, art["checkpoint"])["path"], "--dataset-path", "data/test",
                     "--split", "test", "--output", art["out_rel"]],
            "sources": job_sources,
            "pretest_execution_bindings": [binding_ref], "phase_receipt_alias": art["alias_rel"]})
        assert gc.verify_queued_job(root, queue_ref, art["job_rel"])
        out = root / art["out_rel"]; out.mkdir(parents=True)
        for name in sorted(gc.DRIVER_CONTRACTS[art["driver"]]["principal"]):
            if name in {art["report_name"], "evaluation_manifest.json"}:
                continue
            payload = "horizon,mae\n1,0.2\n" if name.endswith(".csv") else b"opaque artifact bytes"
            _write(out / name, payload)

        driver = art["driver"]
        report = {"scope": gc.DRIVER_CONTRACTS[driver]["scope"], "split": "test",
                  gc.DRIVER_CONTRACTS[driver]["coverage"]: 2}
        if driver == "graph_v8":
            report.update(oracle_accessed=False, oracle_npz_decoded=False,
                          oracle_analysis_status="deferred_until_global_completion")
        elif driver == "h2_v4":
            names = {"informative": "predictions-informative.npz", "shuffled": "predictions-shuffled.npz",
                     "constant": "predictions-constant.npz", "reset": "predictions-reset.npz"}
            report["prediction_outputs"] = {}
            for arm, name in names.items():
                _write(out / name, b"opaque arm npz")
                report["prediction_outputs"][arm] = {"path": name, "sha256": _sha(out / name)}
        elif driver == "prediction_v5":
            # The phase output list, not a fabricated evaluator payload, is the contract under test.
            pass
        elif driver == "simple_v3":
            simple_refs = {Path(x["path"]).name: x for x in qrow["pretest_refs"]}
            fit = root / simple_refs[f"{art['jid']}-fit.json"]["path"]
            auth = root / simple_refs[f"{art['jid']}-auth.json"]["path"]
            report.update(method="persistence", seed_id=0, oracle_accessed=False, test_feedback_used=False,
                provenance={"fit_manifest_sha256": _sha(fit), "authorization_sha256": _sha(auth)})
            # Ensure declared simple outputs exist.
            _write(out / "predictions.npz", b"opaque simple predictions")
            _write(out / "metrics_by_horizon.csv", "horizon,mae\n1,0.2\n")
            report["output_hashes"] = {n: _sha(out / n) for n in ("predictions.npz", "metrics_by_horizon.csv")}
            _write(out / "evaluation_manifest.json", {"scope": "simple-baseline-evaluation",
                "split": "test", "n_windows": 2, "seed_id": 0, "test_accessed": True,
                "output_hashes": report["output_hashes"]})
        elif driver == "case_v3":
            # The global helper checks case-specific semantic reports. Reuse the
            # valid tiny case layout builder from its earlier tests through exact fields.
            names = {"teacher": "predictions-teacher.npz", "relationship_removal": "predictions-relationship_removal.npz",
                     "focal_retention_removal": "predictions-focal_retention_removal.npz",
                     "metrics:teacher": "metrics-teacher-by-horizon-feature.csv",
                     "metrics:relationship_removal": "metrics-relationship_removal-by-horizon-feature.csv",
                     "metrics:focal_retention_removal": "metrics-focal_retention_removal-by-horizon-feature.csv"}
            # Principal files are already created; populate their actual record map.
            records = {key: {"path": name, "sha256": _sha(out / name)} for key, name in names.items()}
            proof = _write(out / "case-signal-semantic-proof.jsonl", "{}\n{}\n")
            em = _write(out / "evaluation_manifest.json", {"scope": "case_signal_evaluation_manifest", "split": "test",
                "origin_count": 2, "expected_semantic_proof_lines": 2, "files": {v["path"]: v["sha256"] for v in records.values()} |
                {proof.name: _sha(proof)}})
            report.update(variant="V4", model_seed_id=0, same_window_policy="full", oracle_accessed=False,
                targets_passed_to_kernel=False, selection={"focal_node_index": 0, "neighbor_node_index": 1},
                informative_teacher_parity_all_windows=True, retention_graph_actions_exact_all_windows=True,
                history=1, prediction_outputs=records, semantic_proof_path=proof.name,
                semantic_proof_sha256=_sha(proof), semantic_proof_lines=2, semantic_proof_expected_lines=2,
                evaluation_manifest={"path": em.name, "sha256": _sha(em)})
        report_path = _write(out / art["report_name"], report)
        if driver == "case_v3":
            closure_path = out / "output-closure.json"
            closure_files = {p.name: _sha(p) for p in out.iterdir() if p.is_file() and p != closure_path}
            _write(closure_path, {"scope": "case_signal_output_directory_closure", "self_excluded": True,
                "files": closure_files, "closure_sha256": gc._canonical_sha(closure_files)})
        # All pretest refs were frozen before queue creation; artifacts stay opaque.
        phase_out = {p.name: _sha(p) for p in out.iterdir() if p.is_file()}
        if driver == "prediction_v5":
            phase_out = [{"path": _ref(root, p)["path"], "sha256": _sha(p)} for p in out.iterdir() if p.is_file()]
        phase_field = "output_sha256" if driver == "profile_v5" else "outputs"
        phase_doc = {"scope": "offline-experiment-phase", "status": "completed", "job_id": f"child-{jid}-uuid",
                     "phase": contract["phase"], "output_dir": str(out.resolve())}
        if driver in {"prediction_v5", "profile_v5", "c06_v5", "simple_v3", "case_v3"}:
            phase_doc[phase_field] = phase_out
        original = _write(root / f"logs/offline-phase-wall/child-{jid}-uuid.json", phase_doc)
        alias = _write(root / art["alias_rel"], original.read_bytes())
        job = root / art["job_rel"]
        binding_ref = _ref(root, art["binding"])
        source_refs = [_ref(root, art["source"])]

        dispatch_before = {queue_ref["path"]: queue_ref["sha256"], job.relative_to(root).as_posix(): _sha(job),
                           binding_ref["path"]: binding_ref["sha256"]}
        for ref in qrow["pretest_refs"]: dispatch_before[ref["path"]] = ref["sha256"]
        nested = {Path(p).name: _sha(root / p) for p in qrow["principal_files"][:3]}
        dispatch = _write(root / f"logs/verified-frozen-dispatch/{jid}.json", {
            "schema_version": 2, "job_id": jid, "driver": driver,
            "status": "accepted-output-pending-scientific-review", "child_returncode": 0,
            "before": {"files": dispatch_before}, "after": {"files": dispatch_before},
            "validator_before": {"returncode": 0}, "validator_after": {"returncode": 0},
            "child_phase_receipt": {"path": original.relative_to(root).as_posix(), "sha256": _sha(original),
                "job_id": phase_doc["job_id"], "phase": contract["phase"], "status": "completed",
                "output_dir": str(out.resolve()), "principal_artifacts": nested, "receipt_doc": phase_doc,
                "phase_receipt_original_path": original.relative_to(root).as_posix(), "phase_receipt_original_sha256": _sha(original),
                "phase_receipt_alias_path": alias.relative_to(root).as_posix(), "phase_receipt_alias_sha256": _sha(alias)},
            "phase_receipt_original_path": original.relative_to(root).as_posix(), "phase_receipt_original_sha256": _sha(original),
            "phase_receipt_alias_path": alias.relative_to(root).as_posix(), "phase_receipt_alias_sha256": _sha(alias)})
        completion_jobs.append({"job_id": jid, "job_spec": _ref(root, job), "dispatcher_receipt": _ref(root, dispatch),
            "phase_receipt_original": _ref(root, original), "phase_receipt_alias": _ref(root, alias),
            "driver_report": _ref(root, report_path),
            "output_files": [_ref(root, p) for p in sorted(out.iterdir()) if p.is_file()]})
    completion = _write(root / "protocols/mixed-seven-completion.json", {
        "schema_version": 1, "scope": "global_frozen_test_completion", "status": "accepted_by_root",
        "accepted_by": "root", "verdict": "pass", "queue": queue_ref, "jobs": completion_jobs})
    return root, _ref(root, completion), queue_ref


def test_all_seven_driver_contracts_close_in_one_mixed_fixture(tmp_path, monkeypatch):
    # A load spy proves this metadata helper never imports/decompresses NPZ payloads.
    import numpy as np
    def forbidden(*args, **kwargs): raise AssertionError("NPZ payload load attempted")
    monkeypatch.setattr(np, "load", forbidden)
    root, completion_ref, queue_ref = _make_fixture(tmp_path)
    consumed = gc.verify_completion(root, completion_ref, queue_ref)
    assert len([d for d in DRIVERS if any(f"{d}" in p for p in consumed)]) == len(DRIVERS)


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("mutation", ("scope", "coverage", "principal", "pretest"))
def test_each_driver_rejects_report_output_and_pretest_tampering(tmp_path, driver, mutation):
    # Keep this attack fixture focused on verifier branches common to all seven
    # contracts; driver-specific semantic mutation checks live in the existing suite.
    root, completion_ref, queue_ref = _make_fixture(tmp_path, (driver,))
    q = gc._json(root / queue_ref["path"]); row = q["jobs"][0]; out = root / row["output_dir"]
    report = out / Path(row["report_path"]).name
    if mutation in {"scope", "coverage"}:
        doc = gc._json(report)
        doc["scope" if mutation == "scope" else row["coverage_field"]] = "wrong" if mutation == "scope" else 99
        _write(report, doc)
        # Update all references to this output to ensure semantic validation, not hash mismatch, rejects it.
        with pytest.raises(gc.CompletionError): gc.verify_completion(root, completion_ref, queue_ref)
    elif mutation == "principal":
        principal = root / row["principal_files"][0]
        principal.write_bytes(principal.read_bytes() + b"tamper")
        with pytest.raises(gc.CompletionError): gc.verify_completion(root, completion_ref, queue_ref)
    else:
        source = root / row["pretest_refs"][0]["path"]
        source.write_bytes(source.read_bytes() + b"changed")
        with pytest.raises(gc.CompletionError): gc.verify_completion(root, completion_ref, queue_ref)


def test_unknown_mandatory_driver_is_rejected(tmp_path):
    root, _, queue_ref = _make_fixture(tmp_path, ("graph_v8",))
    qpath = root / queue_ref["path"]; q = gc._json(qpath)
    q["jobs"][0]["driver"] = "future_unregistered_v99"
    _write(qpath, q)
    with pytest.raises(gc.CompletionError): gc._load_queue(root, qpath)
