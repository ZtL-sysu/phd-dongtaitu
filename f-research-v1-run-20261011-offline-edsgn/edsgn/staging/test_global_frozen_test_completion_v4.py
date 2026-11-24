import hashlib
import json
from pathlib import Path

import pytest

import global_frozen_test_completion_v4 as completion
from global_frozen_test_completion_v4 import CompletionError, verify_completion, verify_queued_job


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


def _extension_fixture(root, drivers=("simple_v3", "case_v3"), mutation=None):
    root.mkdir(parents=True, exist_ok=True)
    job_specs=[]; output_specs=[]; rows=[]
    for index, driver in enumerate(drivers):
        jid=f"extension-{index}-{driver}"
        source_rel=f"edsgn/staging/{driver}-toy-source.py"
        source=_write(root/source_rel, f"# {driver} source pin\n")
        plan=_write(root/f"protocols/{jid}-plan.json", {"scope":"toy-pretest-plan","split":"test"})
        checkpoint=_write(root/f"runs/pre/{jid}-opaque.pt", b"opaque test fixture checkpoint")
        prerefs=[{"path":p.relative_to(root).as_posix(),"sha256":_sha(p)} for p in (source,plan,checkpoint)]
        fit=dev=auth=descriptor=domain=None
        if driver=="simple_v3":
            coefficient=_write(root/f"runs/fit/{jid}-coeff.npy",b"opaque fit coefficient")
            fit=_write(root/f"runs/fit/{jid}-fit.json", {"artifact_role":"train_only_simple_baseline_fit","split":"train","test_accessed":False,
                "dataset_path":"data/toy.npz","linear_coefficients_path":coefficient.relative_to(root).as_posix(),"linear_coefficients_sha256":_sha(coefficient)})
            dev=_write(root/f"reviews/{jid}-dev.json", {"accepted_by_root":True,"verdict":"pass","scope":"simple-baseline-development-fit",
                "fit_manifest_path":fit.relative_to(root).as_posix(),"fit_manifest_sha256":_sha(fit)})
            domain=_write(root/f"protocols/{jid}-domain.json", {"trajectories":[{"trajectory_index":0,"split":"test"}]})
            descriptor=_write(root/f"protocols/{jid}-descriptor.json", {"schema":"edsgn.c06-domain-preflight.v1","split":"test",
                "ordered_collection":[{"trajectory_index":0,"split":"test"}],
                "domain_manifest_path":domain.relative_to(root).as_posix(),"domain_manifest_sha256":_sha(domain)})
            auth=_write(root/f"reviews/{jid}-auth.json", {"status":"accepted_pretest_authorization","accepted_by_root":True,"verdict":"pass","method":"linear",
                "dataset_path":"data/toy.npz","coefficient_sha256":_sha(coefficient),
                "fit_manifest_path":fit.relative_to(root).as_posix(),"fit_manifest_sha256":_sha(fit),
                "domain_descriptor_path":descriptor.relative_to(root).as_posix(),"domain_descriptor_sha256":_sha(descriptor)})
            prerefs.extend({"path":p.relative_to(root).as_posix(),"sha256":_sha(p)} for p in (fit,dev,domain,descriptor,auth,coefficient))
        out_rel=f"runs/test/{jid}"
        report_name="evaluation_manifest.json" if driver=="simple_v3" else "case-signal-sensitivity-report.json"
        principal=( [f"{out_rel}/predictions.npz",f"{out_rel}/metrics_by_horizon.csv",f"{out_rel}/evaluation_manifest.json"]
            if driver=="simple_v3" else [f"{out_rel}/{name}" for name in completion.DRIVER_CONTRACTS[driver]["principal"]])
        alias_rel=f"runs/evidence/{jid}-phase.json"
        spec_rel=f"runs/jobs/{jid}.json"
        binding_rel=f"runs/pre/{jid}-binding.json"
        rows.append({"job_id":jid,"job_spec_path":spec_rel,"pretest_execution_binding_paths":[binding_rel],"driver":driver,
            "output_dir":out_rel,"expected_origin_count":2,"coverage_field":completion.DRIVER_CONTRACTS[driver]["coverage"],
            "expected_report_scope":completion.DRIVER_CONTRACTS[driver]["scope"],"expected_phase":"evaluation",
            "report_path":f"{out_rel}/{report_name}","phase_receipt_alias_path":alias_rel,"principal_files":principal,"pretest_refs":prerefs})
        job_specs.append((jid,driver,spec_rel,binding_rel,alias_rel,source,plan,checkpoint,fit,dev,auth,descriptor,domain))
        output_specs.append((jid,driver,out_rel,report_name,principal))
    queue=_write(root/"protocols/global-extension-queue.json", {"schema_version":1,"scope":"prospective_complete_test_job_set",
        "status":"frozen_before_test","queue_id":"extension-queue-v1","jobs":rows})
    qref={"path":queue.relative_to(root).as_posix(),"sha256":_sha(queue)}
    completions=[]
    for spec, output_spec in zip(job_specs,output_specs):
        jid,driver,spec_rel,binding_rel,alias_rel,source,plan,checkpoint,fit,dev,auth,descriptor,domain=spec
        jobrefpath=root/spec_rel
        binding=_write(root/binding_rel,{"schema_version":2,"global_test_queue":qref,
            "plan_path":plan.relative_to(root).as_posix(),"plan_sha256":_sha(plan),
            "checkpoint":{"path":checkpoint.relative_to(root).as_posix(),"sha256":_sha(checkpoint)}})
        source_paths=[source] + ([fit,dev,domain,descriptor,auth,coefficient] if driver=="simple_v3" else [])
        job=_write(jobrefpath,{"schema_version":2,"job_id":jid,"driver":driver,
            "argv":["--checkpoint",checkpoint.relative_to(root).as_posix(),"--dataset-path","data/test",
                    "--split","test","--output",output_spec[2]],
            "sources":[{"path":p.relative_to(root).as_posix(),"sha256":_sha(p)} for p in source_paths],
            "pretest_execution_bindings":[{"path":binding.relative_to(root).as_posix(),"sha256":_sha(binding)}],
            "phase_receipt_alias":alias_rel})
        jobsha=_sha(job)
        verify_queued_job(root,qref,spec_rel)
        out=root/output_spec[2];out.mkdir(parents=True)
        if driver=="simple_v3":
            pred=_write(out/"predictions.npz",b"opaque predictions npz")
            metrics=_write(out/"metrics_by_horizon.csv","horizon,mae\n1,0.2\n")
            seed=0
            if mutation=="simple-method": method="unsupported"
            elif mutation=="simple-seed": seed=2; method="linear"
            else: method="linear"
            manifest={"scope":"full_protocol","split":"test","method":method,"seed_id":seed,"n_windows":2,
                "output_hashes":{"predictions.npz":_sha(pred),"metrics_by_horizon.csv":_sha(metrics)},
                "provenance":{"fit_manifest_sha256":_sha(fit),"authorization_sha256":_sha(auth),"coefficient_sha256":_sha(coefficient)},
                "test_feedback_used":False,"oracle_accessed":False}
            report=_write(out/"evaluation_manifest.json",manifest)
        else:
            record_names={"teacher":"predictions-teacher.npz","relationship_removal":"predictions-relationship_removal.npz",
                "focal_retention_removal":"predictions-focal_retention_removal.npz",
                "metrics:teacher":"metrics-teacher-by-horizon-feature.csv",
                "metrics:relationship_removal":"metrics-relationship_removal-by-horizon-feature.csv",
                "metrics:focal_retention_removal":"metrics-focal_retention_removal-by-horizon-feature.csv"}
            records={}
            for key,name in record_names.items():
                file=_write(out/name,b"opaque npz" if name.endswith(".npz") else "horizon,mae\n1,0.2\n")
                records[key]={"path":name,"sha256":_sha(file)}
            proof=_write(out/"case-signal-semantic-proof.jsonl","{}\n{}\n{}\n{}\n")
            history=2; expected_lines=3 if mutation=="case-proof-count" else 2*history
            manifest_files={item["path"]:item["sha256"] for item in records.values()};manifest_files[proof.name]=_sha(proof)
            em=_write(out/"evaluation_manifest.json",{"scope":"case_signal_evaluation_manifest","split":"test","origin_count":2,
                "expected_semantic_proof_lines":expected_lines,"files":manifest_files})
            if mutation=="case-missing-arm": records.pop("metrics:focal_retention_removal")
            if mutation=="case-record-hash": records["teacher"]["sha256"]="0"*64
            report_doc={"scope":"frozen_case_signal_sensitivity_diagnostic","split":"test","origin_count":2,"history":history,
                "variant":"V4","model_seed_id":0,"same_window_policy":"full","oracle_accessed":False,
                "targets_passed_to_kernel":False,"selection":{"focal_node_index":0,"neighbor_node_index":1},
                "informative_teacher_parity_all_windows":True,"retention_graph_actions_exact_all_windows":True,
                "prediction_outputs":records,"semantic_proof_path":proof.name,"semantic_proof_sha256":_sha(proof),
                "semantic_proof_lines":expected_lines,"semantic_proof_expected_lines":expected_lines,
                "evaluation_manifest":{"path":em.name,"sha256":_sha(em)}}
            report=_write(out/"case-signal-sensitivity-report.json",report_doc)
            closure_files={p.name:_sha(p) for p in sorted(out.iterdir()) if p.is_file()}
            closure_sha=completion._canonical_sha(closure_files)
            _write(out/"output-closure.json",{"scope":"case_signal_output_directory_closure","self_excluded":True,
                "files":closure_files,"closure_sha256":closure_sha})
        if mutation=="extra-output" and jid==output_specs[0][0]: _write(out/"unexpected.log","extra")
        actual={p.name:_sha(p) for p in out.iterdir() if p.is_file()}
        childid=f"child-{jid}-uuid"
        phase_outputs=( {"predictions.npz":"0"*64} if mutation=="phase-output-map" and driver=="simple_v3" else actual )
        phase_doc={"scope":"offline-experiment-phase","status":"completed","job_id":childid,"phase":"evaluation",
            "output_dir":str(out.resolve()),"outputs":phase_outputs}
        original=_write(root/f"logs/offline-phase-wall/{childid}.json",phase_doc)
        alias=_write(root/alias_rel,b"different phase alias bytes" if mutation=="alias-source" else original.read_bytes())
        principal_names=[Path(x).name for x in output_spec[4]]
        dispatch_before={qref["path"]:qref["sha256"],job.relative_to(root).as_posix():jobsha,
            binding.relative_to(root).as_posix():_sha(binding)}
        for ref in rows[[x["job_id"] for x in rows].index(jid)]["pretest_refs"]: dispatch_before[ref["path"]]=ref["sha256"]
        dispatch=_write(root/f"logs/verified-frozen-dispatch/{jid}.json",{
            "schema_version":2,"job_id":jid,"driver":driver,"status":"accepted-output-pending-scientific-review","child_returncode":0,
            "before":{"files":dispatch_before},"after":{"files":dispatch_before},
            "validator_before":{"returncode":0},"validator_after":{"returncode":0},
            "child_phase_receipt":{"path":original.relative_to(root).as_posix(),"sha256":_sha(original),"job_id":childid,
                "phase":"evaluation","status":"completed","output_dir":str(out.resolve()),"principal_artifacts":{n:actual[n] for n in principal_names[:3]},
                "receipt_doc":phase_doc,"phase_receipt_original_path":original.relative_to(root).as_posix(),"phase_receipt_original_sha256":_sha(original),
                "phase_receipt_alias_path":alias.relative_to(root).as_posix(),"phase_receipt_alias_sha256":_sha(alias)},
            "phase_receipt_original_path":original.relative_to(root).as_posix(),"phase_receipt_original_sha256":_sha(original),
            "phase_receipt_alias_path":alias.relative_to(root).as_posix(),"phase_receipt_alias_sha256":_sha(alias)})
        outputs=[{"path":p.relative_to(root).as_posix(),"sha256":_sha(p)} for p in sorted(out.iterdir()) if p.is_file()]
        completions.append({"job_id":jid,"job_spec":{"path":job.relative_to(root).as_posix(),"sha256":jobsha},
            "dispatcher_receipt":{"path":dispatch.relative_to(root).as_posix(),"sha256":_sha(dispatch)},
            "phase_receipt_original":{"path":original.relative_to(root).as_posix(),"sha256":_sha(original)},
            "phase_receipt_alias":{"path":alias.relative_to(root).as_posix(),"sha256":_sha(alias)},
            "driver_report":{"path":report.relative_to(root).as_posix(),"sha256":_sha(report)},"output_files":outputs})
    completion_path=_write(root/"protocols/extension-completion.json",{"schema_version":1,"scope":"global_frozen_test_completion",
        "status":"accepted_by_root","accepted_by":"root","verdict":"pass","queue":qref,"jobs":completions})
    return root,{"path":completion_path.relative_to(root).as_posix(),"sha256":_sha(completion_path)},qref


@pytest.mark.parametrize("drivers", [("simple_v3",),("case_v3",),("simple_v3","case_v3")])
def test_global_completion_accepts_simple_case_and_mixed_closure(tmp_path, drivers):
    root, completion_ref, queue_ref=_extension_fixture(tmp_path,drivers)
    consumed=verify_completion(root,completion_ref,queue_ref)
    assert completion_ref["path"] in consumed and queue_ref["path"] in consumed
    assert len([p for p in consumed if p.endswith("predictions.npz") or "predictions-" in p]) >= len(drivers)


@pytest.mark.parametrize("mutation", ["simple-method","simple-seed","case-missing-arm","case-proof-count","case-record-hash","extra-output"])
def test_global_completion_rejects_simple_case_contract_mutations(tmp_path, mutation):
    driver="simple_v3" if mutation.startswith("simple-") else "case_v3"
    root, completion_ref, queue_ref=_extension_fixture(tmp_path,(driver,),mutation)
    with pytest.raises(CompletionError):
        verify_completion(root,completion_ref,queue_ref)


@pytest.mark.parametrize("mutation",["phase-output-map","alias-source"])
def test_global_completion_rejects_phase_output_map_and_alias_source_mutation(tmp_path, mutation):
    root, completion_ref, queue_ref=_extension_fixture(tmp_path,("simple_v3",),mutation)
    with pytest.raises(CompletionError):
        verify_completion(root,completion_ref,queue_ref)


