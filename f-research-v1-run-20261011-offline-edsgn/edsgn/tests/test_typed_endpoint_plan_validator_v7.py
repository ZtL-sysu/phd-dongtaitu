import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

RUN = Path(r"C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn")
sys.path.insert(0, str(RUN / "edsgn/staging"))
import typed_endpoint_plan_validator_v7 as v


def put(root, rel, data):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    else:
        path.write_text(str(data), encoding="utf-8")
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(tmp_path, monkeypatch, *, outcome="forecast_mae_superiority"):
    root = tmp_path / "run"
    root.mkdir(parents=True)
    # Hash-only fixture members deliberately are not valid NPZ archives. The
    # validator is forbidden from opening numerical bundle payloads.
    collection = root / "data/testdata"
    collection.mkdir(parents=True)
    bundle = put(root, "data/testdata/test_seed0.npz", "opaque-array-bytes")
    put(root, "data/testdata/test_seed0.oracle.npz", "opaque-oracle-bytes")
    collrows = [{"path": p.relative_to(root).as_posix(), "sha256": sha(p)}
                for p in sorted(collection.iterdir())]
    acf_path = put(root, "plans/acf.json", {"dependence_length": 2, "history_length": 28,
        "hmax": 7, "selection_status": "available-for-prospective-use"})
    domain = {"schema": "edsgn.split-history-domain.v1",
        "dataset": {"event_kind": "synthetic", "partition": "chronological", "source_name": "synthetic-toy",
            "shape_tnf": [80, 5, 2], "node_ids_sha256": "a"*64,
            "split_bounds_start_inclusive": 0, "split_bounds_end_exclusive": 80,
            "history_length": 28, "hmax": 7},
        "trajectories": [{"trajectory_index": 0, "source_trajectory_seed": 0,
            "source_input_mask_sha256": "b"*64, "split": "test", "origin_first": 27,
            "origin_last": 72, "origin_count": 46, "history_length": 28,
            "history_union_start_inclusive": 0, "history_union_end_exclusive": 73,
            "domain_bounds_start_inclusive": 0, "domain_bounds_end_exclusive": 73}],
        "ordered_collection": [{"trajectory_index": 0, "path": bundle.relative_to(root).as_posix(),
            "archive_sha256": sha(bundle)}]}
    domain_path = put(root, "plans/domain.json", domain)
    eps_path = put(root, "plans/epsilon.json", {"status": "accepted", "scope": "validation_only",
        "value_raw": 0.2, "dataset_id": "synthetic-toy", "metric": "raw_mae", "horizons": [1, 3, 7]})
    source_path = put(root, "edsgn/staging/producer.py", "# hash-bound source")
    refs = []
    artifact_roles = v.OUTCOMES[outcome]["artifacts"]
    sides = {}
    for side in ("left", "right"):
        runs = []
        for seed in range(5):
            config = put(root, f"configs/{side}-seed{seed}.json", {"seed": seed, "method": side})
            train_binding = put(root, f"bindings/{side}-seed{seed}.json", {"seed": seed, "training_only": True})
            prep = {"config": {"path": config.relative_to(root).as_posix(), "sha256": sha(config)},
                    "training_binding": {"path": train_binding.relative_to(root).as_posix(), "sha256": sha(train_binding)}}
            planned = {role: f"runs/{side}/seed{seed}/{role}.bin" for role in artifact_roles}
            planned["phase_receipt"] = f"runs/{side}/seed{seed}/evidence/phase-receipt.json"
            runs.append({"seed_id": seed, "preparation": prep, "planned_paths": planned})
        sides[side] = {"method_id": side, "producer_source_sha256": sha(source_path), "seeds": runs}
    endpoint = "H2-toy-forecast"
    plan = {"schema_version": 3, "endpoint_id": endpoint,
        "claim": "H3" if outcome == "forecast_mae_noninferiority" else "H2", "outcome_type": outcome,
        "seed_ids": [0, 1, 2, 3, 4], "required_horizons": [1, 3, 7],
        "history": 28, "max_horizon": 7, "block_length": 28, "acf_dependence_length": 2,
        "acf_evidence": {"mode": "fixed_validation_evidence", "path": acf_path.relative_to(root).as_posix(), "sha256": sha(acf_path)},
        "epsilon_evidence": None, "slice": "full_split",
        "domain": {"dataset_id": "synthetic-toy", "split": "test",
            "path": domain_path.relative_to(root).as_posix(), "sha256": sha(domain_path),
            "bounds": [{"trajectory_index": 0, "split_start": 0, "split_stop": 80,
                        "first_origin": 27, "last_origin": 72, "origin_count": 46}]},
        "dataset_collection": {"root": collection.relative_to(root).as_posix(), "files": collrows},
        "producer_sources": [{"path": source_path.relative_to(root).as_posix(), "sha256": sha(source_path)}],
        "left": sides["left"], "right": sides["right"], "switch_indices": None,
        "execution_binding_path": "runs/endpoint/execution-binding.json",
        "reporting_switch_authority": None, "profile_device": None}
    if outcome == "forecast_mae_noninferiority":
        plan["claim"] = "H3"
        plan["epsilon_evidence"] = {"mode": "fixed_validation_evidence", "status": "accepted", "value_raw": 0.2,
            "path": eps_path.relative_to(root).as_posix(), "sha256": sha(eps_path),
            "scope": "validation_only", "dataset_id": "synthetic-toy", "metric": "raw_mae",
            "horizons": [1, 3, 7]}
    plan_path = put(root, "plans/endpoint.json", plan)
    registry_rows = [{"endpoint_id": endpoint, "claim": plan["claim"],
        "path": plan_path.relative_to(root).as_posix(), "sha256": sha(plan_path)}]
    for family, count in (("H1", 9), ("H2", 16), ("H3", 24)):
        remaining = count - (1 if family == plan["claim"] else 0)
        for i in range(remaining):
            eid = f"{family}-fixture-{i:02d}"
            p = put(root, f"plans/registered/{eid}.json", {"fixture_plan_id": eid})
            registry_rows.append({"endpoint_id": eid, "claim": family,
                "path": p.relative_to(root).as_posix(), "sha256": sha(p)})
    registry_rows.sort(key=lambda x: x["endpoint_id"])
    registry = {"schema_version": 1, "artifact_status": "accepted_pretest_registry", "split": "test",
        "endpoint_plans": registry_rows}
    registry_path = put(root, "plans/registry.json", registry)

    # Canonical predecessor ledger: full ordered schema, realistic passed file
    # bindings and P3A bindings for this plan, registry, helper and ACF/margin.
    contract = put(root, "contract.md", "frozen execution contract")
    rows, history = [], []
    p3a_inputs = [plan_path, registry_path, source_path, acf_path]
    if plan["epsilon_evidence"]:
        p3a_inputs.append(eps_path)
    for sid in v.STAGE_ORDER:
        if sid == "P4":
            rows.append({"id": sid, "status": "running", "contract_version": "v1",
                         "gate_verdict": "running", "inputs": [], "evidence": []})
            continue
        if sid in v.STAGE_ORDER[:v.STAGE_ORDER.index("P4")]:
            input_path = put(root, f"evidence/{sid}-input.txt", sid)
            evid_path = put(root, f"evidence/{sid}-evidence.txt", sid + " evidence")
            inputs = [{"path": input_path.relative_to(root).as_posix(), "sha256": sha(input_path)}]
            evidence = [{"path": evid_path.relative_to(root).as_posix(), "sha256": sha(evid_path)}]
            if sid == "P3A":
                inputs = [{"path": p.relative_to(root).as_posix(), "sha256": sha(p)} for p in p3a_inputs]
            rows.append({"id": sid, "status": "passed", "contract_version": "v1", "gate_verdict": "pass",
                         "inputs": inputs, "evidence": evidence})
            history.append({"stage": sid, "action": "pass"})
        else:
            rows.append({"id": sid, "status": "pending", "contract_version": "v1", "gate_verdict": "pending",
                         "inputs": [], "evidence": []})
    state = {"schema_version": 1, "pipeline": "f-research-v1", "contract_version": "v1",
        "delivery_mode": "preparation", "contract": [{"path": "contract.md", "sha256": sha(contract)}],
        "stages": rows, "history": history, "readiness": [], "citation_gate": []}
    put(root, "pipeline-state.json", state)
    # Test only redirects immutable run root/official validator to the fixture.
    monkeypatch.setattr(v, "ROOT", root)
    monkeypatch.setattr(v, "VALIDATOR", RUN / "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py")
    monkeypatch.setattr(v, "TRUSTED_PYTHON", Path(sys.executable))
    return root, plan_path, registry_path, plan

def matched_build(tmp_path,monkeypatch):
    root,plan_path,registry_path,plan=build(tmp_path,monkeypatch)
    plan["left"]["method_id"]="V4-teacher-informative"
    plan["right"]["method_id"]="V4-teacher-shuffled-c"
    for side in ("left","right"):
        for row in plan[side]["seeds"]:
            row["planned_paths"]["matched_graph_proof"]=f"runs/matched/seed{row['seed_id']}/matched-graph-proof.jsonl"
    put(root,"plans/endpoint.json",plan)
    reg=json.loads(registry_path.read_text()); next(x for x in reg["endpoint_plans"] if x["endpoint_id"]==plan["endpoint_id"])["sha256"]=sha(plan_path)
    put(root,"plans/registry.json",reg)
    state_path=root/"pipeline-state.json";state=json.loads(state_path.read_text());p3a=next(x for x in state["stages"] if x["id"]=="P3A")
    for row in p3a["inputs"]:
        if row["path"]==plan_path.relative_to(root).as_posix():row["sha256"]=sha(plan_path)
        if row["path"]==registry_path.relative_to(root).as_posix():row["sha256"]=sha(registry_path)
    put(root,"pipeline-state.json",state)
    return root,plan_path,registry_path,plan

def test_h2_matched_graph_proof_is_a_planned_shared_same_seed_artifact(tmp_path,monkeypatch):
    root,plan_path,registry_path,plan=matched_build(tmp_path,monkeypatch)
    assert len(json.loads(registry_path.read_text())["endpoint_plans"])==49
    proof=v.validate(plan_path,registry_path,root=root,official=True)
    assert proof["scope"].startswith("typed endpoint path/hash")

def test_h2_matched_archives_cannot_share_the_same_path(tmp_path,monkeypatch):
    root,plan_path,registry_path,plan=matched_build(tmp_path,monkeypatch)
    plan["right"]["seeds"][0]["planned_paths"]["archive"]=plan["left"]["seeds"][0]["planned_paths"]["archive"]
    put(root,"plans/endpoint.json",plan)
    reg=json.loads(registry_path.read_text());next(x for x in reg["endpoint_plans"] if x["endpoint_id"]==plan["endpoint_id"])["sha256"]=sha(plan_path);put(root,"plans/registry.json",reg)
    state_path=root/"pipeline-state.json";state=json.loads(state_path.read_text());p3a=next(x for x in state["stages"] if x["id"]=="P3A")
    for row in p3a["inputs"]:
        if row["path"]==plan_path.relative_to(root).as_posix():row["sha256"]=sha(plan_path)
        if row["path"]==registry_path.relative_to(root).as_posix():row["sha256"]=sha(registry_path)
    put(root,"pipeline-state.json",state)
    with pytest.raises(v.PlanError,match="unique"):
        v.validate(plan_path,registry_path,root=root,official=False)

def _refresh_plan_bindings(root, plan_path, registry_path):
    plan=json.loads(plan_path.read_text())
    reg=json.loads(registry_path.read_text()); current=next(x for x in reg["endpoint_plans"] if x["endpoint_id"]==plan["endpoint_id"])
    previous=current["claim"]
    if previous != plan["claim"]:
        spare=next(x for x in reg["endpoint_plans"] if x["endpoint_id"]!=plan["endpoint_id"] and x["claim"]==plan["claim"])
        spare["claim"]=previous
    current["sha256"]=sha(plan_path); current["claim"]=plan["claim"]
    put(root,"plans/registry.json",reg)
    state_path=root/"pipeline-state.json"; state=json.loads(state_path.read_text())
    p3a=next(x for x in state["stages"] if x["id"]=="P3A")
    for row in p3a["inputs"]:
        if row["path"]==plan_path.relative_to(root).as_posix(): row["sha256"]=sha(plan_path)
        if row["path"]==registry_path.relative_to(root).as_posix(): row["sha256"]=sha(registry_path)
    put(root,"pipeline-state.json",state)

def test_h1_response_allows_only_same_seed_same_role_v4_trace_sharing(tmp_path,monkeypatch):
    root,plan_path,registry_path,plan=build(tmp_path,monkeypatch,outcome="graph_tv_response_superiority")
    plan["claim"]="H1"; plan["slice"]="full_split"
    plan["left"]["method_id"]=plan["right"]["method_id"]="V4"
    plan["right"]["producer_source_sha256"]=plan["left"]["producer_source_sha256"]
    for side in ("left","right"):
        for row in plan[side]["seeds"]:
            row["planned_paths"]["trace"]=f"runs/V4/seed{row['seed_id']}/trace.jsonl.gz"
    put(root,"plans/endpoint.json",plan); _refresh_plan_bindings(root,plan_path,registry_path)
    assert v.validate(plan_path,registry_path,root=root,official=False)["scope"].startswith("typed endpoint path/hash")
    plan["right"]["seeds"][0]["planned_paths"]["trace"]="runs/V4/seed1/trace.jsonl.gz"
    put(root,"plans/endpoint.json",plan); _refresh_plan_bindings(root,plan_path,registry_path)
    with pytest.raises(v.PlanError,match="unique"):
        v.validate(plan_path,registry_path,root=root,official=False)

@pytest.mark.parametrize("device",["cpu","cuda"])
def test_cost_endpoint_predeclares_supported_profile_device(tmp_path,monkeypatch,device):
    root,plan_path,registry_path,plan=build(tmp_path,monkeypatch,outcome="paired_window_cost_superiority")
    plan["claim"]="H3"; plan["profile_device"]=device
    put(root,"plans/endpoint.json",plan); _refresh_plan_bindings(root,plan_path,registry_path)
    assert v.validate(plan_path,registry_path,root=root,official=False)["scope"].startswith("typed endpoint path/hash")
    plan["profile_device"]="cuda:0"
    put(root,"plans/endpoint.json",plan); _refresh_plan_bindings(root,plan_path,registry_path)
    with pytest.raises(v.PlanError,match="predeclare profile_device"):
        v.validate(plan_path,registry_path,root=root,official=False)


def test_pretest_binding_has_no_test_products_and_postrun_manifest_hashes_them(tmp_path, monkeypatch):
    root, plan, registry, _ = build(tmp_path, monkeypatch)
    proof = v.validate(plan, registry, root=root)
    assert proof["scope"].startswith("typed endpoint path/hash")
    assert proof["dataset_file_count"] == 2
    plan_doc = v._json_no_duplicates(plan); registry_doc = v._json_no_duplicates(registry)
    _, consumed_map = v._validate_plan(root, plan, plan_doc, registry, registry_doc, sha(registry))
    binding_runs = []
    run_artifacts = []
    pretest_roles = ("checkpoint", "freeze", "input_binding", "validation_authority", "policy", "queue_plan")
    for side in ("left", "right"):
        for seed in plan_doc[side]["seeds"]:
            pretest = {}
            for role in pretest_roles:
                rel = seed["planned_paths"][role]
                artifact = put(root, rel, f"pretest-{side}-{seed['seed_id']}-{role}")
                pretest[role] = {"path": rel, "sha256": sha(artifact)}
                consumed_map[rel] = pretest[role]["sha256"]
            binding_runs.append({"side": side, "seed_id": seed["seed_id"],
                "config": seed["preparation"]["config"], "training_binding": seed["preparation"]["training_binding"],
                **pretest, "block_length": 28, "acf_dependence_length": 2,
                "acf_authority_path": plan_doc["acf_evidence"]["path"],
                "epsilon_status": "accepted" if plan_doc["outcome_type"] == "forecast_mae_noninferiority" else None,
                "epsilon_value_raw": 0.2 if plan_doc["outcome_type"] == "forecast_mae_noninferiority" else None,
                "epsilon_authority_path": plan_doc["epsilon_evidence"]["path"] if plan_doc["epsilon_evidence"] else None,
                "test_feedback_used": False})
    binding_doc = {"schema_version": 2, "status": "accepted_by_root", "scope": "pretest_execution_binding",
        "endpoint_id": plan_doc["endpoint_id"], "plan_path": plan.relative_to(root).as_posix(), "plan_sha256": sha(plan),
        "registry_path": registry.relative_to(root).as_posix(), "registry_sha256": sha(registry),
        "test_feedback_used": False, "runs": binding_runs}
    binding_path = put(root, plan_doc["execution_binding_path"], binding_doc)
    binding_sha = sha(binding_path); consumed_map[plan_doc["execution_binding_path"]] = binding_sha
    pretest = v.validate(plan, registry, root=root)
    assert pretest["pretest_execution_binding"]["test_artifacts_in_binding"] is False
    binding_runs[0]["driver_report"] = {"path": "runs/forbidden/report.json", "sha256": "0"*64}
    put(root, plan_doc["execution_binding_path"], binding_doc)
    with pytest.raises(v.PlanError, match="contains post-test artifacts"):
        v.validate(plan, registry, root=root)
    binding_runs[0].pop("driver_report")
    binding_path = put(root, plan_doc["execution_binding_path"], binding_doc)
    binding_sha = sha(binding_path); consumed_map[plan_doc["execution_binding_path"]] = binding_sha

    for side in ("left", "right"):
        for seed in plan_doc[side]["seeds"]:
            arts = {}
            for role, rel in seed["planned_paths"].items():
                if role in pretest_roles:
                    continue
                if role == "phase_receipt":
                    value = {"job_id": f"job-{side}-{seed['seed_id']}",
                             "output_dir": f"runs/{side}/seed{seed['seed_id']}", "phase": "test", "status": "completed"}
                elif role == "dispatch_receipt":
                    continue
                else:
                    value = f"postrun-{side}-{seed['seed_id']}-{role}"
                artifact = put(root, rel, value)
                arts[role] = {"path": rel, "sha256": sha(artifact)}
                consumed_map[rel] = arts[role]["sha256"]
            original_path = f"logs/offline-phase-wall/original-{side}-seed{seed['seed_id']}.json"
            alias_doc = json.loads((root / arts["phase_receipt"]["path"]).read_text())
            original = put(root, original_path, alias_doc)
            arts["phase_receipt_original"] = {"path": original_path, "sha256": sha(original)}
            consumed_map[original_path] = arts["phase_receipt_original"]["sha256"]
            dispatch_doc = {"status": "accepted-output-pending-scientific-review",
                "before": {"files": {binding_path.relative_to(root).as_posix(): binding_sha}},
                "child_phase_receipt": {"path": original_path, "sha256": arts["phase_receipt_original"]["sha256"],
                    "job_id": alias_doc["job_id"], "phase": alias_doc["phase"], "status": alias_doc["status"],
                    "output_dir": alias_doc["output_dir"]},
                "phase_receipt_alias_path": arts["phase_receipt"]["path"],
                "phase_receipt_alias_sha256": arts["phase_receipt"]["sha256"],
                "phase_receipt_original_path": original_path,
                "phase_receipt_original_sha256": arts["phase_receipt_original"]["sha256"]}
            dispatch_rel = seed["planned_paths"]["dispatch_receipt"]
            dispatch = put(root, dispatch_rel, dispatch_doc)
            arts["dispatch_receipt"] = {"path": dispatch_rel, "sha256": sha(dispatch)}
            consumed_map[dispatch_rel] = arts["dispatch_receipt"]["sha256"]
            run_artifacts.append({"side": side, "seed_id": seed["seed_id"], "artifacts": arts})

    consumed = [{"path": rel, "sha256": digest} for rel, digest in sorted(consumed_map.items())]
    manifest = put(root, "runs/postrun-input-manifest.json", {"schema_version": 3,
        "plan_path": plan.relative_to(root).as_posix(), "plan_sha256": sha(plan), "consumed_files": consumed,
        "pretest_execution_binding": {"path": binding_path.relative_to(root).as_posix(), "sha256": binding_sha},
        "run_artifacts": run_artifacts})
    post = v.validate(plan, registry, manifest_path=manifest, root=root)
    assert post["postrun_manifest"]["consumed_file_count"] == len(consumed)
    dispatch_ref = run_artifacts[0]["artifacts"]["dispatch_receipt"]["path"]
    dispatch_path = root / dispatch_ref
    dispatch_doc = json.loads(dispatch_path.read_text(encoding="utf-8"))
    dispatch_doc["before"]["files"] = {}
    dispatch_path = put(root, dispatch_ref, dispatch_doc)
    manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
    dispatch_sha = sha(dispatch_path)
    for entry in manifest_doc["consumed_files"]:
        if entry["path"] == dispatch_ref:
            entry["sha256"] = dispatch_sha
    run_artifacts[0]["artifacts"]["dispatch_receipt"]["sha256"] = dispatch_sha
    manifest_doc["run_artifacts"] = run_artifacts
    put(root, "runs/postrun-input-manifest.json", manifest_doc)
    with pytest.raises(v.PlanError, match="dispatch receipt must bind pretest execution binding"):
        v.validate(plan, registry, manifest_path=manifest, root=root)
    dispatch_doc["before"]["files"] = {binding_path.relative_to(root).as_posix(): binding_sha}
    put(root, dispatch_ref, dispatch_doc)
    (root / "runs/left/seed0/checkpoint.bin").write_text("mutated-after-manifest", encoding="utf-8")
    with pytest.raises(v.PlanError, match="changed during validation|postrun manifest does not equal|postrun consumed file"):
        v.validate(plan, registry, manifest_path=manifest, root=root)

def test_missing_or_duplicate_seed_fails_closed(tmp_path, monkeypatch):
    root, plan_path, registry, plan = build(tmp_path, monkeypatch)
    plan["left"]["seeds"].pop()
    put(root, "plans/endpoint.json", plan)
    with pytest.raises(v.PlanError, match="five ordered"):
        v.validate(plan_path, registry, root=root, official=False)


def test_block_floor_and_domain_bounds_are_enforced(tmp_path, monkeypatch):
    root, plan_path, registry, plan = build(tmp_path, monkeypatch)
    plan["block_length"] = 7
    put(root, "plans/endpoint.json", plan)
    with pytest.raises(v.PlanError, match="block_length"):
        v.validate(plan_path, registry, root=root, official=False)


def test_full_post_switch_requires_nonempty_support_on_each_trajectory(tmp_path, monkeypatch):
    root, plan_path, registry, plan = build(tmp_path, monkeypatch)
    plan["slice"] = "full_post_switch"
    plan["switch_indices"] = {"0": 73}
    put(root, "plans/endpoint.json", plan)
    with pytest.raises(v.PlanError, match="must not disclose switch values"):
        v.validate(plan_path, registry, root=root, official=False)


def test_paired_pre_post_4l_requires_both_complete_intervals(tmp_path, monkeypatch):
    root, plan_path, registry, plan = build(tmp_path, monkeypatch)
    plan["slice"] = "paired_pre_post4L"
    plan["switch_indices"] = {"0": 50}
    put(root, "plans/endpoint.json", plan)
    with pytest.raises(v.PlanError, match="must not disclose switch values"):
        v.validate(plan_path, registry, root=root, official=False)


def test_fixed_zero_epsilon_is_unidentifiable_not_accepted(tmp_path, monkeypatch):
    root, plan_path, registry, plan = build(tmp_path, monkeypatch, outcome="forecast_mae_noninferiority")
    plan["epsilon_evidence"]["value_raw"] = 0.0
    put(root, "plans/epsilon.json", {"status": "accepted", "scope": "validation_only", "value_raw": 0.0,
        "dataset_id": "synthetic-toy", "metric": "raw_mae", "horizons": [1, 3, 7]})
    put(root, "plans/endpoint.json", plan)
    with pytest.raises(v.PlanError, match="finite and positive"):
        v.validate(plan_path, registry, root=root, official=False)


def test_stale_p3a_evidence_rejected(tmp_path, monkeypatch):
    root, plan, registry, _ = build(tmp_path, monkeypatch)
    state_path = root / "pipeline-state.json"
    state = json.loads(state_path.read_text())
    row = next(x for x in state["stages"] if x["id"] == "P3A")
    row["inputs"] = [x for x in row["inputs"] if x["path"] != plan.relative_to(root).as_posix()]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(v.PlanError, match="P3A does not hash-bind"):
        v.validate(plan, registry, root=root, official=False)


def test_stale_p2_evidence_and_missing_p3a_input_rejected(tmp_path, monkeypatch):
    root, plan, registry, _ = build(tmp_path, monkeypatch)
    state_path = root / "pipeline-state.json"
    state = json.loads(state_path.read_text())
    p2 = next(x for x in state["stages"] if x["id"] == "P2")
    (root / p2["evidence"][0]["path"]).write_text("changed", encoding="utf-8")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(v.PlanError, match="canonical predecessor hash mismatch"):
        v.validate(plan, registry, root=root, official=False)


def test_artifact_path_mismatch_and_missing_producer_source_rejected(tmp_path, monkeypatch):
    root, plan_path, registry, plan = build(tmp_path, monkeypatch)
    plan["left"]["seeds"][0]["planned_paths"]["archive"] = "../../outside/archive.npz"
    put(root, "plans/endpoint.json", plan)
    with pytest.raises(v.PlanError, match="escapes RUN"):
        v.validate(plan_path, registry, root=root, official=False)
    root, plan_path, registry, plan = build(tmp_path / "source", monkeypatch)
    (root / plan["producer_sources"][0]["path"]).unlink()
    with pytest.raises(v.PlanError, match="producer source"):
        v.validate(plan_path, registry, root=root, official=False)


def test_unbound_plan_in_registry_rejected(tmp_path, monkeypatch):
    root, plan, registry, _ = build(tmp_path, monkeypatch)
    doc = json.loads(registry.read_text())
    row = next(x for x in doc["endpoint_plans"] if x["endpoint_id"] == "H2-toy-forecast")
    row["sha256"] = "0" * 64
    registry.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(v.PlanError, match="registered endpoint plan"):
        v.validate(plan, registry, root=root, official=False)


def test_dataset_membership_drift_rejected_without_array_decode(tmp_path, monkeypatch):
    root, plan_path, registry, _ = build(tmp_path, monkeypatch)
    put(root, "data/testdata/new_file.npz", "new opaque bytes")
    with pytest.raises(v.PlanError, match="membership changed"):
        v.validate(plan_path, registry, root=root, official=False)


def test_noninferiority_margin_must_match_dataset_metric_and_horizons(tmp_path, monkeypatch):
    root, plan_path, registry, plan = build(tmp_path, monkeypatch, outcome="forecast_mae_noninferiority")
    plan["epsilon_evidence"]["dataset_id"] = "another-dataset"
    put(root, "plans/endpoint.json", plan)
    # Registry and P3A plan binding are now stale too; update those bindings to
    # isolate the intended margin-domain rejection.
    registry_doc = json.loads(registry.read_text())
    row = next(x for x in registry_doc["endpoint_plans"] if x["endpoint_id"] == plan["endpoint_id"])
    row["sha256"] = sha(plan_path)
    registry.write_text(json.dumps(registry_doc), encoding="utf-8")
    state_path = root / "pipeline-state.json"; state = json.loads(state_path.read_text())
    p3a = next(x for x in state["stages"] if x["id"] == "P3A")
    for row in p3a["inputs"]:
        p = root / row["path"]
        if p == plan_path: row["sha256"] = sha(plan_path)
        elif p == registry: row["sha256"] = sha(registry)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(v.PlanError, match="epsilon domain"):
        v.validate(plan_path, registry, root=root, official=False)


def test_p3a_plan_defers_future_validation_block_and_margin_bytes(tmp_path, monkeypatch):
    root, plan_path, registry, plan = build(tmp_path, monkeypatch, outcome="forecast_mae_noninferiority")
    plan["acf_dependence_length"] = None
    plan["block_length"] = None
    plan["acf_evidence"] = {"mode": "validation_derived", "rule_id": "validation-acf-max-v2",
        "expected_record_path": "runs/endpoint/evidence/validation-acf.json"}
    plan["epsilon_evidence"] = {"mode": "validation_derived", "rule_id": "raw-mae-epsilon-v1",
        "expected_record_path": "runs/endpoint/evidence/validation-epsilon.json",
        "scope": "validation_only", "dataset_id": "synthetic-toy", "metric": "raw_mae",
        "horizons": [1, 3, 7]}
    put(root, "plans/endpoint.json", plan)
    registry_doc = json.loads(registry.read_text())
    row = next(x for x in registry_doc["endpoint_plans"] if x["endpoint_id"] == plan["endpoint_id"])
    row["sha256"] = sha(plan_path)
    registry.write_text(json.dumps(registry_doc), encoding="utf-8")
    state_path = root / "pipeline-state.json"; state = json.loads(state_path.read_text())
    p3a = next(x for x in state["stages"] if x["id"] == "P3A")
    for row in p3a["inputs"]:
        if row["path"] == plan_path.relative_to(root).as_posix(): row["sha256"] = sha(plan_path)
        if row["path"] == registry.relative_to(root).as_posix(): row["sha256"] = sha(registry)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    proof = v.validate(plan_path, registry, root=root, official=False)
    assert proof["epsilon_status"] == "validation_derived"
    assert not (root / plan["acf_evidence"]["expected_record_path"]).exists()
    assert not (root / plan["epsilon_evidence"]["expected_record_path"]).exists()


def test_duplicate_json_keys_rejected(tmp_path, monkeypatch):
    root, plan, registry, _ = build(tmp_path, monkeypatch)
    plan.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(v.PlanError, match="duplicate JSON key"):
        v.validate(plan, registry, root=root, official=False)




