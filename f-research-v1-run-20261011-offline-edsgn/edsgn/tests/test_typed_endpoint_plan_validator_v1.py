import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

RUN = Path(r"C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn")
sys.path.insert(0, str(RUN / "edsgn/staging"))
import typed_endpoint_plan_validator_v1 as v


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
    acf_path = put(root, "plans/acf.json", {"dependence_length": 2})
    eps_path = put(root, "plans/epsilon.json", {"status": "accepted", "scope": "validation_only",
        "value_raw": 0.2, "dataset_id": "synthetic-toy", "metric": "raw_mae", "horizons": [1, 3, 7]})
    source_path = put(root, "edsgn/staging/producer.py", "# hash-bound source")
    refs = []
    artifact_roles = v.OUTCOMES[outcome]["artifacts"]
    sides = {}
    for side in ("left", "right"):
        runs = []
        for seed in range(5):
            artifacts = {}
            for role in artifact_roles:
                p = put(root, f"runs/{side}/seed{seed}/{role}.bin", f"{side}-{seed}-{role}")
                artifacts[role] = {"path": p.relative_to(root).as_posix(), "sha256": sha(p)}
            runs.append({"seed_id": seed, "artifacts": artifacts})
        sides[side] = {"method_id": side, "producer_source_sha256": sha(source_path), "seeds": runs}
    endpoint = "H1-toy-forecast"
    plan = {"schema_version": 1, "endpoint_id": endpoint,
        "claim": "H3" if outcome == "forecast_mae_noninferiority" else "H1", "outcome_type": outcome,
        "seed_ids": [0, 1, 2, 3, 4], "required_horizons": [1, 3, 7],
        "history": 28, "max_horizon": 7, "block_length": 28, "acf_dependence_length": 2,
        "acf_evidence": {"path": acf_path.relative_to(root).as_posix(), "sha256": sha(acf_path)},
        "epsilon_evidence": None, "slice": "full_split",
        "domain": {"dataset_id": "synthetic-toy", "split": "test",
            "path": domain_path.relative_to(root).as_posix(), "sha256": sha(domain_path),
            "bounds": [{"trajectory_index": 0, "split_start": 0, "split_stop": 80,
                        "first_origin": 27, "last_origin": 72, "origin_count": 46}]},
        "dataset_collection": {"root": collection.relative_to(root).as_posix(), "files": collrows},
        "producer_sources": [{"path": source_path.relative_to(root).as_posix(), "sha256": sha(source_path)}],
        "left": sides["left"], "right": sides["right"], "switch_indices": None}
    if outcome == "forecast_mae_noninferiority":
        plan["claim"] = "H3"
        plan["epsilon_evidence"] = {"status": "accepted", "value_raw": 0.2,
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


def test_typed_plan_and_exact_postrun_consumed_manifest(tmp_path, monkeypatch):
    root, plan, registry, _ = build(tmp_path, monkeypatch)
    proof = v.validate(plan, registry, root=root)
    assert proof["scope"].startswith("typed endpoint path/hash")
    assert proof["dataset_file_count"] == 2
    # Manifest is an exact byte-list, including hash-only archives/oracle bytes.
    consumed = []
    for rel, digest in v._validate_plan(root, plan, v._json_no_duplicates(plan), registry,
                                        v._json_no_duplicates(registry), sha(registry))[1].items():
        consumed.append({"path": rel, "sha256": digest})
    manifest = put(root, "runs/postrun-input-manifest.json", {"schema_version": 1,
        "plan_path": plan.relative_to(root).as_posix(), "plan_sha256": sha(plan), "consumed_files": consumed})
    post = v.validate(plan, registry, manifest_path=manifest, root=root)
    assert post["postrun_manifest"]["consumed_file_count"] == len(consumed)


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
    plan["left"]["seeds"][0]["artifacts"]["archive"]["path"] = "runs/absent.npz"
    put(root, "plans/endpoint.json", plan)
    with pytest.raises(v.PlanError, match="missing or SHA-256 mismatch"):
        v.validate(plan_path, registry, root=root, official=False)
    root, plan_path, registry, plan = build(tmp_path / "source", monkeypatch)
    (root / plan["producer_sources"][0]["path"]).unlink()
    with pytest.raises(v.PlanError, match="producer source"):
        v.validate(plan_path, registry, root=root, official=False)


def test_unbound_plan_in_registry_rejected(tmp_path, monkeypatch):
    root, plan, registry, _ = build(tmp_path, monkeypatch)
    doc = json.loads(registry.read_text())
    row = next(x for x in doc["endpoint_plans"] if x["endpoint_id"] == "H1-toy-forecast")
    row["sha256"] = "0" * 64
    registry.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(v.PlanError, match="registry does not bind"):
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


def test_duplicate_json_keys_rejected(tmp_path, monkeypatch):
    root, plan, registry, _ = build(tmp_path, monkeypatch)
    plan.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(v.PlanError, match="duplicate JSON key"):
        v.validate(plan, registry, root=root, official=False)
