import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "staging" / "guarded_primary_family_assembler_v1.py"
SPEC = importlib.util.spec_from_file_location("guarded_primary_assembler_v1_test", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
ORIGINAL_ROOT = module.ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def fixture(tmp_path, monkeypatch, *, unidentifiable_id=None):
    root = tmp_path.resolve()
    monkeypatch.setattr(module, "ROOT", root)
    definitions_path = root / "protocols/primary-endpoint-registry-root-preparation-v2.json"
    definitions_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ORIGINAL_ROOT / "protocols/primary-endpoint-registry-root-preparation-v2.json", definitions_path)
    monkeypatch.setattr(module, "REGISTRY_DEFINITIONS", definitions_path)
    producer_copy = root / "edsgn/staging/typed_endpoint_producer_v3.py"
    producer_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ORIGINAL_ROOT / "edsgn/staging/typed_endpoint_producer_v3.py", producer_copy)
    monkeypatch.setattr(module, "PRODUCER_PATH", producer_copy)

    definitions = json.loads(definitions_path.read_text(encoding="utf-8"))
    plan_rows, result_rows, acceptance_rows = [], [], []
    registry_path = root / "protocols/accepted-test-endpoint-registry.json"
    records = []
    for definition in definitions["primary_endpoints"]:
        endpoint_id = definition["endpoint_id"]
        claim = definition["claim"]
        outcome = definition["outcome_type"]
        plan_path = root / "plans" / f"{endpoint_id}.json"
        plan = {"schema_version": 2, "endpoint_id": endpoint_id, "claim": claim,
                "outcome_type": outcome, "seed_ids": [0, 1, 2, 3, 4]}
        dump(plan_path, plan)
        plan_ref = {"endpoint_id": endpoint_id, "claim": claim,
                    "path": plan_path.relative_to(root).as_posix(), "sha256": digest(plan_path)}
        plan_rows.append(plan_ref)
        result_path = root / definition["result_path"]
        identifiable = endpoint_id != unidentifiable_id
        result = {"schema_version": 1, "scope": module.PRODUCER_SCOPE, "split": "test",
                  "endpoint_id": endpoint_id, "claim": claim, "outcome_type": outcome,
                  "plan_path": plan_ref["path"], "plan_sha256": plan_ref["sha256"],
                  "registry_path": registry_path.relative_to(root).as_posix(),
                  "registry_sha256": "pending", "seed_ids": [0, 1, 2, 3, 4],
                  "n_full_origins": 12, "identifiable": identifiable,
                  "producer_source_sha256": {str(producer_copy.resolve()): digest(producer_copy)}}
        if identifiable:
            raw_kind = module.EXPECTED_RAW_P_KIND[outcome]
            analysis = {raw_kind: 0.01, "estimate": -0.2}
            result.update(raw_p_kind=raw_kind, raw_p=0.01, effect_estimate=-0.2, analysis=analysis)
            if outcome == "forecast_mae_noninferiority":
                analysis.update(upper_95=0.05)
                result.update(epsilon_raw=0.1, upper_95=0.05, percentile_95_interval=None)
            else:
                analysis.update(percentile_95_interval=[-0.4, -0.1])
                result.update(epsilon_raw=None, upper_95=None, percentile_95_interval=[-0.4, -0.1])
        else:
            result.update(raw_p_kind=module.EXPECTED_RAW_P_KIND[outcome], raw_p=None,
                          effect_estimate=None, percentile_95_interval=None, upper_95=None,
                          epsilon_raw=None, reason="toy unidentifiable")
        result_rows.append((result_path, result))
        records.append((endpoint_id, result_path))

    registry = dump(registry_path, {"schema_version": 1, "split": "test",
                                    "artifact_status": "accepted_pretest_registry", "endpoint_plans": plan_rows})
    registry_sha = digest(registry)
    for path, result in result_rows:
        result["registry_sha256"] = registry_sha
        dump(path, result)
        acceptance_rows.append({"endpoint_id": result["endpoint_id"],
                                "path": path.relative_to(root).as_posix(),
                                "sha256": digest(path), "accepted": True})
    acceptance = dump(root / "reviews/root-result-acceptance.json", {
        "schema_version": 1, "status": "accepted", "accepted_by": "root", "verdict": "pass",
        "scope": module.ACCEPTANCE_SCOPE, "split": "test",
        "registry_path": registry.relative_to(root).as_posix(), "registry_sha256": registry_sha,
        "test_feedback_used": False, "pipeline_advanced": False,
        "scientific_claim_decision": "not_made", "records": acceptance_rows})

    stages = []
    ids = ["P0", "P1", "P1A", "P1B", "P2", "P3", "P3A", "P4", "P5", "P5A"] + [f"P{i}" for i in range(6, 22)]
    for stage_id in ids:
        row = {"id": stage_id, "status": "passed", "gate_verdict": "pass"}
        if stage_id == "P5":
            row = {"id": stage_id, "status": "running"}
        stages.append(row)
    dump(root / "pipeline-state.json", {"stages": stages})
    return root, registry, acceptance, root / "runs/assembled.json", acceptance_rows


def test_complete_accepted_registry_assembles_49_members(tmp_path, monkeypatch):
    root, registry, acceptance, output, _ = fixture(tmp_path, monkeypatch)
    result = module.assemble_primary_families(registry, acceptance, output, _test_root=root)
    assert {k: len(v["members"]) for k, v in result["families"].items()} == {"H1": 9, "H2": 16, "H3": 24}
    assert all(v["status"] == "identified" for v in result["families"].values())
    assert all(member["holm_adjusted_p"] is not None for family in result["families"].values() for member in family["members"])
    assert result["pipeline_advanced"] is False


def test_one_unidentifiable_member_keeps_full_family_and_no_holm_values(tmp_path, monkeypatch):
    root, registry, acceptance, output, _ = fixture(tmp_path, monkeypatch, unidentifiable_id="H1-noise-only-tv-V4-V9")
    result = module.assemble_primary_families(registry, acceptance, output, _test_root=root)
    family = result["families"]["H1"]
    assert len(family["members"]) == 9
    assert family["claim_status"] == "inconclusive"
    assert family["holm_denominator"] is None
    assert all(row["holm_adjusted_p"] is None for row in family["members"])


def test_unaccepted_or_stale_producer_mapping_fails_closed(tmp_path, monkeypatch):
    root, registry, acceptance, output, rows = fixture(tmp_path, monkeypatch)
    accepted = json.loads(acceptance.read_text(encoding="utf-8"))
    accepted["records"].pop()
    dump(acceptance, accepted)
    with pytest.raises(module.AssemblyError, match="all 49"):
        module.assemble_primary_families(registry, acceptance, output, _test_root=root)


def test_wrong_root_acceptance_or_modified_result_is_rejected(tmp_path, monkeypatch):
    root, registry, acceptance, output, rows = fixture(tmp_path, monkeypatch)
    accepted = json.loads(acceptance.read_text(encoding="utf-8"))
    accepted["accepted_by"] = "reviewer"
    dump(acceptance, accepted)
    with pytest.raises(module.AssemblyError, match="root acceptance"):
        module.assemble_primary_families(registry, acceptance, output, _test_root=root)


def test_post_acceptance_result_mutation_is_rejected_before_assembly(tmp_path, monkeypatch):
    root, registry, acceptance, output, rows = fixture(tmp_path, monkeypatch)
    path = root / rows[0]["path"]
    result = json.loads(path.read_text(encoding="utf-8"))
    result["raw_p"] = 0.5
    dump(path, result)
    with pytest.raises(module.AssemblyError, match="stale"):
        module.assemble_primary_families(registry, acceptance, output, _test_root=root)
