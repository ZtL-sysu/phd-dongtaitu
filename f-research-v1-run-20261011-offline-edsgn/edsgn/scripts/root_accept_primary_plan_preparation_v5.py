"""Accept pre-registry structure only; no registry admission or payload decode."""
from pathlib import Path
import collections
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn/staging"))
import typed_endpoint_plan_validator_v9 as validator

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read(path):
    return validator._json_no_duplicates(Path(path))

def main():
    phase_before = sha(ROOT / "pipeline-state.json")
    inventory_path = ROOT / "protocols/primary-endpoint-plans-v5/preparation-inventory.json"
    inv = read(inventory_path)
    assert inv["artifact_status"] == "preparation-only-not-root-accepted-not-frozen-not-executed"
    assert inv["plan_count"] == 49 and inv["family_counts"] == {"H1":9, "H2":16, "H3":24}
    registry_path = ROOT / inv["root_preparation_registry"]["path"]
    registry = read(registry_path)
    assert sha(registry_path) == inv["root_preparation_registry"]["sha256"]
    evidence = {inventory_path.relative_to(ROOT).as_posix():sha(inventory_path)}
    for name in ("root_preparation_registry", "root_operational_definitions", "candidate_catalog", "generator", "metadata_preflight_source", "capacity_review"):
        ref = inv[name]
        assert sha(ROOT / ref["path"]) == ref["sha256"]
        evidence[ref["path"]] = ref["sha256"]
    assert inv["capacity_review"]["h1_h2_planned_invocations"] == 130
    for rel, digest in inv["plan_source_hashes"].items():
        assert sha(ROOT / rel) == digest
        evidence[rel] = digest
    for domain in inv["domains"]:
        original = read(ROOT / domain["accepted_domain"]["path"])
        desc = read(ROOT / domain["accepted_descriptor"]["path"])
        for name in ("accepted_domain", "accepted_descriptor"):
            ref = domain[name]
            assert sha(ROOT / ref["path"]) == ref["sha256"]
            evidence[ref["path"]] = ref["sha256"]
        expected = copy.deepcopy(original)
        expected["ordered_collection"] = [{k:item[k] for k in ("trajectory_index", "path", "archive_sha256")} for item in desc["ordered_collection"]]
        assert read(ROOT / domain["domain_path"]) == expected
        assert sha(ROOT / domain["domain_path"]) == domain["domain_sha256"]
        evidence[domain["domain_path"]] = domain["domain_sha256"]
    plans = inv["plans"]
    assert len({item["endpoint_id"] for item in plans}) == 49
    assert dict(collections.Counter(item["claim"] for item in plans)) == {"H1":9, "H2":16, "H3":24}
    future_paths = set()
    email_ni = 0
    for item in plans:
        path = ROOT / item["path"]
        assert sha(path) == item["sha256"]
        evidence[item["path"]] = item["sha256"]
        plan = read(path)
        assert plan["schema_version"] == 4 and plan["seed_ids"] == [0,1,2,3,4]
        assert plan["block_length"] is None and plan["acf_dependence_length"] is None
        if item["endpoint_id"].startswith("H3-email-") and plan["outcome_type"] == "forecast_mae_noninferiority":
            assert plan["epsilon_evidence"]["mode"] == "unidentifiable_by_rule"
            email_ni += 1
        for side in ("left", "right"):
            for seed in plan[side]["seeds"]:
                future_paths.update(seed["planned_paths"].values())
        try:
            validator._validate_plan(ROOT, path, plan, registry_path, registry, sha(registry_path))
        except validator.PlanError as exc:
            assert str(exc) == "endpoint registry must be schema v1 for test", (item["endpoint_id"], str(exc))
        else:
            raise AssertionError("Prospective preparation must not pass actual test admission")
    assert email_ni == 6 and len(future_paths) == 2780
    assert not any((ROOT / path).exists() for path in future_paths)
    assert sha(ROOT / "pipeline-state.json") == phase_before
    receipt = {"accepted_by":"root", "verdict":"pass", "scope":"primary49_pre_registry_plan_structure_preparation_only",
               "family_counts":{"H1":9,"H2":16,"H3":24}, "future_artifact_paths":2780,
               "all49_stop_at_missing_schema1_test_registry":True, "typed_domain_wrappers_exact_projection":True,
               "email_noninferiority_unidentifiable_rows_retained":6, "evidence_sha256":evidence,
               "actual_admission_accepted":False, "accepted_pretest_registry_created":False,
               "pipeline_advanced":False, "test_payload_decoded":False, "oracle_payload_decoded":False,
               "scientific_hypothesis_accepted":False,
               "limitations":["Post-registry binding, actual policies/checkpoints/margins/blocks, complete whole-campaign coverage and resource acceptance remain pending.",
                               "The global verifier and consumers still need extended simple/case routes before final registry freeze.",
                               "Actual H3 copied checkpoint bytes must match the independently accepted original training source."]}
    dest = ROOT / "reviews/primary49-plan-preparation-v5-root-acceptance-v1.json"
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with dest.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write("# Root review of prospective primary plans\n\nAccepted pre-registry structure only: all 49 plans retain the 9/16/24 families, ordered five seeds, exact metadata domain wrappers and 2,780 absent future artifacts. Using the actual schema2 preparation registry, all reach the missing schema1 test-registry rejection. The writer's earlier explicit root-acceptance barrier claim does not describe this actual unprojected call. Six original-email noninferiority rows remain unidentifiable.\n\nThe checks hash inventory files and read JSON metadata only. No test, target or oracle payload is decoded and no model runs. This does not accept the post-registry bindings, create a frozen registry, pass P3A or complete an experiment. Global simple/case integration and resource acceptance remain unfinished.\n")
    print(json.dumps({"status":"pass_preparation_only", "receipt":str(dest), "sha256":sha(dest)}))

if __name__ == "__main__":
    main()
