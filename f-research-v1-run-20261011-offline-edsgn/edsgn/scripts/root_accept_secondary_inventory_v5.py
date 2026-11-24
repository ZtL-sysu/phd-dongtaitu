"""Root metadata review of the prospective secondary allocation; never launch."""
from pathlib import Path
import collections
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn/src"))
from edsgn.robustness import canonical_recipe, recipe_digest

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            assert key not in value, (path, key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)

def main():
    before_state = sha(ROOT / "pipeline-state.json")
    rel = "protocols/secondary-replay-rows-v5/logical-secondary-replay-inventory-v5.json"
    doc = read(ROOT / rel)
    old = read(ROOT / "protocols/secondary-replay-rows-v4/logical-secondary-replay-inventory-v4.json")
    assert doc["rows"] == old["rows"]
    rows = doc["rows"]
    expected = {"missingness":624, "count-or-event-perturbation":78,
                "synthetic-clean-scenario":104, "trained-state-control":60,
                "count-loss-poisson":2, "synthetic-n256-clean":18,
                "quota-age-grid":1536, "graph-only-periodic-reference":8,
                "simple-baseline-clean":8, "case-signal-sensitivity":8}
    assert len(rows) == 2446 and dict(collections.Counter(r["category"] for r in rows)) == expected == doc["category_counts"]
    assert len({r["logical_id"] for r in rows}) == len(rows)
    assert len({r["output_path_prospective"] for r in rows}) == len(rows)
    assert doc["status"] == "prepared_unadmitted_unexecuted" and doc["accepted_by_root"] is None
    evidence = {rel: sha(ROOT / rel)}
    def check_ref(ref):
        if ref is None:
            return
        path = (ROOT / ref["path"]).resolve()
        assert path.is_relative_to(ROOT) and path.is_file()
        assert path.suffix.casefold() != ".npz", "This metadata review does not open/hash dataset archives"
        assert sha(path) == ref["sha256"], ref["path"]
        evidence[path.relative_to(ROOT).as_posix()] = ref["sha256"]
    for path, digest in doc["design_input_sha256"].items():
        check_ref({"path":path, "sha256":digest})
    for item in doc["recipe_catalog"].values():
        check_ref({"path":item["path"], "sha256":item["sha256"]})
        recipe = read(ROOT / item["path"])
        assert recipe == canonical_recipe(recipe) == item["recipe"]
        assert recipe_digest(recipe) == item["canonical_recipe_sha256"]
    assert len(doc["recipe_catalog"]) == 14
    for row in rows:
        assert row["status"] == "planning_only_unadmitted_unexecuted"
        assert row["gate_accepted"] is False and row["test_accessed"] is False
        for key in ("checkpoint_sha256", "freeze_sha256", "training_binding_sha256", "driver_binding_sha256", "source_bundle_sha256", "policy_config_ref"):
            assert row[key] is None, (row["logical_id"], key)
        for key in ("config_candidate_ref", "domain_manifest_ref", "template_ref"):
            check_ref(row.get(key))
        assert not any(c in row["output_path_prospective"] for c in ':<>"|?*')
        if row["recipe_id"]:
            assert row["recipe_sha256"] == doc["recipe_catalog"][row["recipe_id"]]["canonical_recipe_sha256"]
        if row["category"] == "count-or-event-perturbation":
            assert row["dataset"] in {"email", "mooc", "email-episode"}
    assert collections.Counter(r["action"] for r in rows if r["category"] == "quota-age-grid") == {"prediction":768, "profile":768}
    assert collections.Counter(r["device"] for r in rows if r["category"] == "quota-age-grid") == {"cpu":768, "cuda:0":768}
    for row in rows:
        if row["category"] == "simple-baseline-clean":
            assert row["method_id"] in {"persistence", "linear"} and row["driver_candidate_path"] is None
    assert sha(ROOT / "pipeline-state.json") == before_state
    for path in ("edsgn/scripts/prepare_secondary_replay_inventory_v5.py", "logs/secondary-replay-inventory-v5-equivalence.json", "reviews/secondary-replay-inventory-v5-preparation.md"):
        evidence[path] = sha(ROOT / path)
    receipt = {"scope":"secondary_logical_allocation_metadata_preparation_only", "accepted_by":"root", "verdict":"pass",
               "logical_rows":2446, "category_counts":expected, "v4_rows_exactly_preserved":True,
               "evidence_sha256":evidence, "canonical_recipe_digest_recomputed":True,
               "actual_admission_accepted":False, "actual_execution_completed":False, "scientific_hypothesis_accepted":False,
               "pipeline_advanced":False, "test_payload_accessed":False,
               "limitations":["Future configuration/checkpoint/freeze/policy/driver authorities remain unresolved.",
                               "Resource feasibility and executable full matrix are not established.",
                               "Simple and case-sensitivity guarded execution routes remain unfinished."]}
    dest = ROOT / "reviews/secondary-replay-inventory-v5-root-preparation-v1.json"
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    md = dest.with_suffix(".md")
    with md.open("x", encoding="utf-8") as stream:
        stream.write("# Root secondary inventory metadata preparation review\n\nPassed: 2,446 unique logical rows and output paths, exact v4 row preservation, category/action/device counts, all existing metadata references and 14 canonical recipe digests. Future execution authorities remain null.\n\nNo dataset archives, model, training or test payload were accessed. This accepts preparation only; it does not pass P3A, establish resource feasibility, authorize launch or complete an experiment.\n")
    print(json.dumps({"status":"pass", "rows":len(rows), "receipt":str(dest), "receipt_sha256":sha(dest)}))

if __name__ == "__main__":
    main()
