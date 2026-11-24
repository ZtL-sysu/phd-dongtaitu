"""Root acceptance of the full-policy focal SNN intervention toy kernel."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))
def main():
    before = sha(ROOT / "pipeline-state.json")
    writer = read("logs/case-signal-sensitivity-kernel-v2-tests.json")
    review = read("reviews/case-signal-sensitivity-kernel-independent-v2.json")
    assert writer["status"] == "passed" and writer["tests_passed"] == 6
    assert review["verdict"] == "pass_engineering_toy_scope"
    for key in ("scientific_acceptance", "actual_data_or_checkpoint_accessed", "gpu_or_training_used"):
        assert review[key] is False
    evidence = {}
    for ref in [writer["source"], writer["tests"], writer["test_log"], writer["dependency"],
                review["source"], review["tests"], review["test_log"], *review["consumed_sources"]]:
        assert sha(ROOT / ref["path"]) == ref["sha256"]
        evidence[ref["path"]] = ref["sha256"]
    assert evidence["edsgn/staging/case_signal_sensitivity_kernel_v2.py"] == "a49f823cdae53a2f246dba7875a033ada470ab16bb05b8ffdfe811d9ac0c68b5"
    assert "6 passed" in (ROOT / review["test_log"]["path"]).read_text(encoding="utf-8")
    for rel in ("logs/case-signal-sensitivity-kernel-v2-tests.json", "reviews/case-signal-sensitivity-kernel-v2.md",
                "reviews/case-signal-sensitivity-kernel-independent-v2.md", "reviews/case-signal-sensitivity-kernel-independent-v2.json",
                "edsgn/src/edsgn/model.py", "edsgn/src/edsgn/scheduling.py"):
        evidence[rel] = sha(ROOT / rel)
    assert sha(ROOT / "pipeline-state.json") == before
    result = {"scope":"full_policy_focal_snn_signal_sensitivity_kernel_engineering_only",
              "accepted_by":"root", "verdict":"pass", "evidence_sha256":evidence,
              "writer_cpu_tests_passed":6, "independent_cpu_tests_passed":6,
              "root_source_review":["Strict built-in integer indices and directed nonself relation.",
                                    "Excluded neighbor slot does not enlarge or refill the original candidate cap.",
                                    "Only a row with blocked support is renormalized; untouched weights and validity remain bit-exact.",
                                    "Actual aligned support change and deferred informative log-g are recomputed after graph intervention.",
                                    "Retention arm consumes the original V4 teacher graph, weights, validity and full-policy actions; only focal neighbor-channel log-g is zeroed.",
                                    "Temporary hooks/method overrides are restored and parameter bytes remain unchanged."],
              "scientific_hypothesis_accepted":False, "pipeline_advanced":False,
              "actual_admission_accepted":False, "actual_full_domain_execution_completed":False,
              "limitations":["Full reference policy only; skipped graph/state policy semantics are not established by this kernel.",
                              "Downstream prediction/monitor feedback can change later other-node and local-channel values.",
                              "Validation-only node/neighbor selection, guarded real-domain driver, complete global admission and reports remain unfinished."]}
    path = ROOT / "reviews/case-signal-sensitivity-kernel-v2-root-engineering-v1.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    with path.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write("# Root focal SNN sensitivity kernel engineering acceptance\n\nAccepted after root source review and six writer plus six independent CPU toy fixtures. V2 fixes permissive node IDs and all-row weight renormalization found in v1. Original V4 LIF, local paths, candidate cap and deferred retention are retained; direct interventions are the selected edge exclusion and focal neighbor-channel g=1 under the teacher graph/actions.\n\nThis proves engineering semantics under the full policy only. Later state/monitor feedback can propagate across nodes. No actual data, checkpoint, full-domain replay, node selection or scientific result is accepted. The gated driver and whole-campaign closure remain unfinished.\n")
    print(json.dumps({"status":"pass_engineering_only", "receipt":str(path), "sha256":sha(path)}))
if __name__ == "__main__": main()
