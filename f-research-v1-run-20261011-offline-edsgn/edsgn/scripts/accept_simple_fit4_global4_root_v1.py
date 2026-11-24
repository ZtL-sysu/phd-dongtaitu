"""Root engineering acceptance only; no training, data decoding or test admission."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def accept(name, scope, pins, additional, checks, counts, limits):
    evidence = {}
    for rel, expected in pins.items():
        actual = sha(ROOT / rel)
        if actual != expected.lower():
            raise RuntimeError(f"changed frozen evidence: {rel}")
        evidence[rel] = actual
    for rel in additional:
        evidence[rel] = sha(ROOT / rel)
    target = ROOT / "reviews" / name
    payload = {
        "schema_version": 1, "scope": scope, "accepted_by": "root", "verdict": "pass",
        "accepted_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_sha256": evidence, "root_review": checks, **counts,
        "pipeline_advanced": False, "scientific_hypothesis_accepted": False,
        "actual_test_authorized": False, "actual_execution_completed": False,
        "limitations": limits,
    }
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"path":str(target),"sha256":sha(target),"scope":scope}))

if sha(ROOT / "pipeline-state.json") != "2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093":
    raise RuntimeError("canonical state changed")

accept("simple-baseline-train-fit-verification-v4-root-engineering-v1.json",
       "train_only_simple_fit_verification_local_engineering_only", {
    "edsgn/staging/simple_baseline_train_fit_verification_v4.py":"ef3c91e9ffa75134298d6888d2c6b52ca13e746efd707366ebc01ecb155b7e1c",
    "edsgn/tests/test_simple_baseline_train_fit_verification_v4.py":"844e51d932479f0250c7080efaa6ef0c7af0d3e8bfda9efc57ece90b4d31a28c",
    "edsgn/scripts/owned_python_job_inventory_v1.py":"c31e70671f3846cdd8b1fb61e06aab45c0110d9ad5c5f1f3823377ab6d267d06",
    "reviews/count-loss-training-controller-v5-root-engineering-v1.json":"348b2e085cf7677010140e9c73cdd98f136c1a245b70bcd24ac007a9e7d548ea",
    "reviews/simple-baseline-train-fit-verification-v4-engineering.md":"30b784472bee2485a54faa9a5693bdf7fe28af25179802488d9f261b7c3e5eed",
    "reviews/simple-baseline-train-fit-verification-v4-engineering.json":"58d948b7d4c313eb23cf2e7a2c1d5752e6139206bf9a1c906270fbc1255e704f",
    "reviews/simple-baseline-train-fit-verification-v4-independent-review-v1.md":"82efe7b8f9feb33bebfd297b41323ca779396d265c98c0d7e5f66eef893d3873",
    "reviews/simple-baseline-train-fit-verification-v4-independent-review-v1.json":"53f5e993bc0afba1c1510ca55e3f048570fa6c5393a40c5dd8142347825ee8b2",
    "edsgn/src/edsgn/engine.py":"491da67e3c23d412003b838e3b44d0c732e4378c1b23bf7b29494768b97febb3",
    "edsgn/src/edsgn/data.py":"97fc2ab6a442f4b55b5c65ca065f6ebc5badfecbf08e00fc5c84e1e6aad638ff",
}, ["protocols/simple-baseline-train-fit-verification-root-design-v1.md",
    "logs/simple-baseline-train-fit-verification-v4-tests.log",
    "logs/simple-baseline-train-fit-verification-v4-root-tests-v1.log"], [
    "Independently reconstructs train-only float64 Gram/rhs/ridge coefficients and compares preserved bytes without rewriting legacy coefficients.",
    "Replays all ordered validation origins and required raw horizon/feature metrics; declares possible combined-source member materialization without scoring test origins.",
    "Binds the canonical 21 package modules plus verifier, strict source/input membership, official before-P3 validation and final serialization fences.",
    "Actual Guard checkpoint rejects a nonempty accepted owned-process inventory; CLI fixture tests verify refusal before loading and failed receipt with owned leases released.",
    "CLI holds both leases until completed receipt; coefficient mismatch fails and preserves evidence.",
], {"writer_cpu_tests_passed":20,"independent_cpu_tests_passed":20,"root_cpu_tests_passed":20}, [
    "All tests use small CPU fixtures or process mocks; the four actual train-only fit audits remain unexecuted.",
    "Actual audit requires the serial model resource, remaining single72-hour budget and separate independent numeric/root acceptance.",
    "No historical launch provenance, actual fit acceptance, P3 pass or test authorization is conferred.",
])

accept("global-frozen-test-completion-v4-root-engineering-v1.json",
       "whole_frozen_test_queue_metadata_completion_local_engineering_only", {
    "edsgn/staging/global_frozen_test_completion_v4.py":"e05ea055be0852aecc3b9a0113b177c54a3567ad8adf5c7bbb34aa08a887a5c3",
    "edsgn/staging/global_frozen_test_completion_v3.py":"6b4979738256b0ffe86dd30b94db6543c7ced3e578f88ca70410c0507d06a1c5",
    "edsgn/staging/test_global_frozen_test_completion_v4_all_drivers.py":"07e8a7da83be37e56a81bbf8c2f979ad03251d10770fb7ff6118d0be7b7c3dad",
    "reviews/global-completion-v4-seven-driver-cpu-v1.md":"648cef49bac189aaa26ed12b16b57d776a8a16ca64f8da6e7575fd85ae49662d",
    "logs/global-completion-v4-seven-driver-cpu-v1.json":"a8107b93bb2a4f40a435106f32cffcd9c10f945e329a25e44d8428198e2f2be0",
    "reviews/global-completion-v4-delta-independent-v1.md":"87323e8d552c0dd8a4a7e4da210fed5841e3b7aa2bfe7b7140caf92f6ec956b8",
    "reviews/global-completion-v4-delta-independent-v1.json":"e0fcdef2b3be2dcdb18cf6c311982d9374b285f13d1d9c031ce128df94aa286e",
}, ["edsgn/staging/test_global_frozen_test_completion_v4.py",
    "protocols/global-simple-case-extension-root-design-v2.md",
    "reviews/global-frozen-test-completion-independent-v3.md",
    "reviews/global-frozen-test-completion-independent-v3.json",
    "reviews/global-completion-v3-seven-driver-mixed-cpu-v1.md",
    "logs/global-completion-v4-root-tests-v1.log"], [
    "Root reviewed the v3 full implementation and v4 minimal delta; prior 31 independent v3 tests and mixed seven-driver fixtures remain relevant.",
    "Root65 CPU tests cover graph_v8, prediction_v5, h2_v4, profile_v5, c06_v5, simple_v3 and case_v3 together, without numerical NPZ decoding.",
    "Completed simple/case phase receipts require exact full output filename-to-SHA maps; resealed omission attacks now reject.",
    "Prediction output-list normalization rejects duplicate resolved paths even when hashes agree; exact original receipt, alias, dispatcher and output tree closure are retained.",
    "Unknown mandatory drivers and incomplete all-queue completion fail before oracle/report consumers.",
], {"writer_cpu_tests_passed":65,"independent_delta_cpu_tests_passed":4,"root_cpu_tests_passed":65}, [
    "Metadata fixture acceptance does not prove numerical semantics, scientific outcomes or RP inventory completeness.",
    "Graph/H2 legacy receipt contracts lack full phase maps; their exact closed output set and managed dispatcher closure remain mandatory.",
    "Actual simple/case managed admission, compatible dispatcher, post-P4 bindings, all actual test jobs and completion remain pending.",
])
