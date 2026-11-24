"""Root records source reviews; no model execution or canonical phase changes."""
import json
from root_accept_completed_preparations_v1 import ROOT, write_record
from root_accept_profile_v5_and_record_dispatch_blocks_v1 import block_record


def main():
    write_record("activity-stratified-metrics-root-engineering-v1", """# Root review: training activity strata reporting helper

Three focused CPU toy tests pass, including masked raw-cell pooling and equal
horizon/equal seed aggregation, training ties, empty groups, zero versus
unobserved training nodes and signed synthetic magnitudes. Root inspected the
pure implementation. The reporting rule is prospective and cannot fit thresholds
on test data. A real guarded caller must still bind the actual training slice,
node order, frozen strata file and reporting inputs. No actual strata fit or test
report ran. This engineering record grants no phase, inference or test authority.
""", ["edsgn/staging/activity_stratified_metrics_v1.py",
        "edsgn/tests/test_activity_stratified_metrics_v1.py",
        "logs/activity-stratified-metrics-v1-root-tests.log",
        "protocols/activity-strata-reporting-root-rule-v1.md"],
        "pure-reporting-engineering-only", actual_strata_fit=False, actual_test_report=False)
    block_record("typed-endpoint-producer-v2-root-blockers-v1", """# Root review: typed endpoint producer v2 remains blocked

Root inspected producer v2 and metadata validator v5. Their twenty-eight focused
toy tests do not exercise the public producer through a complete accepted
manifest, official predecessor validation, semantic receipts and actual output
serialization. The successor must add that full fixture integration.

Metadata permits same-seed same-role path sharing for matched H2 only, while H1
paired pre/post response requires one V4 trace reused on the two logical sides.
Cost-device inference depends on a domain field forbidden by the metadata schema
or a dataset-name suffix incompatible with the accepted domain source name; it
also treats cuda:0 differently from cuda. The device must be a separate
predeclared endpoint field and use the accepted canonical device convention.

Switch authority currently requires switch indices in the model-readable strict
history domain. Actual prepared domains deliberately omit oracle switches.
Keep oracle reporting authority separately hash-bound and decode it only after
all endpoint prediction/replay receipts are verified complete, then check the
first-changed mapping against the prospective reporting slice. Do not put oracle
switches into runtime input descriptors to make this producer pass.

No actual endpoint, test, oracle join, model, GPU or pipeline advancement ran.
""", ["edsgn/staging/typed_endpoint_producer_v2.py",
        "edsgn/staging/typed_endpoint_plan_validator_v5.py",
        "edsgn/tests/test_typed_endpoint_producer_v2.py",
        "reviews/typed-endpoint-producer-v2-writer-self-review.md"])
    block_record("verified-frozen-dispatch-v6-root-blockers-v1", """# Root review: dispatcher v6 needs final source/device association repair

The six planned pretest references and concrete side/seed, checkpoint, policy,
dataset, output and phase alias are now associated before dispatch. Twenty-six
CPU mocks pass and the H2 CLI matches its real full-policy driver. Root still
found that source matching accepts any declared plan producer source hash rather
than requiring the actual selected driver path/hash. The cost job also must
match the separate prospective CPU/CUDA endpoint device. Preserve v6 and use a
narrow successor with wrong-driver and wrong-device rejection before the child.

No actual dispatch, data, model, GPU, test or phase advancement ran.
""", ["edsgn/scripts/verified_frozen_dispatch_v6.py",
        "edsgn/tests/test_verified_frozen_dispatch_v6.py"])
    print(json.dumps({"pure_reporting_engineering_records": 1, "blocked_staging_records": 2,
                      "actual_experiment_run": False, "pipeline_advanced": False}))


if __name__ == "__main__":
    main()
