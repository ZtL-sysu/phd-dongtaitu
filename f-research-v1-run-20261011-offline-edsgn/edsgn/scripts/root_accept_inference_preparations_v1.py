"""Root reviewed staging preparation records; never execute a scientific job."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    expected = {
        'edsgn/scripts/verified_frozen_dispatch_v3.py': '957fb75244125c32db5c667e9ea563d63ecaa024a6642e47ec1bc88a3d2ec579',
        'edsgn/staging/blind_reference_cost_policy_v1.py': '2500f9687aa277ad5e67f3d4d41dde095d2e51761109da2f9dfceb6a68e3218d',
        'edsgn/staging/paired_forecast_bootstrap_v1.py': '260a83f91f6e46dfeefd9814746b0c265095ee1565d1fb4b5d7a0417413e0d80',
    }
    for rel, sha in expected.items():
        if digest(ROOT / rel) != sha:
            raise RuntimeError(f'Reviewed bytes changed: {rel}')
    write_record('verified-frozen-dispatch-root-acceptance-v3', '''# Root independent acceptance: managed frozen dispatcher v3

Root independently inspected v3 against the previously independent v2 review:
all four exact driver argument schemas allow their documented optional flags,
and C06 now must bind every principal artifact in the output digest map.
Principal artifact and child receipt hashes are rechecked after the final
official validator and after the final input snapshot. Root ran the two changed-
branch CPU mock tests: 2 passed,9 deselected in1.41s. The writer's complete suite
passes eleven mocks. Passed predecessor contract/input/evidence hashes and
source/data directory membership are bound before and after child execution.
The copied official before-P4 validator is pinned; P4 running is required.

This accepts engineering only. The parent uses fixed Conda/shell=False and
acquires no model leases, avoiding nested-lease deadlock; the child owns its
phase timer and both leases. A zero exit without required artifacts and a
matching new completed phase receipt is refused. Parent orchestration is not
counted as a second training/profile phase. Principal-artifact checks do not
replace scientific review of all diagnostic/statistical artifacts. There is no
production job or actual child execution yet. Concrete source-bound job plans,
phase gates, freeze extras, resource/budget checks and root review remain needed.
The dispatcher does not itself install C07's blind control procedure.
''', ['edsgn/scripts/verified_frozen_dispatch_v3.py',
      'edsgn/tests/test_verified_frozen_dispatch_v3.py',
      'reviews/verified-frozen-dispatch-independent-v2.md',
      'reviews/verified-frozen-dispatch-independent-v2.json',
      'reviews/verified-frozen-dispatch-v3-writer-review.md',
      'reviews/verified-frozen-dispatch-v3-writer-review.json',
      'protocols/verified-pipeline-validator-source-v1.json'],
                 'engineering-staging-only', actual_child_run=False,
                 parent_acquires_model_leases=False)

    write_record('blind-reference-cost-policy-root-acceptance-v1', '''# Root acceptance: C07 blind reference cost helper v1

Root reviewed both instance-bound methods. The selector recorder delegates to
the existing periodic/random selection and records only action step; its cost
fitter ignores q/residual/quality scores, preserves offered nodes and mandatory
work, merges normalized offered rank/quota with deterministic ties, and uses
the same grouped incremental table plus base-subtracted B1 envelope. Independent
eight-test CPU review passes, including score-swap invariance, original-fitter
confound, periodic rotation, device-local seeded random priorities, mandatory
overrun and unchanged policy dataclass digest. Full/adaptive policy behavior
is untouched. Per-device random realization may differ; replay within a device
must remain exact and comparative timing uses that device's same realization.

The unchanged dataclass digest does not identify this procedure: source and
rule version must be bound in freeze extras, checked before/after execution,
and reported by every affected prediction, robustness and profile route. No
actual freeze or reference outcome has run, and integration remains pending.
C07 is not scientifically closed by helper acceptance. This is an engineering
control-isolation correction, not a searched hyperparameter or SNN replacement.
''', ['edsgn/staging/blind_reference_cost_policy_v1.py',
      'edsgn/tests/test_blind_reference_cost_policy_v1.py',
      'reviews/blind-reference-cost-policy-independent-v1.md',
      'reviews/blind-reference-cost-policy-independent-v1.json',
      'protocols/c07-blind-reference-cost-fitting-correction-v1.md'],
                 'engineering-staging-only', actual_blind_control_run=False,
                 c07_experiment_closed=False)

    write_record('paired-forecast-bootstrap-root-acceptance-v1', '''# Root acceptance: required-offset pooled MAE numerical kernel v1

Root reviewed the estimator and NI direction. Raw valid-cell sums/counts pool
over origins per seed and required offset, then offset MAEs and five seeds
receive equal weights. The scalar/shared-v4 schedule is reused exactly. The
independent CPU toy comparison to the original pooled-MAE routine matches point
estimate and95%CI to floating precision, matches two-sided p exactly and matches
draw SHA. The writer's nine tests cover unequal counts/horizons, margins, bad
support and missing seeds. NI uses H0:D>=epsilon with centered lower-tail p and
one-sided95 upper bound; margin acceptance and whole-family Holm are external.

This is pure-array engineering evidence, not a source/freeze-bound endpoint
producer. Caller-bound ordered seed IDs, required offsets, complete legal
origins/targets/masks, accepted epsilon/block and the whole primary registry
remain required. No actual inference, test array or scientific phase ran.
''', ['edsgn/staging/paired_forecast_bootstrap_v1.py',
      'edsgn/tests/test_paired_forecast_bootstrap_v1.py',
      'reviews/paired-forecast-bootstrap-independent-v1.md',
      'reviews/paired-forecast-bootstrap-independent-v1.json',
      'reviews/paired-forecast-bootstrap-v1-self-review.md',
      'reviews/paired-forecast-bootstrap-v1-verdict.json'],
                 'engineering-staging-only', actual_inference_run=False)
    print(json.dumps({'root_acceptances': 3, 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
