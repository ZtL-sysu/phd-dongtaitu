"""Root source/toy acceptance of the final calibration lifecycle."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    source = 'edsgn/staging/frozen_validation_calibration_v7.py'
    expected = 'd0212737ed0f02e6f9342f788efda73c8c3bba53d12662f15e734b96cb6d2d9e'
    review = json.loads((ROOT / 'reviews/frozen-validation-calibration-independent-v7.json').read_text(encoding='utf8'))
    if digest(ROOT / source) != expected or review.get('source_sha256', '').lower() != expected:
        raise RuntimeError('Reviewed calibration bytes changed')
    if review.get('verdict') != 'engineering-only-pass':
        raise RuntimeError('Independent calibration verdict not passed')
    write_record('frozen-validation-calibration-root-acceptance-v7', '''# Root acceptance: frozen validation calibration v7

Root reviewed the complete lifecycle. The original calibrate.calibrate_run,
actual ForecastModel/PreparedData batching, at-launch binder and full validation
origin/target checks remain used. Only resource lease/receipt services are mocked
in the actual small CPU end-to-end fixture. Independent combined tests pass
twenty cases (eleven calibration and nine dispatcher fixtures). In particular,
the final live budget guard now runs inside the finalizer after the caller's
last full input-hash pass, immediately before a successful phase receipt.
Budget exhaustion there writes failed, releases leases and raises; final hash
mismatch likewise cannot return successful completion. Earlier v5/v6 failures
remain preserved. No real calibration ran.

Chronological combined NPZ loading is explicitly disclosed as materialization
of the suffix. Test origins, targets and feedback are never requested by the
calibration callback; test support is metadata-only. Synthetic main-training
statistics remain used for all validation trajectories. Passed P2/P3 artifact
verification is a managed-controller responsibility in addition to local P3A
evidence and canonical fences; standalone scientific launch is not accepted.

This accepts isolated source/toy engineering only. Actual P3/P3A/P4 gates,
completed trained checkpoints, root-accepted dataset-specific epsilon rationale
and baseline evidence, queue/source/data/input bindings, both child leases and
the single72-hour ledger remain required. The main-calendar email epsilon stays
unidentifiable; episode/MOOC decisions cannot repair it. This neither selects a
real policy nor passes a phase or hypothesis. A managed concrete launch still
requires dispatcher acceptance and root review.
''', [source, 'edsgn/tests/test_frozen_validation_calibration_v7.py',
      'reviews/frozen-validation-calibration-independent-v7.md',
      'reviews/frozen-validation-calibration-independent-v7.json',
      'reviews/frozen-validation-calibration-v7-self-review.md',
      'reviews/frozen-validation-calibration-v7-verdict.json'],
                 'engineering-staging-only', actual_calibration_run=False,
                 real_policy_selected=False)
    print(json.dumps({'root_acceptance': 'calibration-v7', 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
