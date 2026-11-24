"""Root acceptance of narrow prepared prediction/family successors, no execution."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    expected = {
        'edsgn/staging/frozen_prediction_guarded_v5.py': '4798a08e59349adf6499d2352219b7721251c8a2beeb6b4803f2764116e216b4',
        'edsgn/staging/typed_primary_family_holm_v3.py': '68efecae5e1def030844fdd881f20e0509141fa9de2bb32d0a738a4c8bb8e3f2',
    }
    for rel, sha in expected.items():
        if digest(ROOT / rel) != sha:
            raise RuntimeError('Reviewed successor source changed: ' + rel)
    pred_log = 'logs/frozen-prediction-v5-root-cpu-v1.log'
    family_log = 'logs/typed-primary-family-holm-v3-root-cpu-v1.log'
    if '3 passed, 16 deselected' not in (ROOT / pred_log).read_text(encoding='utf-8-sig'):
        raise RuntimeError('Root prediction targeted evidence missing')
    if '12 passed' not in (ROOT / family_log).read_text(encoding='utf-8-sig'):
        raise RuntimeError('Root family evidence missing')
    write_record('frozen-prediction-guarded-root-acceptance-v5', '''# Root engineering acceptance: prediction bridge v5

Root inspected v4's full-array comparison, pre-loader test-access receipt,
source/data/checkpoint/freeze/policy/parameter fences, then the two-line v5
successor. Its first final live budget guard is followed by an output byte
fence; a second final guard accounts for that hashing time immediately before
completion. Root ran the three affected CPU budget/output tests: 3 passed,
16 deselected in5.91s. The writer's18 cases and independent final-expiry
regression also pass. Serialized predictions, targets, mask, origins and
model seed must match actual evaluator results exactly. Root blind-helper
acceptance records must retain their explicit science/pipeline/C07 false flags.

This is prepared engineering acceptance only. No actual final test, model
prediction, primary endpoint or phase advancement is authorized by this record.
Actual execution needs unchanged passed P3/P3A authorities, P4 running,
root-accepted pretest bindings, concrete verified-dispatcher jobs, both model
leases and the single cumulative72-hour stop-loss. Input archive suffix
materialization and test-origin evaluation remain separately reported.
''', ['edsgn/staging/frozen_prediction_guarded_v5.py',
      'edsgn/tests/test_frozen_prediction_guarded_v5.py',
      'edsgn/tests/test_frozen_prediction_guarded_v5_independent_review.py',
      'reviews/frozen-prediction-guarded-v4-independent-review-v1.md',
      'reviews/frozen-prediction-guarded-v4-independent-review-v1.json',
      'logs/frozen-prediction-guarded-v5-writer-tests.log',
      'logs/frozen-prediction-guarded-v5-independent-tests.log', pred_log],
      'engineering-staging-only', actual_test_prediction_run=False,
      c07_experiment_closed=False)
    write_record('typed-primary-family-holm-root-acceptance-v3', '''# Root engineering acceptance: pure typed family Holm kernel v3

Root checked explicit primary=True and exact family/claim/type agreement for
every registry row. It accepts exactly49 unique IDs partitioned9/16/24, checks
all numeric types, preserves whole-family multiplicity, and distinguishes
unidentifiable null-p evidence from identifiable negative results. A family
with an unidentifiable member remains inconclusive with no reduced-size Holm
decision. Identified superiority requires a negative estimate and upper95 CI
below zero; NI requires a positive accepted margin and one-sided95 upper bound
strictly below that margin, plus whole-family Holm. Root's12 pure CPU checks
include unequal hand-computed p-values and missing/false/cross-family fields.

This pure injected-array kernel cannot establish provenance, margin authority,
test chronology or scientific support. A separately accepted producer and
complete frozen49-row registry, raw reports and after-output input fences remain
required. No actual inference, hypothesis acceptance or phase advancement ran.
''', ['edsgn/staging/typed_primary_family_holm_v3.py',
      'edsgn/tests/test_typed_primary_family_holm_v3.py',
      'reviews/typed-primary-family-holm-v3-writer-evidence.md',
      'reviews/typed-primary-family-holm-v2-independent-review-v1.md',
      'reviews/typed-primary-family-holm-v2-independent-review-v1.json', family_log],
      'engineering-staging-only', actual_primary_inference_run=False)
    print(json.dumps({'root_acceptances': 2, 'actual_experiment_run': False,
                      'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
