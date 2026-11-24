"""Accept reviewed calibration chronology engineering, without running calibration."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    source = 'edsgn/staging/frozen_validation_calibration_v9.py'
    expected = '975505e0bf7cbfc80392d0c8259b7c8a1f79de2b67284c32c63cf6f6dfb07554'
    if digest(ROOT / source) != expected:
        raise RuntimeError('Reviewed calibration v9 bytes changed')
    log = 'logs/calibration-v9-root-cpu-v1.log'
    if '5 passed, 14 deselected' not in (ROOT / log).read_text(encoding='utf-8-sig'):
        raise RuntimeError('Root concrete calibration branch evidence missing')
    text = '''# Root acceptance: validation calibration chronology v9

Root inspected the v7-to-v9 change and the independent source review. The
immutable P3A validation-only epsilon authority binds the exact candidate plan
and baseline evidence; it does not invent future P4 checkpoint hashes. A
separate root-accepted post-training launch binding pins the actual checkpoint,
config, training-input binding, candidate IDs/queue and unchanged authority.
Accepted primary epsilon must be finite and positive. The declared all-zero
validation route requires all valid targets to be zero, keeps primary epsilon
null, uses a separately labeled descriptive zero fallback and fixed s00, and
does not authorize noninferiority. No margin is borrowed across datasets.

Root ran five focused CPU tests using the original calibration engine, including
successful accepted/unidentifiable cases, finalization failures and required
P3A queue binding: 5 passed, 14 deselected in 10.58s. The writer's full suite
passes 19 tests. The earlier accepted v7 input/parameter/source finalizer,
leases, full validation origins and final cumulative-budget fence remain.
Combined-bundle numeric materialization is disclosed separately from evaluating
test origins or using test feedback.

This accepts engineering only. No real calibration, selected policy, epsilon
decision, block estimate, frozen checkpoint, test result or phase advancement
is accepted here. Operational launch still needs actual P3/P3A completion,
concrete byte-bound authorities and launch records, P4 running, a verified
dispatcher job and resource/budget clearance.
'''
    write_record('frozen-validation-calibration-root-acceptance-v9', text,
        [source, 'edsgn/tests/test_frozen_validation_calibration_v9.py',
         'reviews/calibration-chronology-v9-writer-evidence.md',
         'reviews/frozen-prediction-guarded-v4-independent-review-v1.md',
         'reviews/frozen-prediction-guarded-v4-independent-review-v1.json', log,
         'protocols/prospective-and-post-training-binding-root-v1.md'],
        'engineering-staging-only', actual_calibration_run=False,
        actual_epsilon_accepted=False, actual_policy_selected=False)
    print(json.dumps({'root_acceptances': 1, 'actual_calibration_run': False,
                      'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
