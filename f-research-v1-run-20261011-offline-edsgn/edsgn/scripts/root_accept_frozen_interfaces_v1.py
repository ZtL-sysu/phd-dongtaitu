"""Root's source/toy interface acceptances; no model launch or phase mutation."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    pins = {
        'edsgn/staging/global_graph_oracle_report_v1.py': '41bbf29335820cc6359f32e169fb4434946b54fa01e7da5a0aaf70f876bb34bc',
        'edsgn/staging/test_global_graph_oracle_report_v1.py': '7e73aae2d7be8e1a817f772aa7f16f297665da127243f9bc4d15a9b6b77cdb2e',
        'edsgn/staging/global_frozen_test_completion_v1.py': '59b3d529efa76dc475448b3eccf88529a5ab12cc7ade53913f66d46b7a3850af',
        'edsgn/staging/ordinary_frozen_test_admission_v1.py': '7e265b84d33a98c16032ed152ab6f3668fd1d6ddb7a6d0193ab5de0eada24b48',
        'edsgn/tests/test_ordinary_frozen_test_admission_v1.py': 'fcb99304d03db80ccaa6758c517b4332a5c97e0b3f8bb08290870609905f6457',
        'edsgn/scripts/verified_frozen_dispatch_v9.py': '34ab18bc9f5da5134615e204af314828c444ef041d0efa85847f5b936b039a19',
        'edsgn/tests/test_verified_frozen_dispatch_v9.py': '1085213be6ec523d15ea93140a47edcfbde61ebcaeba5a26651fc0477677940a',
        'edsgn/scripts/verified_frozen_dispatch_v8.py': 'b946169932699f5efad8ae187e5c42cf2e247ccf0c97c895c9103c21d7748998',
    }
    assert digest(ROOT / 'pipeline-state.json') == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    for rel, expected in pins.items():
        assert digest(ROOT / rel) == expected, rel
    for rel, result in [('logs/root-global-graph-oracle-reporter-v2-tests.log', '10 passed'),
                        ('logs/root-ordinary-admission-dispatch-v1-tests.log', '30 passed'),
                        ('logs/root-ordinary-dispatch-v2-tests.log', '17 passed')]:
        content = (ROOT / rel).read_text(encoding='utf-8-sig')
        assert result in content and 'failed' not in content.lower(), rel
    oracle = write_record('global-graph-oracle-reporter-root-engineering-v1', '''# Root engineering acceptance: post-completion graph oracle reporting

Root reviewed the source and the independent v3 review with its v4 type repair,
then reran the ten final CPU toy fixtures. They pass. The accepted global
completion verifier is called before any prediction, trajectory or oracle NPZ
decode; reporter/authority/collection membership and input bytes are fenced
before and after the join. The NPY header supplies raw [T,N,F]; hmax and requested
forecast horizons are checked separately from feature count. Trajectory seed
IDs require exact nonnegative integers. Exact full origins and compact history
trace indices are checked, while support alignment remains explicitly sampled.
The graph7 abrupt/gradual formula and directed positive oracle support are
preserved. Serialized JSON/gzip bytes and output membership are rechecked.

This accepts the reporting interface only. No actual global test completion,
oracle authority, replay, structural recovery, hypothesis or inference is
accepted. Actual oracle access remains prohibited until every job in the
complete frozen global queue has an independently accepted completion. The
initial root test command used the wrong test directory and ran no tests;
its preserved log is not passing evidence. The corrected final command passed.
''', [*list(pins)[:3], 'protocols/global-graph-oracle-report-authority-root-design-v1.md',
      'reviews/global-graph-oracle-reporter-independent-v3.md',
      'reviews/global-graph-oracle-reporter-independent-v3.json',
      'reviews/global-graph-oracle-reporter-independent-v4.md',
      'reviews/global-graph-oracle-reporter-independent-v4.json',
      'logs/root-global-graph-oracle-reporter-v2-tests.log'],
      'post_global_completion_oracle_reporting_interface_only',
      actual_global_completion_accepted=False, actual_oracle_accessed=False,
      actual_test_authorized=False)
    ordinary = write_record('ordinary-frozen-admission-dispatch-root-engineering-v1', '''# Root engineering acceptance: ordinary frozen test admission and dispatcher9

Root reviewed exact static config/blueprint/actual-launch correspondence,
original training phase output ownership and root numeric acceptance, current
job output/alias absence, serial prior-job preservation, and consumed-byte
linkage to the acyclic global queue. Byte-identical replay copies preserve the
original training checkpoint and require a separate adjacent immutable freeze;
both source and execution bytes remain bound. The independent helper review
passes. Root's combined helper/dispatcher run passed30 fixtures, then root
reran all17 dispatcher fixtures after the prelaunch repair. New source content
or directory membership introduced during the official validator is rejected
before the child starts; final input/state/output/receipt alias fences also
remain active. The child in these fixtures is mocked, so these counts are
engineering evidence rather than experiment evidence.

Acceptance is limited to the current static and singleton selected-candidate
ordinary interface. The helper's byte-identical route currently requires a
static configuration; multiple-candidate ordinary selection is not certified
by this record. The actual complete matrix must conform to that restriction
or receive a separately reviewed extension before P3A. Primary49 jobs retain
dispatcher8/producer5/validator8. Dispatcher9 is used for separately admitted
ordinary or secondary jobs. Actual queues, all required rows, resources,
freezes, five-seed results, test replay and scientific conclusions remain
pending. No gate or accepted source8 was changed.
''', [*list(pins)[3:], 'edsgn/staging/global_frozen_test_completion_v1.py',
      'edsgn/staging/prospective_training_candidate_binding_v1.py',
      'protocols/ordinary-frozen-test-job-admission-root-design-v1.md',
      'protocols/byte-identical-checkpoint-replay-root-design-v1.md',
      'reviews/ordinary-admission-independent-v1.md',
      'reviews/ordinary-admission-independent-v1.json',
      'reviews/dispatcher9-ordinary-admission-v2.md',
      'logs/dispatcher9-ordinary-admission-v3-tests.log',
      'logs/root-ordinary-admission-dispatch-v1-tests.log',
      'logs/root-ordinary-dispatch-v2-tests.log'],
      'ordinary_static_or_singleton_admission_dispatch_interface_only',
      actual_matrix_complete=False, actual_test_authorized=False,
      multiple_candidate_ordinary_linkage_accepted=False)
    print(json.dumps({'status': 'pass_scoped', 'scopes': [oracle['scope'], ordinary['scope']]}))


if __name__ == '__main__':
    main()
