"""Root's source/toy acceptance records; no model run or pipeline mutation."""
import json
from pathlib import Path

from root_accept_completed_preparations_v1 import digest, write_record

ROOT = Path(__file__).resolve().parents[2]


def main():
    expected = {
        'edsgn/staging/frozen_graph_state_diagnostic_v6.py':
            '7c91a01f237d64c7f2db383fc31a823265edd7e42b4daa456744e755716302a7',
        'edsgn/staging/frozen_h2_matched_retention_v4.py':
            '6c4ff121436990b8aaf97993525ed9d995f2ded40e252022c78518a49643c578',
        'edsgn/staging/frozen_split_history_robustness_v4.py':
            '15f8996b90e1bb6c01d139e6b9484b1d6272e4cade67b373c37446f97e08c107',
        'edsgn/staging/c06_domain_manifest_preflight_v3.py':
            'ee1b093427407ded224db59d9a88d32f530ee6a1178db44e6ad82bdc252c2c86',
        'edsgn/staging/paired_scalar_bootstrap_v1.py':
            'ad2ae2683f8b0df245e1ca60700e58cf62c1746b6599c1c1ed690b873cdaccf6',
    }
    for rel, sha in expected.items():
        if digest(ROOT / rel) != sha:
            raise RuntimeError(f'Reviewed source changed: {rel}')
    shared = json.loads((ROOT / 'reviews/frozen-driver-successors-independent-v2.json').read_text(encoding='utf8'))
    for name in ('frozen_graph_state_diagnostic_v6', 'frozen_h2_matched_retention_v4'):
        if shared['verdicts'][name]['verdict'] != 'staging-engineering-acceptable':
            raise RuntimeError(f'Independent engineering verdict not acceptable: {name}')
    c06 = json.loads((ROOT / 'reviews/frozen-c06-v4-independent-v1.json').read_text(encoding='utf8'))
    if c06['verdict'] != 'pass_for_staging_engineering_only':
        raise RuntimeError('C06 independent verdict not acceptable')
    scalar = json.loads((ROOT / 'reviews/paired-scalar-bootstrap-independent-v1.json').read_text(encoding='utf8'))
    if not scalar['verdict'].startswith('engineering-only-pass-for-kernel'):
        raise RuntimeError('Scalar independent verdict not acceptable')

    common = '''
This accepts source/toy engineering only. It does not pass a canonical stage,
close a real experimental correction, accept any hypothesis, select a policy,
authorize a holdout redesign, or establish acceleration. Before an actual
frozen launch, root must accept the managed dispatcher and its concrete job:
all passed predecessor input/evidence bytes, canonical state, source/data,
checkpoint, freeze and complete memberships must be validated before and
after the child. Both child leases and the live single72-hour guard remain
required. No actual scientific replay was performed for this acceptance.
'''
    write_record('frozen-graph-state-diagnostic-root-acceptance-v6', '''# Root acceptance: graph/state diagnostic v6

Root reviewed the actual task/payload schema, main-training normalization,
full-origin trace route, self-inclusive artifact byte cap and final gate/source/
data/freeze fences. The independent focused suite passes three tests, including
the real two-origin CPU frozen-model path, after-report P3A mutation refusal
and postserialization budget failure. Both report completion and the outer
phase receipt have the live budget callback. N64/N256 metadata-only cap plans
are prospective individual-run bounds; actual runtime, memory, output size and
campaign disk remain unmeasured. Explicit cap16/64GiB must be frozen and disk
rechecked at each serial job.
''' + common, ['edsgn/staging/frozen_graph_state_diagnostic_v6.py',
              'edsgn/tests/test_frozen_graph_state_diagnostic_v6.py',
              'reviews/frozen-driver-successors-independent-v2.md',
              'reviews/frozen-driver-successors-independent-v2.json',
              'logs/frozen-graph-state-diagnostic-v6-focused-tests.log',
              'logs/frozen-graph-state-diagnostic-v6-cap-preflight.json'],
                 'engineering-staging-only', actual_diagnostic_run=False)

    write_record('frozen-h2-matched-retention-root-acceptance-v4', '''# Root acceptance: H2 matched-retention driver v4

Root reviewed checkpoint task fields, actual PreparedData origin enumeration,
synthetic main-normalizer use, equal non-synthetic checkpoint statistics,
prescribed teacher graph/action replay and informative-teacher parity. The
independent suite passes twelve CPU tests and a separate post-report canonical
mutation check. The full two-origin model route runs the real collection/kernel;
resource lease/receipt services have separate mock coverage. This evidence does
not imply a complete guarded real collection replay. Final serialization,
integrity fences and outer receipt are covered by live budget checks. Scorer
and teacher replay overhead is diagnostic and cannot count as ordinary speed.
''' + common, ['edsgn/staging/frozen_h2_matched_retention_v4.py',
              'edsgn/tests/test_frozen_h2_matched_retention_v4.py',
              'edsgn/staging/h2_matched_graph_retention_v2.py',
              'reviews/frozen-driver-successors-independent-v2.md',
              'reviews/frozen-driver-successors-independent-v2.json'],
                 'engineering-staging-only', actual_matched_graph_replay_run=False)

    write_record('frozen-c06-driver-root-acceptance-v4', '''# Root acceptance: C06 driver v4 and metadata preflight v3

Root reviewed the changed lifecycle against the independent eight-test CPU
review: the same cumulative guard runs after completed origins immediately
before output creation, and after serialization plus final fences before the
completed receipt. Source, data, domain, normalizer, checkpoint, freeze, policy,
parameters and canonical bytes remain bound. The unchanged v1 domain schema
is compatible with existing metadata-only domains, but old v1 descriptors do
not authorize v4: successor descriptors binding v3/v4 bytes are required before
the actual freeze. Positive missing-block coverage exists only as preparation;
C06 stays open until the complete real replay and its acceptance.
''' + common, ['edsgn/staging/frozen_split_history_robustness_v4.py',
              'edsgn/staging/c06_domain_manifest_preflight_v3.py',
              'edsgn/tests/test_frozen_split_history_robustness_v4.py',
              'reviews/frozen-c06-v4-independent-v1.md',
              'reviews/frozen-c06-v4-independent-v1.json'],
                 'engineering-staging-only', actual_robustness_replay_run=False,
                 c06_experiment_closed=False)

    write_record('paired-scalar-bootstrap-root-acceptance-v1', '''# Root acceptance: paired scalar numerical kernel v1

Root reviewed shape/support checks, seed-pooled sum/count estimator, equal
five-seed weights, shared v4 temporal draw schedule and both centered p-value
directions. The independent review passes nine tests and independently rebuilds
the 2000-draw schedule and weighted estimator on a two-trajectory CPU array.
The one-sided test is H0:D>=margin, D=left-right; its percentile upper bound and
family-level decision remain separate requirements. The kernel assumes callers
bind row identity to seeds0..4 and rejects zero support. It is not a source-
bound scientific producer, and cannot turn a graph/cost array into a claim.

No actual array endpoint, family inference, test outcome or phase ran. Gated
producers, accepted raw-unit margins/blocks, complete endpoint registry and
whole-family Holm remain required.
''', ['edsgn/staging/paired_scalar_bootstrap_v1.py',
      'edsgn/tests/test_paired_scalar_bootstrap_v1.py',
      'edsgn/staging/paired_statistics_shared_v4.py',
      'protocols/non-mae-endpoint-kernel-specification-v1.md',
      'reviews/paired-scalar-bootstrap-independent-v1.md',
      'reviews/paired-scalar-bootstrap-independent-v1.json'],
                 'engineering-staging-only', actual_inference_run=False)
    print(json.dumps({'root_acceptances': 4, 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
