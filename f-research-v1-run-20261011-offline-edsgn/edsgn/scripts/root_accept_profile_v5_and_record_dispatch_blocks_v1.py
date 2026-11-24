"""Root's source-reviewed engineering decisions; never execute experiments."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def block_record(stem, body, files):
    record = {
        'reviewer': 'root', 'verdict': 'blocked',
        'scope': 'engineering-staging-only', 'actual_experiment_run': False,
        'pipeline_advanced': False, 'scientific_hypothesis_accepted': False,
        'evidence_sha256': {rel: digest(ROOT / rel) for rel in files},
    }
    base = ROOT / 'reviews' / stem
    if base.with_suffix('.json').exists():
        if json.loads(base.with_suffix('.json').read_text(encoding='utf-8')) != record:
            raise RuntimeError('Existing blocked review has different bindings')
        return
    with base.with_suffix('.md').open('x', encoding='utf-8', newline='\n') as out:
        out.write(body.rstrip() + '\n')
    with base.with_suffix('.json').open('x', encoding='utf-8', newline='\n') as out:
        json.dump(record, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def main():
    expected = {
        'edsgn/staging/frozen_profile_guarded_v5.py': '4e873778a0ae8164bbc2e45cab0b42c753ed0e1dc20f7c1f68e54c931a138b6c',
        'edsgn/scripts/verified_frozen_dispatch_v5.py': '6c2b83797d1571b534db45a1ed9441bd6cddd8762a727c4eda4bd720d012486b',
        'edsgn/staging/typed_endpoint_producer_v1.py': 'eb7787dc052080b3efbffcfb43f30caa8207b80e6dc1821f8789747157499b70',
    }
    for rel, sha in expected.items():
        if digest(ROOT / rel) != sha:
            raise RuntimeError('Reviewed source changed: ' + rel)
    independent = 'reviews/frozen-profile-guarded-v5-independent-review-v1.json'
    audit = json.loads((ROOT / independent).read_text(encoding='utf-8'))
    # The independently written review is human-readable source/test evidence;
    # the exact bytes are bound below rather than asserting an invented schema.
    if not audit:
        raise RuntimeError('Independent profile review is empty')
    write_record('frozen-profile-guarded-root-acceptance-v5', '''# Root engineering acceptance: frozen profiler v5

Root inspected the complete v5 wrapper, original profile runtime and its CPU
fixtures. The independent review reports eleven passing CPU toy tests. The
wrapper binds the checkpoint's main training collection before numeric loading,
checks its exact training statistics, normalizes independent synthetic test or
validation trajectories with those statistics, and binds the normalizer source.
Its timer starts before CPU checkpoint metadata loading. Combined-file test
suffix materialization is distinguished from entering a test-origin replay and
successfully completing the entire replay. Partial replay failure remains
started=true with completion=false; false evaluated is not evidence of zero
partial work.

The original scientific measurements remain B1/FP32, warmup10, five full ordered
replays, synchronized forward plus transfer samples. Guard overhead is a separate
sidecar and does not change model-forward samples. Full canonical origin order is
checked for every repeat. Periodic/random controls apply and revalidate the
source-bound accepted blind scheduling helper. Main training bytes, checkpoint,
freeze, policy, current source tree, predecessor evidence and output bytes are
checked again after the final official gate. The final live aggregate-budget
guard precedes completion; final receipt serialization has the small ordinary
accounting tail recorded by independent review. Both exclusive leases are
released in finally.

This is gated engineering preparation only. No actual profile, final test,
speedup, hypothesis outcome or C07 experiment closure ran. Actual execution still
requires passed P3/P3A, P4 running, root-accepted input-only pretest bindings and
a concrete source-bound dispatcher job. Earlier blocked sources are preserved.
''', ['edsgn/staging/frozen_profile_guarded_v5.py',
      'edsgn/tests/test_frozen_profile_guarded_v5.py',
      'edsgn/src/edsgn/profile.py',
      'reviews/frozen-profile-guarded-v5-self-review.md',
      'reviews/frozen-profile-guarded-v5-independent-review-v1.md', independent,
      'logs/frozen-profile-guarded-v5-tests.log'],
      'engineering-staging-only', actual_profile_run=False,
      actual_final_test_run=False, c07_experiment_closed=False)
    block_record('verified-frozen-dispatch-v5-root-blockers-v1', '''# Root source review: dispatcher v5 blocked

The repaired actual calibration-dependence sidecar and nested reference hashes
are useful, and sixteen mock tests pass. Root nevertheless found that the six
pretest references are not compared with the P3A plan's planned_paths. The
actual job checkpoint, input binding, policy/freeze, dataset, output and receipt
alias are not associated with a bound method/seed and its prospective paths.
An unrelated accepted endpoint binding can therefore accompany unrelated argv.
ACF/epsilon authority strings must also resolve to a hash-bound authority.
Existing tests use empty planned_paths and unrelated argv, so their positive
case does not prove this association. Preserve v5 and repair a v6 successor
with actual schema-shaped fixtures and rejection before the child is called.

No actual job was dispatched, no test was accessed, and no pipeline gate changed.
''', ['edsgn/scripts/verified_frozen_dispatch_v5.py',
      'edsgn/tests/test_verified_frozen_dispatch_v5.py',
      'edsgn/staging/typed_endpoint_plan_validator_v4.py',
      'reviews/verified-frozen-dispatch-v5-writer-evidence.md'])
    block_record('typed-endpoint-producer-v1-root-blockers-v1', '''# Root source review: raw endpoint producer v1 blocked

The five toy tests do not exercise a complete accepted manifest. Actual typed
schema3 run_artifacts contains postrun references only; checkpoint/freeze and
other input references are in schema2 pretest-binding runs. The producer reads
checkpoint/freeze from the wrong map. It selects matched H2 replay by claim H2,
which wrongly includes the independently trained V4-versus-V3 comparison. It
still requires dispatcher4/profile4 source mappings and does not certify the
selected matched-arm archive against actual report prediction_outputs, the
complete graph-action proof or equal device/checkpoint identity for H3.

Repair a producer v2 and minimal metadata-validator successor with exact actual
report formats and full accepted-manifest integration. Hash-consume the matched
graph proof. Same-seed sides may share the same evidence role for one actual
replay; this is evidence reuse, not a new model replica. Different-seed or
different-role path collisions remain errors. In matched comparisons the two
prediction archives must differ and select the prescribed informative/control
arms. H1 response uses paired pre/post offsets from one full graph trace.

No actual primary endpoint, arrays, model, test or pipeline advancement ran.
''', ['edsgn/staging/typed_endpoint_producer_v1.py',
      'edsgn/staging/typed_endpoint_plan_validator_v4.py',
      'edsgn/staging/frozen_h2_matched_retention_v4.py'])
    print(json.dumps({'profile_engineering_acceptances': 1,
                      'blocked_successors_preserved': 2,
                      'actual_experiment_run': False, 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
