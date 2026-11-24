"""Accept reviewed metadata/conversion preparations; never perform inference."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    sources = {
        'edsgn/staging/typed_endpoint_plan_validator_v4.py': '9d099359bbbc4ea3bd97fbd015fbe9c7ddd95f79523d5fb0a278ef3154b4e032',
        'edsgn/staging/endpoint_sufficient_statistics_v4.py': 'afbbd2f459cc680f2689334fe69decc63f87d37308cbbe6956c62c3f994a2519',
    }
    for rel, sha in sources.items():
        if digest(ROOT / rel) != sha:
            raise RuntimeError('Reviewed source changed: ' + rel)
    log = 'logs/typed-metadata-converter-root-cpu-v1.log'
    if '7 passed, 22 deselected' not in (ROOT / log).read_text(encoding='utf-8-sig'):
        raise RuntimeError('Root narrow verification evidence missing')
    write_record('typed-endpoint-plan-validator-root-acceptance-v4', '''# Root engineering acceptance: typed endpoint metadata validator v4

Root inspected the strict plan, input-only ten method/seed pretest rows,
separate postrun manifest, exact49 prospective registry entries and final
consumed-byte barriers. Every domain trajectory must supply complete legal
origins; full post-switch and both4L bounds are checked without truncation.
Blocks use positive validation dependence and max(L,H,dependence). Positive
accepted raw-unit epsilon is distinguished from an explicit unidentifiable
rule. Dispatch BEFORE snapshots must bind the pretest execution-binding SHA,
and completed original/alias receipts must be byte-identical and match the
dispatch child's output/job/phase. Prospective plans do not invent future
model or output hashes. Root's focused metadata/converter run passed7 tests;
the independent validator review passed14 author tests plus two mutations.

This validates paths, hashes and metadata only. It does not validate prediction
contents, teacher parity, timing semantics, actual model provenance or outcomes.
Those must be established by the producer before numeric inference. No real
pretest plan, frozen registry, P3A/P4 transition or final-test access was accepted.
''', ['edsgn/staging/typed_endpoint_plan_validator_v4.py',
      'edsgn/tests/test_typed_endpoint_plan_validator_v4.py',
      'reviews/typed-endpoint-plan-validator-v4-self-review.md',
      'reviews/typed-endpoint-plan-validator-v4-independent-review-v1.md',
      'reviews/typed-endpoint-plan-validator-v4-independent-review-v1.json', log,
      'protocols/prospective-and-post-training-binding-root-v1.md'],
      'engineering-staging-only', actual_endpoint_metadata_acceptance=False,
      actual_inference_run=False)
    write_record('endpoint-sufficient-statistics-root-acceptance-v4', '''# Root engineering acceptance: pure sufficient-statistic converter v4

Root reviewed v3's independently checked complete-trajectory/full-post and4L
support repair, then v4's narrow strict timing-scalar fix. Boolean, strings,
null and complex objects are rejected before conversion; finite positive
numeric values retain median-of-five per-seed/origin semantics. Forecast
statistics pool masked raw absolute errors/counts at required offsets. Streamed
v7 graph records use O(origins+L*N) state, validate complete ordered steps/nodes,
exclude initial construction while retaining initialized zero-TV observations,
and require equal pre/post counts for negative-TV response contrasts. No
unsupported trajectory or temporal interval is dropped. Root's seven focused
CPU checks include the fixed timing mutations and pretest metadata lifecycle.

This is pure numerical engineering, without file access, provenance authority,
margin selection, gating, primary p-values or scientific claims. Future callers
must independently verify bound source/data/model/phase/dispatch evidence and
all actual arrays before and after conversion. No real test or phase ran here.
''', ['edsgn/staging/endpoint_sufficient_statistics_v4.py',
      'edsgn/tests/test_endpoint_sufficient_statistics_v4.py',
      'reviews/endpoint-sufficient-statistics-v4-writer-evidence.md',
      'reviews/endpoint-sufficient-statistics-v3-independent-review-v1.md',
      'reviews/endpoint-sufficient-statistics-v3-independent-review-v1.json', log],
      'engineering-staging-only', actual_endpoint_conversion_run=False)
    print(json.dumps({'root_acceptances': 2, 'actual_inference_run': False,
                      'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
