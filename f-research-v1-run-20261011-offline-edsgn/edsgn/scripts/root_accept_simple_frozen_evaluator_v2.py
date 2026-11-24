"""Root engineering acceptance, without phase advancement or launch authority."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    pipeline_before = sha(ROOT/'pipeline-state.json')
    assert pipeline_before == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    review_path = ROOT/'reviews/simple-baseline-frozen-evaluation-v2-independent-v2.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    assert review['verdict'] == 'pass_staging_path_only_no_launch_authorization'
    assert review['real_data_fit_model_gpu_or_test_arrays_accessed'] is False
    assert review['launch_authorized'] is False
    expected = {
        'edsgn/staging/simple_baseline_frozen_evaluation_v2.py': '160f5a27903264374079928995a5de0da34f3050789937be5419f8291a706d1a',
        'edsgn/staging/test_simple_baseline_frozen_evaluation_v2.py': 'cbfecfc6eb2529eae1d80f2e67859d74096dcb649070974c780fb6c8531e5dc1',
        'edsgn/src/edsgn/engine.py': '491da67e3c23d412003b838e3b44d0c732e4378c1b23bf7b29494768b97febb3',
    }
    evidence = {}
    for rel, digest in expected.items():
        assert sha(ROOT/rel) == digest, rel
        evidence[rel] = digest
    for field in ('evaluator', 'tests', 'independent_test_log'):
        ref = review[field]
        assert sha(ROOT/ref['path']) == ref['sha256'].lower()
        evidence[ref['path']] = ref['sha256'].lower()
    for rel in ('reviews/simple-baseline-frozen-evaluation-v2-independent-v2.json',
                'reviews/simple-baseline-frozen-evaluation-v2-independent-v2.md',
                'reviews/simple-baseline-frozen-evaluation-v2-self-review.md',
                'logs/simple-baseline-frozen-evaluation-v2-tests.json',
                'logs/simple-baseline-frozen-evaluation-v2-root-focused-v1.log',
                'edsgn/staging/c06_domain_manifest_preflight_v4.py',
                'edsgn/staging/simple_baseline_freeze_v1.py', 'edsgn/src/edsgn/data.py'):
        evidence[rel] = sha(ROOT/rel)
    assert '13 passed' in (ROOT/review['independent_test_log']['path']).read_text(encoding='utf-8')
    assert '2 passed' in (ROOT/'logs/simple-baseline-frozen-evaluation-v2-root-focused-v1.log').read_text(encoding='utf-8')
    assert sha(ROOT/'pipeline-state.json') == pipeline_before
    record = {
        'scope': 'frozen_simple_no_refit_evaluator_local_engineering_only',
        'accepted_by': 'root', 'verdict': 'pass', 'evidence_sha256': evidence,
        'writer_cpu_tests_passed': 13, 'independent_cpu_tests_passed': 13,
        'root_focused_cpu_tests_passed': 2,
        'root_review': [
            'Original persistence and frozen linear prediction semantics; test route has no fitter.',
            'C06 descriptor is reconstructed from metadata-only current collection and exact strict domain.',
            'Owned guard, canonical P3/P3A evidence and official before-P4 validator precede loading.',
            'Full source/input/collection snapshot after final budget callback immediately precedes test loading.',
            'Complete ordered origins, array task/mask/node/normalization bindings and raw horizon metrics are verified.'
        ],
        'scientific_hypothesis_accepted': False, 'pipeline_advanced': False,
        'actual_test_authorized': False, 'actual_execution_completed': False,
        'limitations': [
            'Current-source train-only reconstruction and root acceptance of all four preserved legacy fits are pending.',
            'Managed dispatcher, stable phase aliases, global simple driver contract and whole-campaign closure are pending.',
            'Production integration must bind and recheck final output/source/phase closure before a completed child receipt.'
        ]
    }
    output = ROOT/'reviews/simple-baseline-frozen-evaluation-v2-root-engineering-v1.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'pass_engineering_only', 'path': str(output), 'sha256': sha(output)}))

if __name__ == '__main__':
    main()
