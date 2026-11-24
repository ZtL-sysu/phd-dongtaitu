"""Accept matched-loss controller engineering, without training authorization."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def read(rel):
    return json.loads((ROOT/rel).read_text(encoding='utf-8'))

class Normalize(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace('count_loss_training_controller_v5', 'count_loss_training_controller_v4')
        return node

def changed_functions():
    versions = []
    for version in (4, 5):
        tree = ast.parse((ROOT/f'edsgn/scripts/count_loss_training_controller_v{version}.py').read_text(encoding='utf-8'))
        tree = Normalize().visit(tree)
        versions.append({node.name: ast.dump(node, include_attributes=False)
                         for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))})
    assert set(versions[0]) == set(versions[1])
    changes = {name for name in versions[0] if versions[0][name] != versions[1][name]}
    assert changes == {'_validate_job', '_owned_processes'}, changes
    return sorted(changes)

def main():
    pipeline_before = sha(ROOT/'pipeline-state.json')
    assert pipeline_before == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    writer = read('reviews/count-loss-training-controller-v5-engineering-v1.json')
    prior = read('reviews/count-loss-training-controller-v4-independent-v1.json')
    assert prior['verdict'] == 'pass_bounded_engineering_no_launch_authorization'
    assert writer['real_job_started'] is False and writer['pipeline_or_science_advanced'] is False
    evidence = {}
    for key in ('controller', 'shared_inventory', 'tests', 'test_log', 'native_diagnostic_source', 'live_scan'):
        ref = writer[key]
        assert sha(ROOT/ref['path']) == ref['sha256'].lower(), key
        evidence[ref['path']] = ref['sha256'].lower()
    for key in ('controller', 'tests', 'independent_test_log'):
        ref = prior[key]
        assert sha(ROOT/ref['path']) == ref['sha256'].lower()
        evidence[ref['path']] = ref['sha256'].lower()
    assert evidence['edsgn/scripts/count_loss_training_controller_v5.py'] == '7e60b6993ab4437fa8f907d7e63e52ec220c2726c3a13888886fa3c0acd025f0'
    assert evidence['edsgn/scripts/owned_python_job_inventory_v1.py'] == 'c31e70671f3846cdd8b1fb61e06aab45c0110d9ad5c5f1f3823377ab6d267d06'
    assert sha(ROOT/'edsgn/scripts/bound_phase_campaign.py') == '12ccfce1a6c9df0341ede4df14e37c8be04ac2cee78c96b4f75dde6d13d68bd3'
    changes = changed_functions()
    for rel in ('reviews/count-loss-training-controller-v5-engineering-v1.json',
                'reviews/count-loss-training-controller-v5-engineering-v1.md',
                'reviews/count-loss-training-controller-v4-independent-v1.json',
                'logs/count-loss-controller-v5-root-tests-v1.log',
                'edsgn/scripts/bound_phase_campaign.py', 'edsgn/scripts/bind_training_inputs.py',
                'edsgn/staging/count_loss_training_adapter_v1.py',
                'reviews/count-loss-training-adapter-root-engineering-v1.json'):
        evidence[rel] = sha(ROOT/rel)
    assert '34 passed' in (ROOT/writer['test_log']['path']).read_text(encoding='utf-8')
    assert '34 passed' in (ROOT/'logs/count-loss-controller-v5-root-tests-v1.log').read_text(encoding='utf-8')
    sys.path[:0] = [str(ROOT/'edsgn/scripts'), str(ROOT/'edsgn/staging'), str(ROOT/'edsgn/src')]
    import count_loss_training_controller_v5 as controller
    active = controller._owned_processes(ROOT)
    assert any(item['pid'] == 33368 for item in active), active
    scan = {'scope': 'metadata_only_current_full_process_scan', 'active_owned_jobs': active,
            'launch_authorized': False, 'real_job_started': False}
    scan_path = ROOT/'logs/count-loss-controller-v5-root-live-scan-v1.json'
    with scan_path.open('x', encoding='utf-8') as stream:
        json.dump(scan, stream, indent=2)
        stream.write('\n')
    evidence[scan_path.relative_to(ROOT).as_posix()] = sha(scan_path)
    assert sha(ROOT/'pipeline-state.json') == pipeline_before
    record = {
        'scope': 'paired_count_loss_controller_and_owned_process_inventory_engineering_only',
        'accepted_by': 'root', 'verdict': 'pass', 'evidence_sha256': evidence,
        'writer_cpu_tests_passed': 34, 'root_independent_cpu_tests_passed': 34,
        'prior_v4_independent_cpu_tests_passed': 26,
        'normalized_changed_functions_from_v4': changes,
        'root_review': [
            'Unchanged reviewed bound runner, exact adapter child command, complete source/input and phase fences.',
            'Trusted Toolhelp32 name fallback classifies Secure System without hardcoded PID exceptions.',
            'Native failure, empty names and unavailable Python command-line/ownership metadata refuse launch.',
            'Python prefix flags identify actual script/module; command text and non-Python launcher strings are not job targets.',
            'Shared inventory and diagnostic source are mandatory bound child inputs.',
            'Actual full read-only scan detects the active training child and controller; no training was started.'
        ],
        'scientific_hypothesis_accepted': False, 'pipeline_advanced': False,
        'actual_training_completed': False, 'actual_test_authorized': False,
        'limitations': [
            'Exactly matched original-email/MOOC V4 seed0 g00/m00 controls only; a different selected formal policy requires a versioned matched pair.',
            'Actual candidate selection, MSE numeric/root evidence, accepted queue, P3/P3A gates and cumulative resource admission remain required.',
            'Current owned training jobs prohibit launching this controller.'
        ]
    }
    output = ROOT/'reviews/count-loss-training-controller-v5-root-engineering-v1.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'pass_engineering_only', 'receipt': str(output), 'sha256': sha(output), 'active_owned_jobs': active}))

if __name__ == '__main__':
    main()
