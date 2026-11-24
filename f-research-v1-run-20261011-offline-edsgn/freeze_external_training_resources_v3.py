"""Root accepts measured allocation proposals only, before external full training."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
rows = []
original = json.loads((ROOT / 'configs/external-resource-v1/resource-probe-queue.json').read_text(encoding='utf-8'))
for row in original['queue']:
    dataset, method = row['dataset'], row['method']
    checkpointed = method == 'AGCRN' and dataset in ('email', 'email-episode')
    path = ROOT / f'logs/external-resource-v1/{dataset}-AGCRN-seed0-checkpoint-B4-v4.json' if checkpointed else Path(row['output'])
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('status') != 'passed' or not report.get('finite_predictions') or not report.get('finite_gradients'):
        raise RuntimeError(f'Resource cell lacks finite passed evidence: {dataset}/{method}')
    if report['optimizer_steps'] != 0 or report['test_accessed'] is not False:
        raise RuntimeError('Unexpected resource scope')
    if checkpointed and report['config'].get('baseline', {}).get('training_checkpoint_cells') is not True:
        raise RuntimeError('Large-task AGCRN repair flag absent')
    rows.append({'dataset': dataset, 'method': method, 'device': 'cuda',
                 'batch_size': report['batch_size'], 'training_checkpoint_cells': checkpointed,
                 'peak_cuda_allocated_bytes_diagnostic': report['peak_cuda_allocated_bytes'],
                 'resource_report': str(path), 'resource_report_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
contract = {'schema_version': 3, 'status': 'root-frozen-before-external-optimizer-launch',
            'scope': 'P3 full seed0 train/validation; allocation acceptance only',
            'budget_policy': 'one cumulative72-hour training/profile/resource ledger; no reset',
            'methods': rows, 'formal_core_seed_policy': 'unchanged seeds0-4',
            'training': {'epochs_max': 100, 'patience': 10, 'lr': .001, 'weight_decay': .0001, 'grad_clip': 1., 'threads': 2},
            'applicability': 'Fourteen pre-C05 passes retain unchanged method equations/default-off AGCRN; the exact C05 branch is verified separately on both large AGCRN tasks. Resource checks bound memory, not optimizer convergence or frozen inference throughput.',
            'independent_evidence': ['reviews/p3-external-resource-independent-v1.md', 'reviews/p3-agcrn-checkpoint-resource-independent-v4.md', 'reviews/agcrn-checkpoint-runtime-independent-v1.md'],
            'unchanged': ['node cohorts', 'all train/validation origins', 'history/horizon/features', 'train-only statistics', 'author architecture and parameters', 'raw masked MSE', 'test isolation'],
            'training_batch_fairness': 'Complete epochs and stopping cap are shared; accepted batch sizes differ, so optimizer update counts differ and are reported.',
            'scientific_hypothesis_accepted': False, 'inference_speed_accepted': False}
destination = ROOT / 'protocols/p3-external-training-resource-contract-v3.json'
with destination.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(contract, indent=2) + '\n')
queue_path = ROOT / 'protocols/p3-external-bound-queue-v2.json'
queue = json.loads(queue_path.read_text(encoding='utf-8'))
acceptance = {'queue': str(queue_path), 'queue_sha256': hashlib.sha256(queue_path.read_bytes()).hexdigest(),
              'resource_contract': str(destination), 'resource_contract_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
              'scope': 'root-approved full P3 development train/validation only; no test or full proposed-method launch',
              'jobs': len(queue['jobs']), 'accepted': True,
              'launch_condition': 'current strong baseline queue completes and releases resource lease'}
with (ROOT / 'protocols/p3-external-bound-queue-root-acceptance-v2.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(acceptance, indent=2) + '\n')
print(json.dumps({'accepted_resource_cells': len(rows), 'prepared_training_jobs': len(queue['jobs'])}))
