"""Root authorization for two real train-only checkpointed AGCRN B4 probes."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
stages = {x['id']: x['status'] for x in json.loads((ROOT / 'pipeline-state.json').read_text(encoding='utf-8'))['stages']}
if stages.get('P2') != 'passed' or stages.get('P3') != 'running':
    raise RuntimeError('C05 P2 acceptance and P3 resumption required')
original = ROOT / 'configs/external-resource-v1/resource-probe-queue.json'
proposal = json.loads(original.read_text(encoding='utf-8'))
rows, configs = [], []
for dataset in ('email', 'email-episode'):
    row = deepcopy(next(r for r in proposal['queue'] if r['dataset'] == dataset and r['method'] == 'AGCRN'))
    source = Path(row['config'])
    cfg = json.loads(source.read_text(encoding='utf-8'))
    cfg['batch_size'] = 4
    cfg['baseline']['training_checkpoint_cells'] = True
    cfg['resource_proposal'] = {'batch_status': 'accepted_for_resource_probe_only',
                               'batch_size_proposal': 4, 'training_checkpoint_cells': True,
                               'purpose': 'one train-only forward/backward, zero optimizer and held-out windows'}
    config_path = source.with_name(f'{dataset}-AGCRN-seed0-checkpoint-B4-v4.json')
    with config_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(cfg, indent=2) + '\n')
    output = ROOT / f'logs/external-resource-v1/{dataset}-AGCRN-seed0-checkpoint-B4-v4.json'
    row.update(config=str(config_path), output=str(output), batch_size_proposed=4,
               status='accepted_for_resource_probe_only',
               note='C05 author-cell recomputation; new measured memory required; no optimizer')
    for flag, value in (('--config', str(config_path)), ('--output', str(output)), ('--batch-size', '4')):
        row['argv'][row['argv'].index(flag) + 1] = value
    row.pop('command', None)
    rows.append(row)
    configs.append({key: row[key] for key in ('dataset', 'method', 'config', 'batch_size_proposed')})
    configs[-1]['config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
queue = dict(schema_version=4, status='accepted_for_resource_probe',
             resource_probe_script=proposal['resource_probe_script'], configs=configs, queue=rows,
             conditions=['C05 engineering gate accepted', 'One train-only pass per task; optimizer0; validation/test windows0'],
             root_acceptance={'protocol': 'protocols/c05-agcrn-memory-repair-v1.md',
                              'training_batch_acceptance': False, 'scientific_result_acceptance': False})
destination = ROOT / 'protocols/p3-agcrn-checkpoint-resource-queue-accepted-v4.json'
with destination.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(queue, indent=2) + '\n')
print(destination)
