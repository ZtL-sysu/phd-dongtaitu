"""Additive remaining probes and smaller AGCRN retry; original failure immutable."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
original = ROOT / 'protocols/p3-external-resource-queue-accepted-v1.json'
queue = json.loads(original.read_text(encoding='utf-8'))
rows, configs, retry = [], [], None
for row in queue['queue']:
    output = Path(row['output'])
    if output.exists():
        report = json.loads(output.read_text(encoding='utf-8'))
        if report['status'] == 'passed':
            continue
        if row['dataset'] != 'email' or row['method'] != 'AGCRN' or report.get('error_type') != 'OutOfMemoryError':
            raise RuntimeError('Unexpected failed resource attempt; root diagnosis required')
        cfg_path = Path(row['config'])
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        cfg['batch_size'] = 1
        next_cfg = cfg_path.with_name('email-AGCRN-seed0-B1-v2.json')
        with next_cfg.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
        retry = deepcopy(row)
        retry.update(config=str(next_cfg), batch_size_proposed=1,
                     output=str(output.with_name('email-AGCRN-seed0-B1-v2.json')))
        retry['argv'][retry['argv'].index('--config') + 1] = retry['config']
        retry['argv'][retry['argv'].index('--output') + 1] = retry['output']
        retry['argv'][retry['argv'].index('--batch-size') + 1] = '1'
        continue
    rows.append(deepcopy(row))
if retry is None:
    raise RuntimeError('Expected immutable email AGCRN B2 OOM report is absent')
rows.append(retry)
for row in rows:
    path = Path(row['config'])
    configs.append({key: row[key] for key in ('dataset', 'method', 'config', 'batch_size_proposed')})
    configs[-1]['config_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
queue.update(queue=rows, configs=configs,
             status='accepted_for_resource_probe',
             continuation={'parent_queue_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                           'reason': 'five successful probes retained; email AGCRN B2 OOM preserved; ten unstarted configurations plus B1 retry',
                           'architecture_changed': False,
                           'training_batch_acceptance': False})
destination = ROOT / 'protocols/p3-external-resource-queue-accepted-v2.json'
with destination.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(queue, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'queue': str(destination), 'rows': len(rows)}))
