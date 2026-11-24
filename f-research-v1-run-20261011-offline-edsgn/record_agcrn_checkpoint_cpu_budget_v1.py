"""Cover the explicitly measured staged whole-command CPU diagnostic once."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
source = ROOT / 'logs/agcrn-checkpoint-adapter-cpu-v1.json'
receipt = json.loads(source.read_text(encoding='utf-8'))
seconds = float(receipt['task_duration_seconds'])
if receipt['status'] != 'passed' or seconds <= 0:
    raise RuntimeError('Invalid measured staging receipt')
ledger = ROOT / 'protocols/training-budget-manual-ledger.json'
raw = ledger.read_bytes()
doc = json.loads(raw)
identifier = 'agcrn-checkpoint-staged-cpu-v1'
if any(row.get('id') == identifier for row in doc['entries']):
    raise RuntimeError('Receipt is already covered')
with (ROOT / 'snapshots/training-budget-ledger-before-agcrn-checkpoint-cpu-v1.json').open('xb') as stream:
    stream.write(raw)
doc['entries'].append({'id': identifier, 'seconds': seconds, 'timing_basis': 'measured',
                      'source': str(source), 'covers_source': str(source),
                      'description': 'Whole staged CPU toy checkpoint parity command, explicit task_duration_seconds; zero optimizer and actual dataset; no GPU.'})
ledger.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'id': identifier, 'seconds': seconds}))
