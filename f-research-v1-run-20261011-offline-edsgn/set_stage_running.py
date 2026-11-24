"""Mark the first unfinished, authorized stage running; never grant acceptance."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VALIDATOR = Path(r'C:\Users\xhf\.codex\skills\f-research-v1\scripts\pipeline_state.py')

stage = sys.argv[1]
path = ROOT / 'pipeline-state.json'
subprocess.run([sys.executable, str(VALIDATOR), str(path), '--before', stage], check=True)
state = json.loads(path.read_text(encoding='utf-8'))
row = next(r for r in state['stages'] if r['id'] == stage)
if row['status'] not in {'pending', 'running'}:
    raise ValueError(f"Cannot start {stage}: {row['status']}")
row['status'] = 'running'
path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
subprocess.run([sys.executable, str(VALIDATOR), str(path)], check=True)
lines = ['# Offline EDSGN pipeline ledger', '', 'Canonical authority: pipeline-state.json. User scope: offline implementation and experiments P0–P5A.', '', '| Stage | Status | Evidence |', '|---|---|---|']
for r in state['stages']:
    lines.append(f"| {r['id']} | {r['status']} | {', '.join(x['path'] for x in r['evidence'])} |")
lines.extend(['', 'Offline work complete: false. Full manuscript workflow complete: false. Submission ready: false.'])
(ROOT / 'pipeline-state.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
