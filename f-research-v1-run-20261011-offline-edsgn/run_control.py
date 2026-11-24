"""Hash and advance real gates; use only after primary-agent evidence acceptance."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VALIDATOR = Path(r'C:\Users\xhf\.codex\skills\f-research-v1\scripts\pipeline_state.py')

def record(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file() or not p.stat().st_size:
        raise ValueError(f'Invalid artifact: {path}')
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--evidence', nargs='+', required=True)
    args = ap.parse_args()
    path = ROOT / 'pipeline-state.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    state['contract'] = [record(p) for p in ['execution-contract-v1.md', 'inputs/implementation-plan.md', 'inputs/PhD_Research_Proposal.pdf', 'path-containment-manifest.json']]
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    subprocess.run([sys.executable, str(VALIDATOR), str(path), '--before', args.stage], check=True)
    row = next(r for r in state['stages'] if r['id'] == args.stage)
    if row['status'] == 'passed':
        raise ValueError('Already passed; use an explicit back-edge for correction')
    row.update(status='passed', gate_verdict='pass', contract_version=state['contract_version'], inputs=[record(p) for p in args.inputs], evidence=[record(p) for p in args.evidence])
    state['history'].append({'action':'pass','stage':args.stage})
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable, str(VALIDATOR), str(path)], check=True)
    lines=['# Offline EDSGN pipeline ledger', '', 'Canonical authority: pipeline-state.json. User scope: offline implementation and experiments P0–P5A.', '', '| Stage | Status | Evidence |','|---|---|---|']
    for r in state['stages']:
        lines.append(f"| {r['id']} | {r['status']} | {', '.join(x['path'] for x in r['evidence'])} |")
    lines.extend(['', 'Offline work complete: false. Full manuscript workflow complete: false. Submission ready: false.'])
    (ROOT/'pipeline-state.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__ == '__main__':
    main()
