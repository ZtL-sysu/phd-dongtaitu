"""Record root resource-probe authorization without altering proposal/config bytes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
state = json.loads((ROOT / 'pipeline-state.json').read_text(encoding='utf-8'))
stages = {row['id']: row['status'] for row in state['stages']}
if stages.get('P2') != 'passed' or stages.get('P3') != 'running':
    raise RuntimeError('Resource approval requires accepted C04 P2 and resumed P3')
proposal = ROOT / 'configs/external-resource-v1/resource-probe-queue.json'
queue = json.loads(proposal.read_text(encoding='utf-8'))
if queue.get('status') != 'proposed_unlaunched_unaccepted':
    raise RuntimeError('Unexpected proposal state')
for row in queue['configs']:
    actual = hashlib.sha256(Path(row['config']).read_bytes()).hexdigest()
    if actual != row['config_sha256']:
        raise RuntimeError(f"Immutable proposal config changed: {row['config']}")
queue['status'] = 'accepted_for_resource_probe'
queue['root_acceptance'] = {
    'scope': 'single train-only forward/backward resource pass, zero optimizer steps',
    'proposal_sha256': hashlib.sha256(proposal.read_bytes()).hexdigest(),
    'p2_gate': 'protocols/p2-engineering-delta-gate-v3.md',
    'p2_source_archive_sha256': hashlib.sha256((ROOT / 'snapshots/p2-code-v3.zip').read_bytes()).hexdigest(),
    'training_batch_acceptance': False,
    'scientific_result_acceptance': False,
}
destination = ROOT / 'protocols/p3-external-resource-queue-accepted-v1.json'
with destination.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(queue, ensure_ascii=False, indent=2) + '\n')
print(destination)
