"""Execute one reviewed recovery queue through the unchanged serial controller."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(r'C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe')
QUEUE = ROOT / 'protocols/p3-coverage-recovery-queue-v1.json'
EXPECTED_QUEUE = '534bf70d79a50dbbd533681210a0e1056d802bcc9e32489b4699604e43af8c81'
ACCEPTANCE = ROOT / 'reviews/p3-coverage-recovery-root-acceptance-v1.json'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def inside(raw):
    p = Path(raw)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT): raise RuntimeError('artifact outside RUN')
    return p

def preflight():
    from aggregate_budget import aggregate
    from bound_phase_campaign import validate_job, gate
    from owned_python_job_inventory_v1 import owned_processes
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    assert sha(QUEUE) == EXPECTED_QUEUE
    a, q = read(ACCEPTANCE), read(QUEUE)
    assert a['accepted_by'] == 'root' and a['verdict'] == 'pass'
    assert a['scope'] == 'P3 baseline coverage recovery admission'
    assert a['queue_sha256'] == EXPECTED_QUEUE and a['job_count'] == len(q['jobs']) == 68
    assert a['method_config_changes'] is False and a['test_access_authorized'] is False
    assert a['source_sha256'] == sha(Path(__file__))
    for rel, digest in a['evidence_sha256'].items(): assert sha(inside(rel)) == digest, rel
    for entry in q['preparation_source_bindings']:
        assert sha(inside(entry['path'])) == entry['sha256'], entry['path']
    assert sha(ROOT / 'edsgn/scripts/bound_phase_campaign.py') == '12ccfce1a6c9df0341ede4df14e37c8be04ac2cee78c96b4f75dde6d13d68bd3'
    assert sha(ROOT / 'edsgn/scripts/owned_python_job_inventory_v1.py') == 'c31e70671f3846cdd8b1fb61e06aab45c0110d9ad5c5f1f3823377ab6d267d06'
    for rel in ('logs/model-resource.lock','logs/offline-phase-wall/campaign.lock',
                'model-resource.lock','offline-phase-wall/campaign.lock'):
        assert not (ROOT / rel).exists(), f'existing resource lease {rel}'
    active = owned_processes(ROOT)
    assert not active, active
    budget = aggregate(ROOT)
    assert budget['coverage_complete'] and not budget['stop_loss_reached']
    for j in q['jobs']:
        kind, out = validate_job(j)
        gate(j, kind)
        assert kind == 'training' and not out.exists()
        assert not (ROOT / 'logs/offline-phase-wall' / (j['id']+'.json')).exists()
    checked = subprocess.run([str(PYTHON), str(ROOT / 'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py'),
        str(ROOT/'pipeline-state.json'),'--before','P3'],cwd=ROOT,capture_output=True,text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    # Last fence follows the official validator and its possible side effects.
    assert sha(QUEUE) == EXPECTED_QUEUE and read(ACCEPTANCE) == a
    for rel, digest in a['evidence_sha256'].items(): assert sha(inside(rel)) == digest, rel
    for entry in q['preparation_source_bindings']:
        assert sha(inside(entry['path'])) == entry['sha256'], entry['path']
    for rel in ('logs/model-resource.lock','logs/offline-phase-wall/campaign.lock',
                'model-resource.lock','offline-phase-wall/campaign.lock'):
        assert not (ROOT / rel).exists(), f'new resource lease {rel}'
    active = owned_processes(ROOT)
    assert not active, active
    budget = aggregate(ROOT)
    assert budget['coverage_complete'] and not budget['stop_loss_reached']
    return {'scope':'P3 baseline recovery prelaunch metadata check','status':'pass',
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'queue_sha256':EXPECTED_QUEUE,'acceptance_sha256':sha(ACCEPTANCE),
        'source_sha256':sha(Path(__file__)),'job_count':68,'active_owned_jobs':active,
        'scientific_source_changed':False,'test_access':False,
        'spent_seconds':budget['spent_seconds'],'remaining_seconds':budget['remaining_seconds']}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    record = preflight()
    if not args.execute:
        print(json.dumps(record)); return
    path = ROOT/'logs/p3-coverage-recovery-prelaunch-v1.json'
    with path.open('x',encoding='utf-8') as f: json.dump(record,f,indent=2); f.write('\n')
    final = preflight()
    for key in ('queue_sha256','acceptance_sha256','source_sha256','job_count'):
        assert final[key] == record[key], f'prelaunch receipt write changed {key}'
    result = subprocess.run([str(PYTHON),str(ROOT/'edsgn/scripts/bound_phase_campaign.py'),
                             '--queue',str(QUEUE)],cwd=ROOT,shell=False)
    result_path = ROOT/'logs/p3-coverage-recovery-controller-result-v1.json'
    with result_path.open('x',encoding='utf-8') as f:
        json.dump({'scope':'serial recovery controller exit','returncode':result.returncode,
            'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'queue_sha256':EXPECTED_QUEUE,'training_completion_requires_per_job_receipts':True},f,indent=2)
        f.write('\n')
    raise SystemExit(result.returncode)

if __name__ == '__main__': main()
