"""Preserve a host-terminated attempt and prepare a fresh first job only."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import psutil
from aggregate_budget import aggregate
from owned_python_job_inventory_v1 import owned_processes

ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(d,f,indent=2);f.write('\n')
def copy_new(a,b):
    b.parent.mkdir(parents=True,exist_ok=True)
    with b.open('xb') as f:f.write(a.read_bytes())

def main():
    assert not psutil.pid_exists(37416), 'original controller still alive'
    active=owned_processes(ROOT)
    assert not active,active
    phase=ROOT/'logs/offline-phase-wall/coverage-recovery-v1-internal-bound-v4-email-V0-seed0.json'
    d=read(phase)
    assert d['status']=='running' and d['wall_seconds']==67.37345189999905
    assert not (ROOT/'logs/p3-coverage-recovery-controller-result-v1.json').exists()
    out=Path(d['output_dir'])
    assert not (out/'checkpoint.pt').exists() and not (out/'training_summary.json').exists()
    evidence=ROOT/'logs/host-interruption-evidence-v2'
    copy_new(phase,evidence/'raw-running-phase.json')
    locks=[]
    for rel in ('logs/model-resource.lock','logs/offline-phase-wall/campaign.lock'):
        p=ROOT/rel
        if p.exists():
            lock=read(p);pid=int(lock['pid'])
            assert not psutil.pid_exists(pid),f'live lease owner {pid}'
            backup=evidence/(p.name+'.saved')
            copy_new(p,backup)
            locks.append({'path':rel,'pid':pid,'backup':backup.relative_to(ROOT).as_posix(),'sha256':sha(backup)})
            # Absolute containment and dead owner verified before unlink.
            assert p.resolve().is_relative_to(ROOT)
            p.unlink()
    observed=dt.datetime(2026,10,11,14,37,45,tzinfo=dt.timezone.utc)
    start=dt.datetime.fromisoformat(d['started_utc'])
    upper=(observed-start).total_seconds()
    assert upper==158 and upper>=d['wall_seconds']
    supplement=upper-d['wall_seconds']
    before=aggregate(ROOT)
    assert before['coverage_complete'] and not before['stop_loss_reached']
    ledger=ROOT/'protocols/training-budget-manual-ledger.json'
    copy_new(ledger,ROOT/'protocols/training-budget-manual-ledger-before-host-repair-v2.json')
    ld=read(ledger)
    ld['entries'].append({'id':'host-termination-unobserved-interval-v2','seconds':supplement,
        'timing_basis':'conservative-bound','source':str(evidence/'raw-running-phase.json'),
        'description':'Only excess over last measured controller wall67.3734519s: conservative upper elapsed158s through first timestamped authoritative no-controller/no-model observation14:37:45UTC. Adds90.6265481s; same cumulative72h cap, no reset.'})
    tmp=ledger.with_name('training-budget-manual-ledger-host-repair-v2.tmp')
    write(tmp,ld);os.replace(tmp,ledger)
    stopped={**d,'status':'stopped','returncode':None,
        'reason':'host tool-turn interruption; controller and model absent; no completed epoch or summary',
        'external_terminal_observed_utc':observed.isoformat(),
        'finished_utc':None,'wall_seconds_basis':'last actual controller perf-counter measurement',
        'unobserved_interval_conservative_ledger_entry':'host-termination-unobserved-interval-v2',
        'root_reconciliation':{'raw_receipt_path':str(evidence/'raw-running-phase.json'),
            'raw_receipt_sha256':sha(evidence/'raw-running-phase.json'),'controller_pid_confirmed_absent':37416,
            'active_owned_jobs':active,'no_training_completion_inferred':True}}
    phase_tmp=phase.with_name(phase.name+'.host-reconcile-v2.tmp')
    write(phase_tmp,stopped);os.replace(phase_tmp,phase)
    event=ROOT/'logs/p3-host-interruption-root-v2.json'
    write(event,{'scope':'engineering_only_host_interruption_reconciliation','status':'terminal_processes_confirmed_absent',
        'observed_utc':observed.isoformat(),'reconciled_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'controller_pid':37416,'active_owned_jobs':active,'stale_dead_owner_locks':locks,
        'raw_phase_receipt':str(evidence/'raw-running-phase.json'),'raw_phase_sha256':sha(evidence/'raw-running-phase.json'),
        'measured_controller_seconds':d['wall_seconds'],'conservative_supplement_seconds':supplement,
        'total_budget_allowance_seconds':upper,'original_output_reused':False,'new_training_launched':False})
    q=read(ROOT/'protocols/p3-coverage-recovery-queue-v1.json')
    assert len(q['jobs'])==68
    first=q['jobs'][0]
    old_cfg=Path(first['args'][first['args'].index('--config')+1])
    cfg=read(old_cfg)
    newout=ROOT/'runs/p3-host-recovery-v2/email-V0-seed0'
    assert not newout.exists()
    cfg['output_dir']=str(newout)
    cfgpath=ROOT/'configs/p3-host-recovery-v2/email-V0-seed0.json'
    write(cfgpath,cfg)
    first['id']='host-recovery-v2-internal-bound-v4-email-V0-seed0'
    first['args'][first['args'].index('--config')+1]=str(cfgpath)
    if 'config' in first:first['config']=str(cfgpath)
    first['output_dir']=str(newout)
    first['inputs'] += [str(event),str(phase),str(evidence/'raw-running-phase.json')]
    pins={x['path']:x['sha256'] for x in q['preparation_source_bindings']}
    pins['protocols/training-budget-manual-ledger.json']=sha(ledger)
    for p in (cfgpath,phase,event,Path(__file__),ROOT/'protocols/training-budget-manual-ledger-before-host-repair-v2.json',evidence/'raw-running-phase.json'):
        pins[p.relative_to(ROOT).as_posix()]=sha(p)
    for key,expected in pins.items():assert sha(ROOT/key)==expected,key
    q['preparation_source_bindings']=[{'path':k,'sha256':v} for k,v in sorted(pins.items())]
    q['scope']='P3 baseline-only serial host-session recovery; first job fresh; no test'
    qpath=ROOT/'protocols/p3-host-recovery-queue-v2.json'
    write(qpath,q)
    after=aggregate(ROOT)
    assert after['coverage_complete'] and not after['stop_loss_reached']
    assert abs(after['spent_seconds']-before['spent_seconds']-supplement)<1e-6
    write(ROOT/'logs/p3-host-recovery-preparation-v2.json',{'scope':'engineering_only_recovery_delta_preparation',
        'queue_sha256':sha(qpath),'job_count':68,'changed_logical_jobs':1,
        'only_changed_model_config_field':'output_dir','other67jobs_identical':q['jobs'][1:]==read(ROOT/'protocols/p3-coverage-recovery-queue-v1.json')['jobs'][1:],
        'old_attempt_preserved':True,'new_training_started':False,'budget_seconds':after['spent_seconds'],
        'coverage_complete':after['coverage_complete'],'stop_loss_reached':after['stop_loss_reached'],
        'source_pins':len(pins),'pipeline_advanced':False})
    print(json.dumps({'prepared':68,'queue_sha256':sha(qpath),'spent_seconds':after['spent_seconds'],'pins':len(pins)}))

if __name__=='__main__':main()
