"""Accept the independently reviewed one-job host-recovery delta."""
from pathlib import Path
import datetime as dt
import hashlib
import json
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(rel):return json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))
def main():
    pins={}
    def pin(raw,expected=None):
        p=Path(raw);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve()
        assert p.is_relative_to(ROOT)
        value=sha(p);assert expected is None or value==expected,str(p)
        rel=p.relative_to(ROOT).as_posix()
        assert rel not in pins or pins[rel]==value
        pins[rel]=value
    pin('reviews/p3-host-recovery-prelaunch-independent-v2.json','fa946bd442fbaa202065165ca5677c66bf2eafe2c1ad5531e7c003107c8e97e8')
    pin('reviews/p3-host-recovery-prelaunch-independent-v2.md','be8c6c81276ea348a8e0a69a0f352340f6a5cdbd6b851c9be53bdb157e26ff24')
    prior_rel='reviews/p3-coverage-recovery-root-acceptance-v1.json'
    pin(prior_rel,'38de1003b09928454ff498354879d25a893e574cce97acd563b208aa52dc6178')
    prior=read(prior_rel)
    assert prior['accepted_by']=='root' and prior['verdict']=='pass'
    # The prior mutable ledger version is preserved, rather than pretending it
    # still occupies the current-ledger path after the explicit supplementation.
    for rel,digest in prior['evidence_sha256'].items():
        if rel=='protocols/training-budget-manual-ledger.json':
            pin('protocols/training-budget-manual-ledger-before-host-repair-v2.json',digest)
        else:pin(rel,digest)
    qrel='protocols/p3-host-recovery-queue-v2.json'
    pin(qrel,'7752189ff13fad4a69b8792fa4a7010bb468897b37c9c9d26d16563464c8985f')
    q=read(qrel);old=read('protocols/p3-coverage-recovery-queue-v1.json')
    assert len(q['jobs'])==68 and q['jobs'][1:]==old['jobs'][1:]
    assert len(q['preparation_source_bindings'])==301
    for entry in q['preparation_source_bindings']:pin(entry['path'],entry['sha256'])
    newcfg=Path(q['jobs'][0]['args'][1]);oldcfg=Path(old['jobs'][0]['args'][1])
    nc=json.loads(newcfg.read_text());oc=json.loads(oldcfg.read_text())
    assert {k:v for k,v in nc.items() if k!='output_dir'}=={k:v for k,v in oc.items() if k!='output_dir'}
    assert q['jobs'][0]['id']=='host-recovery-v2-internal-bound-v4-email-V0-seed0'
    for j in q['jobs']:
        assert not Path(j['output_dir']).exists()
        assert not (ROOT/'logs/offline-phase-wall'/f"{j['id']}.json").exists()
    for rel in ('logs/p3-host-interruption-root-v2.json','logs/p3-host-recovery-preparation-v2.json',
                'edsgn/scripts/prepare_p3_host_launcher_root_v2.py'):
        pin(rel)
    pin('edsgn/scripts/launch_p3_host_recovery_root_v2.py','f99ec0945fb97fecc1e2bec939b303f5636b460e16a5a2e05196a7269dfd15f0')
    event=read('logs/p3-host-interruption-root-v2.json')
    assert event['new_training_launched'] is False and event['original_output_reused'] is False
    assert event['measured_controller_seconds']+event['conservative_supplement_seconds']==158
    assert read('logs/offline-phase-wall/coverage-recovery-v1-internal-bound-v4-email-V0-seed0.json')['status']=='stopped'
    from aggregate_budget import aggregate
    from owned_python_job_inventory_v1 import owned_processes
    b=aggregate(ROOT)
    assert b['coverage_complete'] and not b['stop_loss_reached']
    assert abs(b['spent_seconds']-27979.204501300013)<1e-6
    assert not owned_processes(ROOT)
    pin(Path(__file__))
    d={'scope':'P3 baseline host recovery admission','accepted_by':'root','verdict':'pass',
       'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'queue_path':qrel,'queue_sha256':sha(ROOT/qrel),
       'job_count':68,'source_sha256':sha(ROOT/'edsgn/scripts/launch_p3_host_recovery_root_v2.py'),
       'evidence_sha256':pins,'prior_numeric_root_acceptances':prior['prior_numeric_root_acceptances'],
       'reused_completed_numeric_count':12,'method_config_changes':False,
       'test_access_authorized':False,'pipeline_advanced':False,'scientific_hypothesis_accepted':False,
       'actual_training_started_at_acceptance':False,'interrupted_output_reused':False,
       'budget_policy':'same single cumulative72-hour campaign; both stopped attempts preserved and charged; no reset',
       'spent_seconds_at_acceptance':b['spent_seconds'],'remaining_seconds_at_acceptance':b['remaining_seconds'],
       'launch_mode':'Hidden background process; retain authoritative PID and per-job receipts across tool turns',
       'prior_admission_is_historical':'Its then-current manual ledger bytes are preserved in before-host-repair-v2 snapshot; other prior dependencies and12numeric results rehash unchanged.'}
    dest=ROOT/'reviews/p3-host-recovery-root-acceptance-v2.json'
    with dest.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2);f.write('\n')
    print(json.dumps({'accepted':True,'job_count':68,'source_pins':len(pins),'sha256':sha(dest)}))
if __name__=='__main__':main()
