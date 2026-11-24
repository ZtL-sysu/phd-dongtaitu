"""Root acceptance of the reviewed unchanged-model P3 recovery admission."""
from pathlib import Path
import datetime as dt
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(rel): return json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))

def main():
    evidence = {}
    def pin(rel, expected=None):
        p=Path(rel); p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve()
        assert p.is_relative_to(ROOT)
        digest=sha(p)
        assert expected is None or digest==expected, str(p)
        name=p.relative_to(ROOT).as_posix()
        assert name not in evidence or evidence[name]==digest
        evidence[name]=digest
    independent='reviews/p3-coverage-recovery-prelaunch-independent-v1.json'
    pin(independent,'4c04ae887cf8106c35acc7bd998d7530c262051c47ab6c0408a3db44ccf139e7')
    pin('reviews/p3-coverage-recovery-prelaunch-independent-v1.md','04f901c0b4dbdc15f612ed1f4880432d3c99de803acf8e8d537c10902ed389b6')
    r=read(independent)
    assert r['verdict']=='preparation_matches; launch_blocked_until_root_acceptance'
    assert r['evidence']['completed_prior_results']['accepted_items']==12
    queue='protocols/p3-coverage-recovery-queue-v1.json'
    pin(queue,'534bf70d79a50dbbd533681210a0e1056d802bcc9e32489b4699604e43af8c81')
    q=read(queue)
    for entry in q['preparation_source_bindings']: pin(entry['path'],entry['sha256'])
    assert len(q['jobs'])==68 and len(q['preparation_source_bindings'])==295
    pin('logs/p3-coverage-recovery-preparation-v1.json','fb6c1002cfb2ca4368faf075836817b691b5c959bec0226a6bea83668ea86994')
    prep=read('logs/p3-coverage-recovery-preparation-v1.json')
    assert prep['actual_training_started'] is False and prep['pipeline_advanced'] is False
    assert len(prep['mapping'])==68 and all(row['changed_training_fields']==[] for row in prep['mapping'])
    pin('edsgn/scripts/launch_p3_coverage_recovery_root_v1.py','007427958e9519ced2e4f283f61c1165f686cff0b5abe3143185fca2f23368b4')
    acceptances=['reviews/p3-strong-seed0-numeric-root-acceptance-v1.json',
        *[f'reviews/p3-bound-email-episode-{method}-root-acceptance-v3.json' for method in ('DGCRN','SeqSNN')],
        *[f'reviews/p3-internal-{dataset}-{method}-root-acceptance-v1.json'
            for dataset in ('email','email-episode') for method in ('GRU','STID')]]
    for arel in acceptances:
        a=read(arel)
        assert a['accepted_by']=='root' and a['verdict']=='pass'
        assert a['pipeline_advanced'] is False and a['scientific_hypothesis_accepted'] is False
        pin(arel)
        for dep,digest in a['evidence_sha256'].items(): pin(dep,digest)
    assert len(acceptances)==7
    from bind_training_inputs import verify_binding
    original_queue=read('protocols/p3-bound-strong-baseline-queue-v3.json')
    internal=read('protocols/p3-internal-bound-queue-v4.json')
    completed=[*original_queue['jobs'],*internal['jobs'][:4]]
    assert len(completed)==12
    complete_rows=[]
    for j in completed:
        receipt_rel=f"logs/offline-phase-wall/{j['id']}.json"
        d=read(receipt_rel)
        assert d['status']=='completed' and d['returncode']==0
        pin(receipt_rel)
        for entry in d['inputs']+d['outputs']: pin(entry['path'],entry['sha256'])
        output=Path(j['output_dir'])
        summary=json.loads((output/'training_summary.json').read_text(encoding='utf-8-sig'))
        assert summary['scope']=='full_protocol' and summary['test_accessed'] is False
        assert summary['stop_loss_reached'] is False
        assert summary['checkpoint_sha256']==sha(output/'checkpoint.pt')
        assert verify_binding(output/'input-binding-at-launch.json')['status']=='verified'
        complete_rows.append({'original_job_id':j['id'],'phase_receipt':receipt_rel,'sha256':sha(ROOT/receipt_rel)})
    for j in q['jobs']:
        assert not Path(j['output_dir']).exists()
        assert not (ROOT/'logs/offline-phase-wall'/f"{j['id']}.json").exists()
    pin('reviews/count-loss-training-controller-v5-root-engineering-v1.json','348b2e085cf7677010140e9c73cdd98f136c1a245b70bcd24ac007a9e7d548ea')
    pin(Path(__file__))
    from aggregate_budget import aggregate
    budget=aggregate(ROOT)
    assert budget['coverage_complete'] and not budget['stop_loss_reached']
    assert abs(budget['spent_seconds']-27821.204501300013)<1e-6
    assert sha(ROOT/'pipeline-state.json')=='2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    record={'scope':'P3 baseline coverage recovery admission','accepted_by':'root','verdict':'pass',
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'queue_path':queue,
        'queue_sha256':sha(ROOT/queue),'job_count':68,'source_sha256':sha(ROOT/'edsgn/scripts/launch_p3_coverage_recovery_root_v1.py'),
        'evidence_sha256':evidence,'prior_numeric_root_acceptances':acceptances,
        'reused_completed_rows':complete_rows,'reused_completed_numeric_count':12,
        'method_config_changes':False,'changed_fields':['six internal output_dir values and all pending receipt IDs'],
        'interrupted_output_reused':False,'old_failed_attempt_preserved_and_charged':True,
        'budget_policy':'same single cumulative72-hour campaign; no reset',
        'spent_seconds_at_acceptance':budget['spent_seconds'],'remaining_seconds_at_acceptance':budget['remaining_seconds'],
        'official_P3_gate_and_resources_required_by_launcher':True,
        'actual_training_started_at_acceptance':False,'test_access_authorized':False,
        'pipeline_advanced':False,'scientific_hypothesis_accepted':False,
        'acceptance_boundary':'Serial execution admission only; every new complete validation numeric result needs independent and root acceptance. All P3A and actual test routes remain pending.'}
    out=ROOT/'reviews/p3-coverage-recovery-root-acceptance-v1.json'
    with out.open('x',encoding='utf-8') as f: json.dump(record,f,indent=2); f.write('\n')
    print(json.dumps({'accepted':True,'job_count':68,'prior_numeric_verified':12,'pins':len(evidence),'sha256':sha(out)}))

if __name__=='__main__': main()
