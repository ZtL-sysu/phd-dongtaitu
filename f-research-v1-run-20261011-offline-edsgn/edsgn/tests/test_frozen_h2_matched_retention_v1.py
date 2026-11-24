import json, sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'edsgn'/'staging'))
sys.path.insert(0,str(ROOT/'edsgn'/'src'))
import frozen_h2_matched_retention_v1 as h2run
import h2_matched_graph_retention_v2 as kernel
from edsgn.model import ForecastModel, ModelConfig

class DS:
    n_nodes=5; n_features=2; history=6; hmax=3; event_kind='toy'
    metadata={'seed':17}
    node_ids=np.arange(5)
    def origins(self,split): return [2,3]

def state(root):
    ev=root/'p3a.json'; ev.write_text('{"pass":true}',encoding='utf-8')
    return {'stages':[
        {'id':'P2','status':'passed','gate_verdict':'pass'},
        {'id':'P3','status':'passed','gate_verdict':'pass'},
        {'id':'P3A','status':'passed','gate_verdict':'pass','evidence':[{'path':str(ev),'sha256':h2run.sha256(ev)}]},
        {'id':'P4','status':'running'}]}

def test_stage_gate_accepts_only_canonical_pass_and_hashes(tmp_path):
    assert h2run.stage_gate(state(tmp_path),tmp_path)['P3A']=='passed'
    s=state(tmp_path); s['stages'][1]['status']='running'
    with pytest.raises(RuntimeError,match='P2/P3'):
        h2run.stage_gate(s,tmp_path)
    s=state(tmp_path); (tmp_path/'p3a.json').write_text('changed')
    with pytest.raises(RuntimeError,match='hash mismatch'):
        h2run.stage_gate(s,tmp_path)

def test_resource_guard_locks_owned_jobs_and_72h(tmp_path):
    output=tmp_path/'out'
    good=lambda root,exclude_active_output=None:{'coverage_complete':True,'spent_seconds':10,'budget_seconds':100}
    assert h2run.resource_preflight(tmp_path,output,good,lambda root:[])['spent_seconds']==10
    lock=tmp_path/'logs/model-resource.lock';lock.parent.mkdir();lock.write_text('{}')
    with pytest.raises(RuntimeError,match='lock'):
        h2run.resource_preflight(tmp_path,output,good,lambda root:[])
    lock.unlink()
    campaign=tmp_path/'logs/offline-phase-wall/campaign.lock';campaign.parent.mkdir(parents=True);campaign.write_text('{}')
    with pytest.raises(RuntimeError,match='lock'):
      h2run.resource_preflight(tmp_path,output,good,lambda root:[])
    campaign.unlink()
    with pytest.raises(RuntimeError,match='active'):
        h2run.resource_preflight(tmp_path,output,good,lambda root:['python train'])
    with pytest.raises(RuntimeError,match='exhausted'):
        h2run.resource_preflight(tmp_path,output,lambda *a,**k:{'coverage_complete':True,'spent_seconds':100,'budget_seconds':100},lambda root:[])

def test_test_freeze_requires_exact_hashes_and_all_arms_constants_policy():
    vals={'checkpoint_sha':'c','parameter_sha':'p','dataset_fingerprint':{'x':'y'},'collection_sha':'col','driver_sha':'d','kernel_sha':'k','source_sha':'s','dependency_sha':'dep'}
    extra={'h2_matched_retention_driver_sha256':'d','h2_matched_retention_kernel_sha256':'k',
      'h2_matched_retention_collection_sha256':'col','h2_matched_retention_source_manifest_sha256':'s',
      'h2_matched_retention_dependency_manifest_sha256':'dep','h2_matched_retention_arms':list(h2run.ARMS),
      'h2_matched_retention_control_constants':h2run.CONTROL_CONSTANTS,'h2_matched_retention_policy_kind':'full'}
    freeze={'checkpoint_sha256':'c','parameters_sha256':'p','dataset_fingerprint':{'x':'y'},'extra':extra}
    h2run.validate_test_freeze(freeze,checkpoint_sha=vals['checkpoint_sha'],parameter_sha=vals['parameter_sha'],
      dataset_fingerprint=vals['dataset_fingerprint'],collection_sha=vals['collection_sha'],driver_sha=vals['driver_sha'],
      kernel_sha=vals['kernel_sha'],source_sha=vals['source_sha'],dependency_sha=vals['dependency_sha'])
    for key in ('h2_matched_retention_arms','h2_matched_retention_policy_kind','h2_matched_retention_source_manifest_sha256'):
      broken=json.loads(json.dumps(freeze));broken['extra'].pop(key)
      with pytest.raises(RuntimeError,match='freeze'):
        h2run.validate_test_freeze(broken,checkpoint_sha='c',parameter_sha='p',dataset_fingerprint={'x':'y'},
          collection_sha='col',driver_sha='d',kernel_sha='k',source_sha='s',dependency_sha='dep')

def test_full_origin_order_and_no_gaps():
    ds=DS()
    assert h2run.validate_origins([(0,2),(0,3)],[ds],'validation')==[(0,2),(0,3)]
    for bad in ([(0,3),(0,2)],[(0,2)],[(0,2),(0,3),(0,3)]):
      with pytest.raises(RuntimeError,match='complete ordered'):
        h2run.validate_origins(bad,[ds],'validation')

def test_proof_signature_and_exclusive_outputs(tmp_path):
    t={'row_ids':torch.tensor([0]),'state_ids':torch.tensor([4]),'supports':torch.tensor([[1,2]]),
       'weights':torch.tensor([[.2,.8]]),'valid':torch.tensor([[True,True]])}
    a={'row_ids':t['row_ids'].clone(),'state_ids':t['state_ids'].clone(),'matched_supports':t['supports'].clone(),
       'matched_weights':t['weights'].clone(),'matched_valid':t['valid'].clone()}
    assert h2run._proof_signature(t,a)==h2run._proof_signature(t,a)
    a['matched_weights'][0,0]=.3
    with pytest.raises(RuntimeError,match='weights'):
      h2run._proof_signature(t,a)
    p=tmp_path/'report.json';h2run.write_json(p,{'ok':True},exclusive=True)
    with pytest.raises(FileExistsError):h2run.write_json(p,{'ok':False},exclusive=True)
    assert json.loads(p.read_text())['ok'] is True

def test_existing_run_output_is_never_reused_or_modified(tmp_path):
    out=tmp_path/'runs/already-there';out.mkdir(parents=True)
    sentinel=out/'sentinel.txt';sentinel.write_text('immutable')
    with pytest.raises(FileExistsError):
      h2run.run_checkpoint_collection('missing.pt','missing-data','validation',out,_root=tmp_path,
        _services={'state':{'stages':[]}})
    assert sentinel.read_text()=='immutable' and list(out.iterdir())==[sentinel]

def test_toy_replay_emits_full_origin_seed_predictions_and_step_proofs(tmp_path,monkeypatch):
    torch.manual_seed(4)
    model=ForecastModel(ModelConfig(n_nodes=5,n_features=2,history=6,hmax=3,count_data=False,variant='V4',
      d_total=16,encoder_width=8,readout_hidden=8,k=2,candidate_cap=4,retention_mode='informative'))
    model.eval()
    x=torch.randn(1,6,5,2)*.1; mask=torch.ones_like(x,dtype=torch.bool); age=torch.zeros_like(x);raw=x.clone()
    target=torch.rand(1,3,5,2);target_mask=torch.ones_like(target,dtype=torch.bool)
    batch={'x':x,'mask':mask,'age':age,'raw':raw,'events':None,'targets':target,'target_mask':target_mask,
      'windows':[{'time_indices':list(range(6))}]}
    import edsgn.engine as engine
    monkeypatch.setattr(engine,'batch_at',lambda datasets,pairs,device:batch)
    report=h2run.replay_windows(model,{'config':{'seed':5,'history':6,'hmax':3}},[DS()],[(0,2),(0,3)],'validation',tmp_path,
      kernel.run_matched_retention_probe,'cpu')
    assert report['origin_count']==2 and report['model_seed_id']==5 and report['trajectory_seed_ids']==[17,17]
    assert report['informative_teacher_parity_all_windows'] is True
    assert report['graph_proof_lines']==2*4*6
    assert report['targets_passed_to_kernel'] is False
    z=np.load(tmp_path/'predictions-teacher.npz',allow_pickle=False)
    assert z['predictions'].shape==(2,3,5,2) and z['origins'].tolist()==[[0,2],[0,3]]
    assert z['seed_id'].item()==5 and z['trajectory_seed_id'].tolist()==[17,17]
    lines=(tmp_path/'matched-graph-proof.jsonl').read_text().splitlines()
    assert len(lines)==48 and all(json.loads(row)['graph_match_proved'] for row in lines)
