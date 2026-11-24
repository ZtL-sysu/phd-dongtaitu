from __future__ import annotations
import hashlib, importlib.util, json, sys, types
from pathlib import Path
import pytest
ROOT0=Path(__file__).resolve().parents[2]
SCRIPT=ROOT0/'edsgn'/'scripts'
sys.path.insert(0,str(SCRIPT));sys.path.insert(0,str(ROOT0/'edsgn'/'src'))
import bound_phase_campaign as campaign


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj),encoding='utf-8')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def init_root(tmp,monkeypatch,states=None):
    monkeypatch.setattr(campaign,'ROOT',tmp)
    write_json(tmp/'pipeline-state.json',{'stages':states or [{'id':'P2','status':'passed'},{'id':'P3','status':'running'},{'id':'P3A','status':'passed'},{'id':'P4','status':'running'}]})
    return tmp
def make_queue(root,phase='calibration',job_id='j1',output=None,inputs=None,args=None):
    output=output or root/'runs'/'out';inputs=inputs or []
    q=root/'queue.json'; auxiliary=root/'calibration-queue.json';write_json(auxiliary,{'jobs':[]})
    checkpoint=root/'ckpt.pt'
    if not checkpoint.exists(): checkpoint.write_bytes(b'frozen-checkpoint')
    if args is None:
      args=['--output',str(output),'--checkpoint',str(checkpoint)]
      if phase=='calibration': args += ['--queue',str(auxiliary),'--epsilon','0.1','--device','cpu']
      elif phase in {'evaluation','profiling'}: args += ['--dataset-path',str(root/'dataset'),'--split','test']
      elif phase=='robustness': args += ['--recipe',str(root/'recipe.json'),'--batch-size','1']
    q.write_text(json.dumps({'jobs':[{'id':job_id,'phase':phase,'output_dir':str(output),'args':args,'inputs':inputs}]}),encoding='utf-8')
    return q,output
def permissive_budget(*args,**kwargs):return {'coverage_complete':True,'spent_seconds':0,'coverage_gaps':[]}
def declared(root,path):return [{'path':str(path),'sha256':digest(path)}]


def test_gate_order_and_output_containment(tmp_path,monkeypatch):
    root=init_root(tmp_path,monkeypatch)
    cfg=root/'v1.json';write_json(cfg,{'variant':'V1'})
    campaign.gate({'scope':'development','args':['--config',str(cfg)]},'training')
    write_json(root/'pipeline-state.json',{'stages':[{'id':'P2','status':'passed'},{'id':'P3A','status':'pending'},{'id':'P4','status':'running'}]})
    with pytest.raises(RuntimeError,match='P3A=passed/P4=running'):campaign.gate({},'calibration')
    write_json(root/'pipeline-state.json',{'stages':[{'id':'P2','status':'pending'},{'id':'P3','status':'running'}]})
    with pytest.raises(RuntimeError,match='P2'):campaign.gate({},'training')
    out=root/'runs'/'safe'
    
    q,out2=make_queue(root,output=out); job=json.loads(q.read_text())['jobs'][0]
    assert campaign.validate_job(job)==('calibration',out)
    with pytest.raises(ValueError,match='outside'):
      campaign.validate_job({'id':'x','phase':'calibration','output_dir':str(root.parent/'escape'),'args':['--output',str(root.parent/'escape'),'--checkpoint',str(root/'ckpt.pt'),'--queue',str(root/'queue.json'),'--epsilon','0.1']})


@pytest.mark.parametrize('variant', ['V2','V3','V4','V5','V6','V7','V8','V9'])
def test_proposed_training_waits_for_p3a_and_p4(tmp_path,monkeypatch,variant):
    root=init_root(tmp_path,monkeypatch)
    cfg=root/f'{variant}.json';write_json(cfg,{'variant':variant})
    job={'scope':'development','args':['--config',str(cfg)]}
    with pytest.raises(RuntimeError,match='P3 training is baseline-only'):
        campaign.gate(job,'training')
    write_json(root/'pipeline-state.json',{'stages':[{'id':'P2','status':'passed'},{'id':'P3','status':'passed'},{'id':'P3A','status':'pending'},{'id':'P4','status':'pending'}]})
    with pytest.raises(RuntimeError,match='P3A=passed/P4=running'):
        campaign.gate(dict(job,scope='pilot'),'training')
    write_json(root/'pipeline-state.json',{'stages':[{'id':'P2','status':'passed'},{'id':'P3','status':'passed'},{'id':'P3A','status':'passed'},{'id':'P4','status':'running'}]})
    with pytest.raises(RuntimeError,match='pilot/formal scope'):
        campaign.gate(job,'training')
    campaign.gate(dict(job,scope='pilot'),'training')
    campaign.gate(dict(job,scope='formal'),'training')


def test_p3_formal_baseline_waits_for_p4(tmp_path,monkeypatch):
    root=init_root(tmp_path,monkeypatch)
    cfg=root/'gru.json';write_json(cfg,{'variant':'GRU'})
    job={'scope':'formal','args':['--config',str(cfg)]}
    with pytest.raises(RuntimeError,match='development/baseline scope'):
        campaign.gate(job,'training')
    write_json(root/'pipeline-state.json',{'stages':[{'id':'P2','status':'passed'},{'id':'P3','status':'passed'},{'id':'P3A','status':'passed'},{'id':'P4','status':'running'}]})
    campaign.gate(job,'training')


def test_failed_phase_receipt_is_preserved(tmp_path,monkeypatch):
    root=init_root(tmp_path,monkeypatch);q,out=make_queue(root);wall=root/'logs'/'offline-phase-wall';wall.mkdir(parents=True)
    rp=wall/'j1.json';write_json(rp,{'status':'failed','queue_sha256':digest(q),'inputs':[]});before=rp.read_bytes()
    monkeypatch.setattr(campaign,'aggregate',permissive_budget);monkeypatch.setattr(campaign,'_is_training_process',lambda:[])
    monkeypatch.setattr(campaign.subprocess,'Popen',lambda *a,**k:pytest.fail('failed phase must not relaunch'))
    with pytest.raises(RuntimeError,match='explicit recovery'):campaign.run_queue(q)
    assert rp.read_bytes()==before


def _calibration_artifacts(output,checkpoint,epsilon=0.1,device='cpu',bad=None):
    table={'batch_size':1,'device':device,'base_ms':1.0}
    report={'scope':'full_validation','parameters_before':'p-hash','parameters_after':'p-hash','selected_policy':{'g':1,'s':1},'checkpoint_sha256':digest(checkpoint),'primary_epsilon':epsilon,'selection_uses_test':False,'batch_size':1,'cost_table_batch1_only':True,'device':device,'cost_table':table}
    if bad=='checkpoint': report['checkpoint_sha256']='bad'
    if bad=='epsilon': report['primary_epsilon']=0.9
    if bad=='table': report['cost_table']={'batch_size':1,'device':'cuda','base_ms':9.0}
    write_json(output/'calibration.json',report)
    write_json(output/'cost-table.json',table if bad!='table' else {'batch_size':1,'device':'cpu','base_ms':1.0})
    (output/'grouped_action_costs.csv').write_text('group,ms\nall,1.0\n',encoding='utf-8')


def fake_popen_factory(output,checkpoint,write_outputs=True,mutate=None,bad=None):
    class Proc:
      returncode=0;terminated=0;waited=0
      def __init__(self,*args,**kwargs):
        if write_outputs:_calibration_artifacts(output,checkpoint,bad=bad)
        if mutate:mutate()
      def poll(self):return 0
      def wait(self,timeout=None):self.waited+=1;return self.returncode
      def terminate(self):self.terminated+=1;self.returncode=-15
      def kill(self):self.returncode=-9
    return Proc


def test_complete_calibration_mock_success_records_output_hashes(tmp_path,monkeypatch):
    root=init_root(tmp_path,monkeypatch);frozen=root/'freeze.json';frozen.write_text('{}',encoding='utf-8')
    out=root/'runs'/'calib';q,_=make_queue(root,output=out,inputs=[str(frozen)],args=['--checkpoint',str(frozen),'--queue',str(root/'calibration-queue.json'),'--epsilon','0.1','--device','cpu','--output',str(out)])
    monkeypatch.setattr(campaign,'aggregate',permissive_budget);monkeypatch.setattr(campaign,'_is_training_process',lambda:[])
    monkeypatch.setattr(campaign.subprocess,'Popen',fake_popen_factory(out,frozen))
    campaign.run_queue(q)
    receipt=json.loads((root/'logs'/'offline-phase-wall'/'j1.json').read_text())
    assert receipt['status']=='completed' and len(receipt['outputs'])==3
    campaign.verify_entries(receipt['outputs'])


def test_zero_return_with_empty_outputs_fails_and_preserves_receipt(tmp_path,monkeypatch):
    root=init_root(tmp_path,monkeypatch);out=root/'runs'/'calib';q,_=make_queue(root,output=out)
    monkeypatch.setattr(campaign,'aggregate',permissive_budget);monkeypatch.setattr(campaign,'_is_training_process',lambda:[])
    monkeypatch.setattr(campaign.subprocess,'Popen',fake_popen_factory(out,root/'ckpt.pt',write_outputs=False))
    with pytest.raises(RuntimeError,match='Preserved unsuccessful phase'):campaign.run_queue(q)
    rp=root/'logs'/'offline-phase-wall'/'j1.json';rec=json.loads(rp.read_text())
    assert rec['status']=='failed' and 'Completed phase lacks artifact' in rec['reason'] and rp.is_file()


def test_changed_output_rejects_completed_receipt_without_relaunch(tmp_path,monkeypatch):
    root=init_root(tmp_path,monkeypatch);out=root/'runs'/'calib';q,_=make_queue(root,output=out)
    monkeypatch.setattr(campaign,'aggregate',permissive_budget);monkeypatch.setattr(campaign,'_is_training_process',lambda:[])
    monkeypatch.setattr(campaign.subprocess,'Popen',fake_popen_factory(out,root/'ckpt.pt'));campaign.run_queue(q)
    (out/'cost-table.json').write_text('corrupted',encoding='utf-8')
    monkeypatch.setattr(campaign.subprocess,'Popen',lambda *a,**k:pytest.fail('changed output must not relaunch'))
    with pytest.raises(RuntimeError,match='Phase input changed'):campaign.run_queue(q)
    assert (out/'cost-table.json').read_text()=='corrupted'


def test_referenced_frozen_inputs_must_be_in_queue_hashes(tmp_path,monkeypatch):
    """Current driver only hashes job.inputs, not file-valued CLI arguments."""
    root=init_root(tmp_path,monkeypatch);checkpoint=root/'ckpt.pt';checkpoint.write_bytes(b'checkpoint')
    out=root/'runs'/'calib';q,_=make_queue(root,output=out,inputs=[],args=['--checkpoint',str(checkpoint),'--queue',str(root/'calibration-queue.json'),'--epsilon','0.1','--device','cpu','--output',str(out)])
    monkeypatch.setattr(campaign,'aggregate',permissive_budget);monkeypatch.setattr(campaign,'_is_training_process',lambda:[])
    monkeypatch.setattr(campaign.subprocess,'Popen',fake_popen_factory(out,checkpoint,mutate=lambda:checkpoint.write_bytes(b'changed')))
    with pytest.raises(RuntimeError,match='Preserved unsuccessful phase'):campaign.run_queue(q)
    rec=json.loads((root/'logs'/'offline-phase-wall'/'j1.json').read_text())
    assert rec['status']=='failed' and 'Phase input changed' in rec['reason'] and any(Path(x['path'])==checkpoint for x in rec['inputs'])


def test_owned_pid_is_blocked_but_stop_targets_only_child(tmp_path,monkeypatch):
    root=init_root(tmp_path,monkeypatch)
    class Owned:
      pid=987654
      info={'pid':pid,'cmdline':['python','-m','edsgn.calibrate'],'cwd':str(root),'exe':'C:/python.exe'}
      terminate_calls=0
      def terminate(self):self.terminate_calls+=1
    owned=Owned()
    fakeps=types.SimpleNamespace(process_iter=lambda fields:[owned],NoSuchProcess=type('NSP',(Exception,),{}),AccessDenied=type('AD',(Exception,),{}))
    monkeypatch.setitem(sys.modules,'psutil',fakeps)
    assert campaign._is_training_process()==[{'pid':owned.pid}]
    # Next, a mocked wall-budget stop may terminate only the Popen owned by this queue run.
    monkeypatch.setattr(campaign,'_is_training_process',lambda:[])
    out=root/'runs'/'calib';q,_=make_queue(root,output=out);child=fake_popen_factory(out,root/'ckpt.pt',write_outputs=False)
    instances=[]
    class Child(child):
      def __init__(self,*a,**kw):super().__init__(*a,**kw);instances.append(self)
      def poll(self):return None if not self.terminated else self.returncode
    calls=[]
    def budget(*args,**kwargs):
      calls.append(kwargs)
      return {'coverage_complete':len(calls)==1,'spent_seconds':0,'coverage_gaps':['mock stop'] if len(calls)>1 else []}
    monkeypatch.setattr(campaign,'aggregate',budget);monkeypatch.setattr(campaign.subprocess,'Popen',Child);monkeypatch.setattr(campaign.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='Preserved unsuccessful phase'):campaign.run_queue(q)
    assert len(instances)==1 and instances[0].terminated==1 and instances[0].waited==1
    assert owned.terminate_calls==0


@pytest.mark.parametrize('bad', ['checkpoint','epsilon','table'])
def test_calibration_report_bindings_must_match_cli(tmp_path,monkeypatch,bad):
    root=init_root(tmp_path,monkeypatch);out=root/'runs'/'calib';q,_=make_queue(root,output=out)
    monkeypatch.setattr(campaign,'aggregate',permissive_budget);monkeypatch.setattr(campaign,'_is_training_process',lambda:[])
    monkeypatch.setattr(campaign.subprocess,'Popen',fake_popen_factory(out,root/'ckpt.pt',bad=bad))
    with pytest.raises(RuntimeError,match='Preserved unsuccessful phase'): campaign.run_queue(q)
    rec=json.loads((root/'logs'/'offline-phase-wall'/'j1.json').read_text())
    assert rec['status']=='failed'
    assert any(token in rec['reason'] for token in ('checkpoint','epsilon','cost table'))


def test_phase_cli_contract_requires_checkpoint_and_rejects_simple_baseline(tmp_path,monkeypatch):
    root=init_root(tmp_path,monkeypatch);out=root/'runs'/'eval';ck=root/'ckpt.pt';ck.write_bytes(b'x')
    with pytest.raises(ValueError,match='--checkpoint'):
      campaign.validate_job({'id':'e','phase':'evaluation','output_dir':str(out),'args':['--output',str(out),'--dataset-path',str(root),'--split','test']})
    job={'id':'e','phase':'evaluation','output_dir':str(out),'args':['--output',str(out),'--checkpoint',str(ck),'--dataset-path',str(root),'--split','test','--simple-baselines']}
    with pytest.raises(ValueError,match='simple baselines'): campaign.validate_job(job)


def test_training_validation_coverage_uses_metadata_and_array_headers(tmp_path):
    import numpy as np
    root=tmp_path;out=root/'training';out.mkdir();dataset=root/'data';dataset.mkdir()
    meta={'partition':'chronological','history':3,'hmax':2,'splits':{'train':[0,20],'validation':[10,20]}}
    for split in ('train','validation'):
      np.savez_compressed(dataset/f'{split}_seed0.npz',metadata_json=np.asarray(json.dumps(meta)))
    train_count=16;val_count=9
    write_json(out/'config_resolved.json',{'dataset_path':str(dataset),'epochs':2})
    write_json(out/'manifest.json',{'train_windows':train_count,'val_windows':val_count})
    write_json(out/'training_summary.json',{'scope':'full_protocol','test_accessed':False,'stop_loss_reached':False,'checkpoint_sha256':'x','epochs_executed':2})
    (out/'checkpoint.pt').write_bytes(b'toy')
    summary=json.loads((out/'training_summary.json').read_text());summary['checkpoint_sha256']=digest(out/'checkpoint.pt');write_json(out/'training_summary.json',summary)
    (out/'training_log.csv').write_text('epoch,train_windows\n1,16\n2,16\n',encoding='utf-8')
    np.savez_compressed(out/'validation_predictions.npz',predictions=np.zeros((val_count,2,1,1)),targets=np.zeros((val_count,2,1,1)),origins=np.zeros((val_count,2),dtype=int))
    (out/'metrics_by_horizon_val.csv').write_text('horizon,mae\n1,0\n',encoding='utf-8')
    job={'args':['--config',str(root/'config.json')]}
    write_json(root/'config.json',{'output_dir':str(out)})
    assert campaign.validate_outputs(job,out,'training')
    # A truncated prediction bundle and an underreported epoch count are both rejected.
    np.savez_compressed(out/'validation_predictions.npz',predictions=np.zeros((val_count-1,2,1,1)),targets=np.zeros((val_count-1,2,1,1)),origins=np.zeros((val_count-1,2),dtype=int))
    with pytest.raises(RuntimeError,match='validation prediction coverage'): campaign.validate_outputs(job,out,'training')
    np.savez_compressed(out/'validation_predictions.npz',predictions=np.zeros((val_count,2,1,1)),targets=np.zeros((val_count,2,1,1)),origins=np.zeros((val_count,2),dtype=int))
    summary['epochs_executed']=1;write_json(out/'training_summary.json',summary)
    with pytest.raises(RuntimeError,match='epoch/window coverage'): campaign.validate_outputs(job,out,'training')

def test_robustness_command_uses_callable_standalone_wrapper(monkeypatch):
    monkeypatch.setattr(campaign,'PYTHON',Path(sys.executable))
    command=campaign.command_for('robustness',['--checkpoint','frozen.pt','--recipe','recipe.json','--output','out','--batch-size','1'])
    wrapper=Path(command[1]);assert wrapper.name=='frozen_robustness_phase.py' and wrapper.is_file()
    spec=importlib.util.spec_from_file_location('frozen_robustness_phase_test',wrapper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert callable(module.main) and command[2:]==['--checkpoint','frozen.pt','--recipe','recipe.json','--output','out','--batch-size','1']


