import hashlib, json, shutil, subprocess, sys
from pathlib import Path
import pytest
RUN=Path(r'C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn')
sys.path.insert(0,str(RUN/'edsgn/scripts'))
import verified_frozen_dispatch_v2 as d

def put(p,data='x'):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(data,encoding='utf-8'); return p
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def setup(tmp_path,monkeypatch,driver):
 root=tmp_path/'run'; root.mkdir(parents=True)
 for rel in d.VALIDATOR_FILES:
  dst=root/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(RUN/rel,dst)
 dr=d.DRIVERS[driver][0]; put(root/dr,'# mocked driver')
 dispatcher='edsgn/scripts/verified_frozen_dispatch_v2.py'; put(root/dispatcher,(RUN/dispatcher).read_text(encoding='utf-8'))
 monkeypatch.setattr(d,'__file__',str(root/dispatcher))
 put(root/'contract.md','contract')
 stages=[]; history=[]
 all_ids=['P0','P1','P1A','P1B','P2','P3','P3A']
 for sid in all_ids:
  a=put(root/f'evidence/{sid}-input',sid); b=put(root/f'evidence/{sid}-evidence',sid+'e')
  stages.append({'id':sid,'status':'passed','contract_version':'v1','gate_verdict':'pass','inputs':[{'path':a.relative_to(root).as_posix(),'sha256':sha(a)}],'evidence':[{'path':b.relative_to(root).as_posix(),'sha256':sha(b)}]}); history.append({'stage':sid,'action':'pass'})
 stages.append({'id':'P4','status':'running','contract_version':'v1','gate_verdict':'running','inputs':[],'evidence':[]})
 for sid in ['P5','P5A']+[f'P{i}' for i in range(6,22)]: stages.append({'id':sid,'status':'pending','contract_version':'v1','gate_verdict':'pending','inputs':[],'evidence':[]})
 state={'schema_version':1,'pipeline':'f-research-v1','contract_version':'v1','delivery_mode':'preparation','contract':[{'path':'contract.md','sha256':sha(root/'contract.md')}],'stages':stages,'history':history,'readiness':[],'citation_gate':[]}
 put(root/'pipeline-state.json',json.dumps(state))
 for n in ['ckpt','data.npz','cfg.json','queue.json','epsilon.json','binding.json','domain.json','recipe.json','policy.json']: put(root/n,'{}')
 (root/'out-parent').mkdir()
 args={
  'graph_v6':['--checkpoint','ckpt','--dataset-path','data.npz','--output','out-parent/graph','--policy','full','--policy-config','policy.json','--device','cpu'],
  'h2_v4':['--checkpoint','ckpt','--dataset-path','data.npz','--split','validation','--output','out-parent/h2','--device','cpu'],
  'calibration_v7':['--checkpoint','ckpt','--config','cfg.json','--queue','queue.json','--epsilon-evidence','epsilon.json','--binding','binding.json','--output','out-parent/cal','--device','cpu','--root',str(root)],
  'c06_v4':['--checkpoint','ckpt','--dataset-path','data.npz','--domain-manifest','domain.json','--recipe-json','recipe.json','--output','out-parent/c06','--split','validation','--policy','full','--policy-config','policy.json','--device','cpu','--batch-size','1']}
 job={'schema_version':1,'job_id':'mock-'+driver.replace('_','-'),'driver':driver,'argv':args[driver],'sources':[]}
 jp=put(root/'jobs/job.json',json.dumps(job)); monkeypatch.setattr(d,'TRUSTED_PYTHON',Path(sys.executable)); monkeypatch.setattr(d,'VALIDATOR',root/'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py')
 return root,jp

def child_runner(root,driver,mutation=None,omit_output=False):
 def run(cmd,**kw):
  assert kw['shell'] is False
  if Path(cmd[1]).name=='pipeline_state.py': return subprocess.run(cmd,**kw)
  assert Path(cmd[1]).name==d.DRIVERS[driver][0].split('/')[-1]
  args=cmd[3:]; output=Path(args[args.index('--output')+1]); output=output if output.is_absolute() else root/output
  if not omit_output:
   output.mkdir(parents=True,exist_ok=True)
   if driver=='graph_v6':
    put(output/'predictions.npz','pred'); put(output/'frozen-graph-state-diagnostic.json',json.dumps({'predictions_file':'predictions.npz','full_test_origin_count':3}))
   elif driver=='h2_v4': put(output/'h2-matched-retention-report.json','report'); put(output/'predictions-base.npz','pred')
   elif driver=='calibration_v7': put(output/'calibration.json','calibration'); put(output/'validation-predictions.npz','pred'); put(output/'calibration-validation-dependence-v2.json','report')
   elif driver=='c06_v4':
    for n in ['predictions.npz','metrics_by_horizon.csv','counter-rows.json','perturbation-manifest.json']: put(output/n,n)
  phase_dir=root/'logs/offline-phase-wall'; phase_dir.mkdir(parents=True,exist_ok=True); receipt=phase_dir/'child-mock.json'
  doc={'scope':'offline-experiment-phase','phase':'calibration' if driver=='calibration_v7' else 'evaluation','status':'completed','job_id':'childmock','output_dir':str(output.resolve())}
  if driver=='graph_v6': doc['origin_count']=3
  if driver=='h2_v4': doc['origin_count']=3; doc['report_sha256']=sha(output/'h2-matched-retention-report.json')
  if driver=='c06_v4': doc['origins_completed']=3; doc['outputs']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
  if driver=='calibration_v7': doc.update(input_hashes_match=True,input_sha256_before='same',input_sha256_after='same',parameter_hash_before='params',parameter_hash_after='params')
  put(receipt,json.dumps(doc))
  if mutation: mutation(root)
  return subprocess.CompletedProcess(cmd,0,'mocked child stdout','')
 return run

@pytest.mark.parametrize('driver',['graph_v6','h2_v4','calibration_v7','c06_v4'])
def test_all_schemas_accept_optional_flags_and_check_receipt(driver,tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,driver)
 result=d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,driver))
 assert result['status']=='accepted-output-pending-scientific-review'
 assert result['child_phase_receipt']['status']=='completed'

def test_missing_principal_artifact_rejected_even_with_zero_return(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'graph_v6')
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'graph_v6',omit_output=True))

def test_stopped_receipt_and_missing_receipt_rejected(tmp_path,monkeypatch):
 for suffix,mutation in [('stopped',lambda r: (r/'logs/offline-phase-wall/child-mock.json').write_text(json.dumps({'scope':'offline-experiment-phase','phase':'evaluation','status':'stopped','job_id':'x','output_dir':str((r/'out-parent/graph').resolve()),'origin_count':3}))),('no-receipt',lambda r:(r/'logs/offline-phase-wall/child-mock.json').unlink())]:
  root,job=setup(tmp_path/suffix,monkeypatch,'graph_v6')
  with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'graph_v6',mutation=mutation))

def test_missing_required_duplicate_and_extra_flags_fail_closed(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'h2_v4'); x=json.loads(job.read_text()); x['argv'].remove('--split'); x['argv'].remove('validation'); job.write_text(json.dumps(x))
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'h2_v4'))
 root,job=setup(tmp_path/'dup',monkeypatch,'h2_v4'); x=json.loads(job.read_text()); x['argv']+=['--device','cpu']; job.write_text(json.dumps(x))
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'h2_v4'))
 root,job=setup(tmp_path/'extra',monkeypatch,'h2_v4'); x=json.loads(job.read_text()); x['argv']+=['--unsafe','yes']; job.write_text(json.dumps(x))
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'h2_v4'))

def test_source_membership_drift_and_missing_v6_cal_driver_fail(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'c06_v4'); src=root/'bundle'; put(src/'a','a'); x=json.loads(job.read_text()); x['sources']=[{'path':'bundle','sha256':d.tree_digest(d.file_map(src,root))}]; job.write_text(json.dumps(x))
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'c06_v4',mutation=lambda r:put(r/'bundle/b','b')))
 assert d.DRIVERS['calibration_v7'][0]=='edsgn/staging/frozen_validation_calibration_v7.py'
 assert (RUN/d.DRIVERS['calibration_v7'][0]).is_file()

def test_dispatcher_not_owned_by_model_scanner():
 scanner=(RUN/'edsgn/scripts/candidate_gpu_fullpath_probe_v3.py').read_text(encoding='utf-8')
 assert 'verified_frozen_dispatch_v2.py' not in scanner
