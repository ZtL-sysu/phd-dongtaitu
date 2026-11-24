import hashlib, json, shutil, subprocess, sys
from pathlib import Path
import pytest

RUN=Path(r'C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn')
sys.path.insert(0,str(RUN/'edsgn/scripts'))
import verified_frozen_dispatch_v5 as d

def put(p,data='x'):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(data,encoding='utf-8'); return p
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def setup(tmp_path,monkeypatch,driver,split='test',with_binding=True):
 root=tmp_path/'run'; root.mkdir(parents=True)
 for rel in d.VALIDATOR_FILES:
  dst=root/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(RUN/rel,dst)
 dr=d.DRIVERS[driver][0]; put(root/dr,'# mocked driver')
 dispatcher='edsgn/scripts/verified_frozen_dispatch_v5.py'; put(root/dispatcher,(RUN/dispatcher).read_text(encoding='utf-8'))
 monkeypatch.setattr(d,'__file__',str(root/dispatcher))
 put(root/'contract.md','contract')
 stages=[]; history=[]
 for sid in ['P0','P1','P1A','P1B','P2','P3','P3A']:
  a=put(root/f'evidence/{sid}-input',sid); b=put(root/f'evidence/{sid}-evidence',sid+'e')
  stages.append({'id':sid,'status':'passed','contract_version':'v1','gate_verdict':'pass','inputs':[{'path':a.relative_to(root).as_posix(),'sha256':sha(a)}],'evidence':[{'path':b.relative_to(root).as_posix(),'sha256':sha(b)}]}); history.append({'stage':sid,'action':'pass'})
 stages += [{'id':'P4','status':'running','contract_version':'v1','gate_verdict':'running','inputs':[],'evidence':[]},*({'id':x,'status':'pending','contract_version':'v1','gate_verdict':'pending','inputs':[],'evidence':[]} for x in ['P5','P5A']+[f'P{i}' for i in range(6,22)])]
 put(root/'pipeline-state.json',json.dumps({'schema_version':1,'pipeline':'f-research-v1','contract_version':'v1','delivery_mode':'preparation','contract':[{'path':'contract.md','sha256':sha(root/'contract.md')}],'stages':stages,'history':history,'readiness':[],'citation_gate':[]}))
 for n in ['ckpt','data.npz','cfg.json','queue.json','epsilon.json','authority.json','launch.json','binding.json','freeze.json','domain.json','recipe.json','policy.json']: put(root/n,'{}')
 for parent in ['runs/out-parent','runs/run1/evidence']: (root/parent).mkdir(parents=True,exist_ok=True)
 args={
  'graph_v7':['--checkpoint','ckpt','--dataset-path','data.npz','--output','runs/out-parent/graph','--policy','full','--device','cpu'],
  'h2_v4':['--checkpoint','ckpt','--dataset-path','data.npz','--split',split,'--output','runs/out-parent/h2','--device','cpu'],
  'calibration_v9':['--checkpoint','ckpt','--config','cfg.json','--queue','queue.json','--epsilon-authority','authority.json','--launch-binding','launch.json','--binding','binding.json','--output','runs/out-parent/cal','--device','cpu','--root',str(root)],
  'c06_v5':['--checkpoint','ckpt','--dataset-path','data.npz','--domain-manifest','domain.json','--recipe-json','recipe.json','--output','runs/out-parent/c06','--split',split,'--policy','full','--device','cpu'],
  'prediction_v5':['--checkpoint','ckpt','--dataset-path','data.npz','--binding','binding.json','--output','runs/out-parent/pred','--split','test','--policy','full','--device','cpu','--root',str(root)],
  'profile_v5':['--checkpoint','ckpt','--dataset-path','data.npz','--freeze','freeze.json','--output','runs/out-parent/profile','--split',split,'--policy','full','--device','cpu']}
 bind=[]
 if with_binding:
  runs=[]; side_desc={}
  for side in ('left','right'):
   seeds=[]
   for seed in range(5):
    refs={}
    for role in ('config','training_binding','checkpoint','freeze','input_binding','validation_authority','policy','queue_plan'):
     refpath=f'pretest/{side}/seed{seed}/{role}.json'; ref=put(root/refpath,'bound-'+role)
     refs[role]={'path':refpath,'sha256':sha(ref)}
    prep={'config':refs['config'],'training_binding':refs['training_binding']}
    seeds.append({'seed_id':seed,'preparation':prep,'planned_paths':{}})
    runs.append({'side':side,'seed_id':seed,**refs,'block_length':28,'acf_dependence_length':2,
      'acf_authority_path':'plans/acf.json','epsilon_status':None,'epsilon_value_raw':None,
      'epsilon_authority_path':None,'test_feedback_used':False})
   side_desc[side]={'method_id':side,'producer_source_sha256':'a'*64,'seeds':seeds}
  acf=put(root/'plans/acf.json',json.dumps({'dependence_length':2,'history_length':28,'hmax':7,'selection_status':'accepted-validation-only'}))
  plan_doc={'schema_version':1,'endpoint_id':'H2-toy','claim':'H2','outcome_type':'forecast_mae_superiority',
   'history':28,'max_horizon':7,'block_length':28,'acf_dependence_length':2,
   'acf_evidence':{'mode':'fixed_validation_evidence','path':'plans/acf.json','sha256':sha(acf)},
   'epsilon_evidence':None,'domain':{'bounds':[{'origin_count':46}]},**side_desc}
  plan=put(root/'plans/endpoint.json',json.dumps(plan_doc))
  registry_rows=[{'endpoint_id':'H2-toy','claim':'H2','path':'plans/endpoint.json','sha256':sha(plan)}]
  for family,count in [('H1',9),('H2',15),('H3',24)]:
   for i in range(count):
    path=f'plans/registered/{family}-{i}.json'; item=put(root/path,'registered')
    registry_rows.append({'endpoint_id':f'{family}-fixture-{i}','claim':family,'path':path,'sha256':sha(item)})
  registry=put(root/'plans/registry.json',json.dumps({'schema_version':1,'artifact_status':'accepted_pretest_registry','split':'test','endpoint_plans':registry_rows}))
  doc={'schema_version':2,'status':'accepted_by_root','scope':'pretest_execution_binding',
   'endpoint_id':'H2-toy','plan_path':'plans/endpoint.json','plan_sha256':sha(plan),
   'registry_path':'plans/registry.json','registry_sha256':sha(registry),'test_feedback_used':False,'runs':runs}
  bp=put(root/'bindings/pretest.json',json.dumps(doc)); bind=[{'path':bp.relative_to(root).as_posix(),'sha256':sha(bp)}]
 job={'schema_version':2,'job_id':'mock-'+driver.replace('_','-'),'driver':driver,'argv':args[driver],'sources':[],'pretest_execution_bindings':bind,'phase_receipt_alias':'runs/run1/evidence/phase-receipt.json'}
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
   names={
    'graph_v7':['predictions.npz','counter-rows.json','frozen-graph-state-diagnostic.json'],
    'h2_v4':['h2-matched-retention-report.json','predictions-base.npz'],
    'calibration_v9':['calibration.json','validation-predictions.npz','calibration-validation-dependence-v2.json'],
    'c06_v5':['predictions.npz','metrics_by_horizon.csv','counter-rows.json','perturbation-manifest.json'],
    'prediction_v5':['predictions.npz','evaluation_manifest.json'],
    'profile_v5':['runtime_summary.json','window_timings.csv','trace.jsonl','profile-budget-accounting.json','profile-guard-overhead.json']}
   for n in names[driver]:
    if n=='validation-predictions.npz':
     import numpy as np
     np.savez_compressed(output/n,origins=np.asarray([[0,0],[0,1],[0,2]],dtype=np.int64))
     continue
    data={'frozen-graph-state-diagnostic.json':{'predictions_file':'predictions.npz','full_test_origin_count':3},
          'h2-matched-retention-report.json':{'origin_count':3},
          'calibration.json':{'validation_origin_count':3,'test_origins_evaluated':False,'test_feedback_used':False},
          'evaluation_manifest.json':{'n_windows':3,'split':'test'},
          'runtime_summary.json':{'scope':'full_selected_split','smoke_limit':None,'n_windows':3}}.get(n,n)
    put(output/n,json.dumps(data) if isinstance(data,dict) else data)
   if driver=='calibration_v9':
    predsha=sha(output/'validation-predictions.npz')
    dependence={'validation_origin_count':3,'test_origins_evaluated':False,'test_feedback_used':False,
      'validation_prediction_sha256':predsha,'test_accessed':False,'precision':'float32','batch_size':1}
    put(output/'calibration-validation-dependence-v2.json',json.dumps(dependence))
  phase_dir=root/'logs/offline-phase-wall'; phase_dir.mkdir(parents=True,exist_ok=True); receipt=phase_dir/'child-mock.json'
  phase={'calibration_v9':'calibration','profile_v5':'profiling'}.get(driver,'evaluation')
  doc={'scope':'offline-experiment-phase','phase':phase,'status':'completed','job_id':'childmock','output_dir':str(output.resolve())}
  if driver=='graph_v7': doc['origin_count']=3
  if driver=='h2_v4': doc['origin_count']=3; doc['report_sha256']=sha(output/'h2-matched-retention-report.json')
  if driver=='c06_v5': doc['origins_completed']=3; doc['outputs']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
  if driver=='calibration_v9': doc.update(input_hashes_match=True,input_sha256_before='same',input_sha256_after='same',parameter_hash_before='params',parameter_hash_after='params')
  if driver=='prediction_v5':
   doc.update(origin_count=3,test_accessed=True,test_feedback_used=False,parameters_sha256='p',input_hashes_match=True)
   doc['outputs']=[{'path':str((output/name).resolve()),'sha256':sha(p)} for name,p in sorted({p.name:p for p in output.iterdir() if p.is_file()}.items())]
  if driver=='profile_v5':
   doc.update(canonical_origin_count=3,repeats=5,warmup_windows=10,profile_parameter_hash_before='p',profile_parameter_hash_after='p')
   doc['output_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
  put(receipt,json.dumps(doc))
  if mutation: mutation(root)
  return subprocess.CompletedProcess(cmd,0,'mocked child stdout','')
 return run

@pytest.mark.parametrize('driver',[*d.DRIVERS])
def test_v4_supported_driver_fences_child_and_exclusively_aliases_receipt(driver,tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,driver)
 result=d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,driver))
 assert result['status']=='accepted-output-pending-scientific-review'
 alias=root/'runs/run1/evidence/phase-receipt.json'
 original=root/result['phase_receipt_original_path']
 assert alias.read_bytes()==original.read_bytes()
 assert result['phase_receipt_alias_sha256']==result['phase_receipt_original_sha256']==sha(alias)
 assert result['child_phase_receipt']['status']=='completed'

def test_validation_h2_may_have_empty_pretest_bindings_but_alias_remains_required(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'h2_v4',split='validation',with_binding=False)
 result=d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'h2_v4'))
 assert result['status']=='accepted-output-pending-scientific-review'

def test_test_job_rejects_missing_binding_and_schema1(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'graph_v7',with_binding=False)
 with pytest.raises(d.DispatchError,match='requires at least one'):
  d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'graph_v7'))
 root,job=setup(tmp_path/'schema',monkeypatch,'graph_v7')
 doc=json.loads(job.read_text());doc['schema_version']=1;job.write_text(json.dumps(doc))
 with pytest.raises(d.DispatchError,match='schema invalid'):
  d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'graph_v7'))

def test_binding_future_test_reference_rejected_and_alias_must_be_absent(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'profile_v5')
 bind=root/'bindings/pretest.json';doc=json.loads(bind.read_text());doc['runs'][0]['future_test_prediction_path']='runs/test/predictions.npz';bind.write_text(json.dumps(doc))
 j=json.loads(job.read_text());j['pretest_execution_bindings'][0]['sha256']=sha(bind);job.write_text(json.dumps(j))
 with pytest.raises(d.DispatchError,match='malformed or contains post-test'):
  d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'profile_v5'))
 root,job=setup(tmp_path/'alias',monkeypatch,'graph_v7')
 put(root/'runs/run1/evidence/phase-receipt.json','preexisting')
 with pytest.raises(d.DispatchError,match='already exists'):
  d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'graph_v7'))

def test_failed_child_never_creates_alias(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'graph_v7')
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'graph_v7',omit_output=True))
 assert not (root/'runs/run1/evidence/phase-receipt.json').exists()

def test_child_receipt_mutation_after_alias_copy_is_rejected(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'graph_v7')
 original=d._verify_child_completion; calls=[]
 def mutate_on_final_read(driver,output,phase_before,phase_dir,root_arg):
  calls.append(1)
  if len(calls)==4:
   receipt=root/'logs/offline-phase-wall/child-mock.json';doc=json.loads(receipt.read_text());doc['origin_count']=4;receipt.write_text(json.dumps(doc))
  return original(driver,output,phase_before,phase_dir,root_arg)
 monkeypatch.setattr(d,'_verify_child_completion',mutate_on_final_read)
 with pytest.raises(d.DispatchError,match='report/receipt mismatch|principal report/receipt mismatch'):
  d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'graph_v7'))
 assert (root/'runs/run1/evidence/phase-receipt.json').is_file()
 result=json.loads(next((root/'logs/verified-frozen-dispatch').glob('*.json')).read_text())
 assert result['status']=='failed'

def test_mutated_pretest_binding_during_child_fails_snapshot_and_leaves_no_alias(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'prediction_v5')
 bind=root/'bindings/pretest.json'
 def change_binding(r): bind.write_text(bind.read_text()+' ',encoding='utf-8')
 with pytest.raises(d.DispatchError,match='snapshot|declared source hash mismatch'):
  d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'prediction_v5',mutation=change_binding))
 assert not (root/'runs/run1/evidence/phase-receipt.json').exists()


def test_v5_pretest_binding_nested_refs_are_hashed_into_before_snapshot(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'prediction_v5')
 result=d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'prediction_v5'))
 before=result['before']['files']
 assert 'plans/endpoint.json' in before and 'plans/registry.json' in before
 assert 'pretest/left/seed0/config.json' in before
 assert 'pretest/right/seed4/queue_plan.json' in before

def test_v5_pretest_nested_reference_hash_mismatch_rejected_before_child(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'graph_v7')
 binding=root/'bindings/pretest.json'; doc=json.loads(binding.read_text())
 config_path=root/doc['runs'][0]['config']['path']; put(config_path,'tampered-after-binding')
 jobdoc=json.loads(job.read_text()); jobdoc['pretest_execution_bindings'][0]['sha256']=sha(binding); put(job,json.dumps(jobdoc))
 with pytest.raises(d.DispatchError,match='config SHA-256 mismatch'):
  d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'graph_v7'))
 assert not (root/'logs/offline-phase-wall/child-mock.json').exists()


def test_v5_calibration_requires_dependence_sidecar_integrity(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'calibration_v9')
 def corrupt_sidecar(r):
  out=r/'runs/out-parent/cal'; side=out/'calibration-validation-dependence-v2.json'
  doc=json.loads(side.read_text()); doc['test_origins_evaluated']=True; put(side,json.dumps(doc))
 with pytest.raises(d.DispatchError,match='test feedback/access'):
  d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'calibration_v9',mutation=corrupt_sidecar))
 assert not (root/'runs/run1/evidence/phase-receipt.json').exists()

def test_v5_binding_requires_exact_typed_v4_top_and_run_keys(tmp_path,monkeypatch):
 root,job=setup(tmp_path,monkeypatch,'graph_v7')
 binding=root/'bindings/pretest.json'; doc=json.loads(binding.read_text()); doc['runs'][9].pop('queue_plan')
 put(binding,json.dumps(doc)); jobdoc=json.loads(job.read_text()); jobdoc['pretest_execution_bindings'][0]['sha256']=sha(binding); put(job,json.dumps(jobdoc))
 with pytest.raises(d.DispatchError,match='malformed or contains post-test'):
  d.dispatch(job,root=root,python=Path(sys.executable),runner=child_runner(root,'graph_v7'))
