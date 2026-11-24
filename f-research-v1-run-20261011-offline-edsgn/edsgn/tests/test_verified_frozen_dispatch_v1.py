import hashlib, json, shutil, subprocess, sys
from pathlib import Path
import pytest

RUN=Path(r'C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn')
sys.path.insert(0,str(RUN/'edsgn/scripts'))
import verified_frozen_dispatch_v1 as d


def put(p, data='x'):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(data,encoding='utf-8'); return p

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def fixture(tmp_path, monkeypatch):
 root=tmp_path/'run'; root.mkdir(parents=True)
 # Copy the exact pinned official validator bundle and the allowlisted driver.
 for rel in d.VALIDATOR_FILES:
  src=RUN/rel; dst=root/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
 driver_rel=d.DRIVERS['h2_v4'][0]; put(root/driver_rel,'# allowlisted inert mock driver\n')
 dispatcher_rel='edsgn/scripts/verified_frozen_dispatch_v1.py'; put(root/dispatcher_rel,(RUN/dispatcher_rel).read_text(encoding='utf-8'))
 monkeypatch.setattr(d,'__file__',str(root/dispatcher_rel))
 put(root/'contract.md','contract')
 stages=[]; history=[]
 for sid in ['P0','P1','P1A','P1B','P2','P3','P3A']:
  ip=put(root/f'evidence/{sid}-in.txt',sid+' input'); ep=put(root/f'evidence/{sid}-out.txt',sid+' evidence')
  stages.append({'id':sid,'status':'passed','contract_version':'v1','gate_verdict':'pass','inputs':[{'path':ip.relative_to(root).as_posix(),'sha256':digest(ip)}],'evidence':[{'path':ep.relative_to(root).as_posix(),'sha256':digest(ep)}]})
  history.append({'stage':sid,'action':'pass'})
 stages.append({'id':'P4','status':'running','contract_version':'v1','gate_verdict':'running','inputs':[],'evidence':[]})
 for sid in ['P5','P5A']+[f'P{i}' for i in range(6,22)]: stages.append({'id':sid,'status':'pending','contract_version':'v1','gate_verdict':'pending','inputs':[],'evidence':[]})
 state={'schema_version':1,'pipeline':'f-research-v1','contract_version':'v1','delivery_mode':'preparation','contract':[{'path':'contract.md','sha256':digest(root/'contract.md')}],'stages':stages,'history':history,'readiness':[],'citation_gate':[]}
 (root/'pipeline-state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
 put(root/'ckpt.bin','ckpt'); put(root/'dataset.npz','dataset'); (root/'out').mkdir()
 job={'schema_version':1,'job_id':'toy-job','driver':'h2_v4','argv':['--checkpoint','ckpt.bin','--dataset-path','dataset.npz','--split','validation','--output','out/result.json'],'sources':[]}
 jobp=put(root/'jobs/job.json',json.dumps(job))
 monkeypatch.setattr(d,'TRUSTED_PYTHON',Path(sys.executable))
 return root,jobp

def runner_factory(root, mutation=None, rc=0):
 def runner(cmd,**kwargs):
  assert kwargs['shell'] is False
  if Path(cmd[1]).name=='pipeline_state.py':
   return subprocess.run(cmd,**kwargs)
  assert Path(cmd[1]).name=='frozen_h2_matched_retention_v4.py'
  # Simulate output production and optional source drift.
  (root/'out/result.json').write_text('{}',encoding='utf-8')
  if mutation: mutation(root)
  return subprocess.CompletedProcess(cmd,rc,'mock child output','')
 return runner

def test_success_requires_both_validator_passes_and_returns_pending_review(tmp_path,monkeypatch):
 root,job=fixture(tmp_path,monkeypatch); calls=[]
 def runner(cmd,**kwargs):
  calls.append(Path(cmd[1]).name)
  return runner_factory(root)(cmd,**kwargs)
 out=d.dispatch(job,root=root,python=Path(sys.executable),runner=runner)
 assert out['status']=='accepted-output-pending-scientific-review'
 assert calls.count('pipeline_state.py')==2
 assert json.loads((root/'logs/verified-frozen-dispatch/toy-job.json').read_text())['status']==out['status']

def test_stale_passed_p2_evidence_fails_before_child(tmp_path,monkeypatch):
 root,job=fixture(tmp_path,monkeypatch); ep=root/'evidence/P2-out.txt'; ep.write_text('stale')
 called=[]
 def runner(cmd,**kwargs):
  called.append(Path(cmd[1]).name)
  return subprocess.run(cmd,**kwargs)
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=runner)
 assert called==[]

def test_post_child_passed_evidence_change_rejected(tmp_path,monkeypatch):
 root,job=fixture(tmp_path,monkeypatch)
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=runner_factory(root,lambda r:(r/'evidence/P3-out.txt').write_text('changed')))
 assert json.loads((root/'logs/verified-frozen-dispatch/toy-job.json').read_text())['status']=='failed'

def test_directory_membership_change_rejected(tmp_path,monkeypatch):
 root,job=fixture(tmp_path,monkeypatch); src=root/'bundle'; put(src/'a.txt','a')
 jd=json.loads(job.read_text()); jd['sources']=[{'path':'bundle','sha256':d.tree_digest(d.file_map(src,root))}]; job.write_text(json.dumps(jd))
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=runner_factory(root,lambda r:put(r/'bundle/new.txt','new')))

def test_bad_stage_and_unapproved_or_outside_argv_rejected(tmp_path,monkeypatch):
 root,job=fixture(tmp_path,monkeypatch); st=json.loads((root/'pipeline-state.json').read_text()); st['stages'][4]['gate_verdict']='fail'; (root/'pipeline-state.json').write_text(json.dumps(st))
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=runner_factory(root))
 root,job=fixture(tmp_path/'two',monkeypatch); jd=json.loads(job.read_text()); jd['argv'][1]='..\\outside.ckpt'; job.write_text(json.dumps(jd))
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=runner_factory(root))
 root,job=fixture(tmp_path/'three',monkeypatch); jd=json.loads(job.read_text()); jd['argv'].extend(['--arbitrary','yes']); job.write_text(json.dumps(jd))
 with pytest.raises(d.DispatchError): d.dispatch(job,root=root,python=Path(sys.executable),runner=runner_factory(root))

def test_dispatcher_is_not_owned_model_process(tmp_path):
 scanner=(RUN/'edsgn/scripts/candidate_gpu_fullpath_probe_v3.py').read_text(encoding='utf-8')
 assert 'verified_frozen_dispatch_v1.py' not in scanner
