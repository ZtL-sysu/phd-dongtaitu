from __future__ import annotations
import hashlib, json, shutil, sys
from pathlib import Path
import pytest

RUN=Path(r'C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn')
sys.path.insert(0,str(RUN/'edsgn/scripts'))
sys.path.insert(0,str(RUN/'edsgn/tests'))
import verified_frozen_dispatch_v7 as d
import test_verified_frozen_dispatch_v6 as fixtures
fixtures.d=d

def put(path,text):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8');return path
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def setup_v7(tmp_path,monkeypatch,driver='profile_v5',*,device='cpu',producer_driver=None,authority=False):
 root,jobpath=fixtures.setup(tmp_path,monkeypatch,driver)
 script_rel='edsgn/scripts/verified_frozen_dispatch_v7.py'
 put(root/script_rel,(RUN/script_rel).read_text(encoding='utf-8'))
 monkeypatch.setattr(d,'__file__',str(root/script_rel))
 planpath=root/'plans/endpoint.json';plan=json.loads(planpath.read_text())
 bound_driver=producer_driver or driver;source_rel=d.DRIVERS[bound_driver][0]
 source=root/source_rel
 if not source.exists():
  put(source,(RUN/source_rel).read_text(encoding='utf-8'))
 source_row={'path':source_rel,'sha256':sha(source)}
 plan['producer_sources']=[source_row]
 plan['profile_device']=device if plan.get('outcome_type')=='paired_window_cost_superiority' else None
 for side in ('left','right'):plan[side]['producer_source_sha256']=source_row['sha256']
 if authority:
  switch_source=put(root/'switch-source.npz','opaque-report-only-fixture')
  switch_doc={'schema':'edsgn.reporting-switch-authority.v1','scope':'reporting_only_postrun',
              'dataset_id':'fixture','join_rule':'trajectory_index_exact_one',
              'entries':[{'trajectory_index':0,'path':'switch-source.npz','sha256':sha(switch_source),'field':'switch_index'}]}
  switch_path=put(root/'switch-authority.json',json.dumps(switch_doc))
  plan['reporting_switch_authority']={'path':'switch-authority.json','sha256':sha(switch_path)}
 plan_sha=put(planpath,json.dumps(plan))
 registry_path=root/'plans/registry.json';registry=json.loads(registry_path.read_text())
 for row in registry['endpoint_plans']:
  if row['endpoint_id']==plan['endpoint_id']:row['sha256']=sha(planpath)
 put(registry_path,json.dumps(registry))
 binding_path=root/'bindings/pretest.json';binding=json.loads(binding_path.read_text())
 binding['plan_sha256']=sha(planpath);binding['registry_sha256']=sha(registry_path)
 put(binding_path,json.dumps(binding))
 job=json.loads(jobpath.read_text());job['sources']=[source_row]
 job['pretest_execution_bindings'][0]['sha256']=sha(binding_path)
 put(jobpath,json.dumps(job))
 return root,jobpath

def dispatch_mock(root,jobpath,driver):
 return d.dispatch(jobpath,root=root,python=Path(sys.executable),runner=fixtures.child_runner(root,driver))

def test_v7_rejects_plan_bound_source_from_different_driver_before_child(tmp_path,monkeypatch):
 root,jobpath=setup_v7(tmp_path,monkeypatch,'profile_v5',producer_driver='graph_v7')
 called=[];base=fixtures.child_runner(root,'profile_v5')
 def runner(cmd,**kwargs):
  if Path(cmd[1]).name==d.DRIVERS['profile_v5'][0].split('/')[-1]:called.append(True)
  return base(cmd,**kwargs)
 with pytest.raises(d.DispatchError):d.dispatch(jobpath,root=root,python=Path(sys.executable),runner=runner)
 assert not called

def test_v7_rejects_wrong_declared_driver_source_before_child(tmp_path,monkeypatch):
 root,jobpath=setup_v7(tmp_path,monkeypatch,'graph_v7')
 job=json.loads(jobpath.read_text());other=d.DRIVERS['profile_v5'][0]
 other_file=put(root/other,(RUN/other).read_text(encoding='utf-8'))
 job['sources']=[{'path':other,'sha256':sha(other_file)}];put(jobpath,json.dumps(job))
 called=[];base=fixtures.child_runner(root,'graph_v7')
 def runner(cmd,**kwargs):
  if Path(cmd[1]).name==d.DRIVERS['graph_v7'][0].split('/')[-1]:called.append(True)
  return base(cmd,**kwargs)
 with pytest.raises(d.DispatchError):d.dispatch(jobpath,root=root,python=Path(sys.executable),runner=runner)
 assert not called

def test_v7_h3_profile_device_must_match_predeclared_device(tmp_path,monkeypatch):
 root,jobpath=setup_v7(tmp_path,monkeypatch,'profile_v5',device='cuda')
 job=json.loads(jobpath.read_text());argv=job['argv'];argv[argv.index('--device')+1]='cpu';put(jobpath,json.dumps(job))
 called=[];base=fixtures.child_runner(root,'profile_v5')
 def runner(cmd,**kwargs):
  if Path(cmd[1]).name==d.DRIVERS['profile_v5'][0].split('/')[-1]:called.append(True)
  return base(cmd,**kwargs)
 with pytest.raises(d.DispatchError):d.dispatch(jobpath,root=root,python=Path(sys.executable),runner=runner)
 assert not called

def test_v7_h3_cuda_cli_matches_predeclared_cuda(tmp_path,monkeypatch):
 root,jobpath=setup_v7(tmp_path,monkeypatch,'profile_v5',device='cuda')
 job=json.loads(jobpath.read_text());job['argv'][job['argv'].index('--device')+1]='cuda';put(jobpath,json.dumps(job))
 result=dispatch_mock(root,jobpath,'profile_v5')
 assert result['status']=='accepted-output-pending-scientific-review'

@pytest.mark.parametrize('driver',['profile_v5','prediction_v5'])
def test_v7_rejects_cuda_zero_cli_before_child(tmp_path,monkeypatch,driver):
 root,jobpath=setup_v7(tmp_path,monkeypatch,driver,device='cuda')
 job=json.loads(jobpath.read_text());argv=job['argv']
 if '--device' in argv:argv[argv.index('--device')+1]='cuda:0'
 else:argv.extend(['--device','cuda:0'])
 put(jobpath,json.dumps(job))
 called=[]
 def runner(cmd,**kwargs):called.append(cmd);raise AssertionError('child must not launch')
 with pytest.raises(d.DispatchError,match='CLI accepts only'):d.dispatch(jobpath,root=root,python=Path(sys.executable),runner=runner)
 assert not called

def test_v7_profile_requires_device_before_child(tmp_path,monkeypatch):
 root,jobpath=setup_v7(tmp_path,monkeypatch,'profile_v5')
 job=json.loads(jobpath.read_text());argv=job['argv'];i=argv.index('--device');del argv[i:i+2];put(jobpath,json.dumps(job))
 with pytest.raises(d.DispatchError,match='required flags missing'):d._argv(job,root)

def test_v7_h3_different_cuda_index_rejected_before_child(tmp_path,monkeypatch):
 root,jobpath=setup_v7(tmp_path,monkeypatch,'profile_v5',device='cuda')
 job=json.loads(jobpath.read_text());job['argv'][job['argv'].index('--device')+1]='cuda:1';put(jobpath,json.dumps(job))
 called=[];base=fixtures.child_runner(root,'profile_v5')
 def runner(cmd,**kwargs):
  if Path(cmd[1]).name==d.DRIVERS['profile_v5'][0].split('/')[-1]:called.append(True)
  return base(cmd,**kwargs)
 with pytest.raises(d.DispatchError):d.dispatch(jobpath,root=root,python=Path(sys.executable),runner=runner)
 assert not called

def test_v7_report_only_switch_authority_and_npz_are_hashed_not_passed(tmp_path,monkeypatch):
 root,jobpath=setup_v7(tmp_path,monkeypatch,'graph_v7',authority=True)
 job=json.loads(jobpath.read_text())
 assert all('switch-authority.json' not in x and 'switch-source.npz' not in x for x in job['argv'])
 result=dispatch_mock(root,jobpath,'graph_v7')
 before=result['before']['files']
 assert 'switch-authority.json' in before and 'switch-source.npz' in before
