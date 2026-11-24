import csv, json, shutil, sys
from pathlib import Path
import pytest, torch, numpy as np
RUN=Path(r'C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn')
sys.path.insert(0,str(RUN/'edsgn/staging'))
import frozen_profile_guarded_v2 as g
from edsgn.model import ForecastModel, ModelConfig

class TinyDataset:
 def __init__(self):
  self.values=np.zeros((4,3,1),np.float32); self.history=2; self.hmax=1; self.count_data=False; self.event_kind='synthetic'
 def origins(self,split): return np.asarray([1,2],dtype=np.int64)
 def window(self,origin):
  vals=np.asarray([[origin,origin+1,origin+2],[origin+1,origin+2,origin+3]],np.float32)[...,None]
  return {'values':vals,'input_mask':np.ones_like(vals,dtype=bool),'obs_age':np.zeros_like(vals),
          'raw_observations':vals.copy(),'events':np.empty((0,3),np.int64),'time_indices':np.asarray([origin-1,origin]),
          'interval_end':np.asarray([origin-1,origin],dtype=np.float64)}

def write(p,s): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s,encoding='utf-8'); return p
def digest(p):
 import hashlib
 return hashlib.sha256(p.read_bytes()).hexdigest()

def fixture(tmp_path,monkeypatch):
 root=tmp_path/'run';root.mkdir(parents=True)
 # Source bytes are copied exactly; Python bytecode cache is not part of code membership.
 shutil.copytree(RUN/'edsgn/src/edsgn',root/'edsgn/src/edsgn')
 for rel in g.VALIDATOR_HASHES:
  dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(RUN/rel,dst)
 for rel in ['edsgn/scripts/aggregate_budget.py','edsgn/scripts/candidate_gpu_fullpath_probe_v3.py']:
  dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(RUN/rel,dst)
 wrapper='edsgn/staging/frozen_profile_guarded_v2.py';write(root/wrapper,(RUN/wrapper).read_text(encoding='utf-8'))
 monkeypatch.setattr(g,'__file__',str(root/wrapper));monkeypatch.setattr(g,'ROOT',root)
 # Tiny fully prepared collection and immutable fixture checkpoint/freeze.
 datadir=root/'data';write(datadir/'prepared.npz','fixture-only dataset fingerprint bytes')
 freeze=write(root/'freeze.json','{}')
 model=ForecastModel(ModelConfig(n_nodes=3,n_features=1,history=2,hmax=1))
 config={'variant':'V4','model':{},'dataset_path':str(datadir)}
 stats={'norm_mean':[0.],'norm_std':[1.],'residual_scale':[1.],'output_scale':[1.],'output_mean':[0.]}
 ckpt=root/'checkpoint.pt';torch.save({'config':config,'task':{'n_nodes':3,'n_features':1,'history':2,'hmax':1,'count_data':False},'statistics':stats,'state_dict':model.state_dict()},ckpt)
 write(root/'contract.md','contract'); stages=[];history=[]
 for sid in ['P0','P1','P1A','P1B','P2','P3','P3A']:
  inp=write(root/f'ev/{sid}-in',sid);ev=write(root/f'ev/{sid}-out',sid+'e')
  stages.append({'id':sid,'status':'passed','contract_version':'v1','gate_verdict':'pass','inputs':[{'path':inp.relative_to(root).as_posix(),'sha256':digest(inp)}],'evidence':[{'path':ev.relative_to(root).as_posix(),'sha256':digest(ev)}]});history.append({'stage':sid,'action':'pass'})
 stages.append({'id':'P4','status':'running','contract_version':'v1','gate_verdict':'running','inputs':[],'evidence':[]})
 for sid in ['P5','P5A']+[f'P{i}' for i in range(6,22)]:stages.append({'id':sid,'status':'pending','contract_version':'v1','gate_verdict':'pending','inputs':[],'evidence':[]})
 state={'schema_version':1,'pipeline':'f-research-v1','contract_version':'v1','delivery_mode':'preparation','contract':[{'path':'contract.md','sha256':digest(root/'contract.md')}],'stages':stages,'history':history,'readiness':[],'citation_gate':[]}
 write(root/'pipeline-state.json',json.dumps(state))
 outparent=root/'runs';outparent.mkdir()
 import edsgn.profile as prof
 from edsgn.engine import datasets_at as original_datasets_at
 from edsgn.prepare import write_bundle
 raw=np.arange(5*3,dtype=np.float32).reshape(5,3,1)
 bundle=datadir/'prepared.npz'
 write_bundle(bundle,raw_values=raw,input_mask=np.ones_like(raw,dtype=bool),interval_end=np.arange(5,dtype=np.float64),
              node_ids=np.arange(3),events=np.empty((0,3),np.int64),mean=np.array([0.]),std=np.array([1.]),scale=np.ones((3,1)),
              count_data=False,horizons=(1,),history=2,hmax=1,splits={'train':(0,2),'validation':(2,5),'test':(0,0)},
              event_kind='synthetic',metadata={'source':'CPU fixture only'})
 config['dataset_path']=str(bundle)
 torch.save({'config':config,'task':{'n_nodes':3,'n_features':1,'history':2,'hmax':1,'count_data':False},'statistics':stats,'state_dict':model.state_dict()},ckpt)
 def datasets_at(path,split):
  assert (root/'logs/model-resource.lock').is_file()
  assert (root/'logs/offline-phase-wall/campaign.lock').is_file()
  return original_datasets_at(path,split)
 monkeypatch.setattr(prof,'datasets_at',datasets_at)
 monkeypatch.setattr(g,'PYTHON',Path(sys.executable))
 def runner(cmd,**kwargs):return __import__('subprocess').run(cmd,**kwargs)
 def agg(*args,**kwargs):return {'coverage_complete':True,'spent_seconds':1.0,'budget_seconds':g.WALL_BUDGET_S}
 deps={'python':Path(sys.executable),'runner':runner,'aggregate':agg,'owned':lambda root:[]}
 return root,ckpt,bundle,freeze,outparent,prof,deps

def test_full_cpu_profile_uses_real_profile_run_and_corrects_guard_overhead(tmp_path,monkeypatch):
 root,ckpt,data,freeze,parent,prof,deps=fixture(tmp_path,monkeypatch)
 result=g.main_profile(ckpt,data,freeze,parent/'out','val','full',None,'cpu',root=root,deps=deps)
 assert result['status']=='completed'; assert result['summary']['scope']=='full_selected_split'
 assert result['summary']['repeats']==5 and result['summary']['warmup_windows']==10
 assert result['summary']['n_windows']==3
 out=parent/'out'; rows=(out/'window_timings.csv').read_text(encoding='utf-8').splitlines(); assert len(rows)==16
 assert 'guard_overhead_ms' in rows[0]
 overhead=json.loads((out/'profile-guard-overhead.json').read_text()); assert overhead['guard_calls_total']==28
 assert overhead['measured_calls']==15 and overhead['warmup_calls']==10 and overhead['diagnostic_calls']==3
 assert 'preprocessing_ms_per_window_including_guard' in result['summary']
 receipt=json.loads(Path(result['receipt']).read_text()); assert receipt['status']=='completed'; assert receipt['phase']=='profiling'
 assert not (root/'logs/model-resource.lock').exists() and not (root/'logs/offline-phase-wall/campaign.lock').exists()

def test_final_gate_mutation_fails_with_failed_receipt_and_releases_locks(tmp_path,monkeypatch):
 root,ckpt,data,freeze,parent,prof,deps=fixture(tmp_path,monkeypatch)
 original=prof.profile_run
 def mutate(*args,**kwargs):
  result=original(*args,**kwargs);(root/'ev/P3A-out').write_text('mutated',encoding='utf-8');return result
 monkeypatch.setattr(prof,'profile_run',mutate)
 with pytest.raises(g.GuardError,match='stale P3A evidence hash'):
  g.main_profile(ckpt,data,freeze,parent/'mutated','val','full',None,'cpu',root=root,deps=deps)
 rec=list((root/'logs/offline-phase-wall').glob('frozen-profile-*.json'))[0];assert json.loads(rec.read_text())['status']=='failed'
 assert (parent/'mutated/runtime_summary.json').exists()
 assert not (root/'logs/model-resource.lock').exists() and not (root/'logs/offline-phase-wall/campaign.lock').exists()

def test_final_budget_guard_failure_never_marks_completed(tmp_path,monkeypatch):
 root,ckpt,data,freeze,parent,prof,deps=fixture(tmp_path,monkeypatch);calls={'n':0}
 def aggregate(*args,**kwargs):
  calls['n']+=1
  spent=g.WALL_BUDGET_S if calls['n']>=24 else 1
  return {'coverage_complete':True,'spent_seconds':spent,'budget_seconds':g.WALL_BUDGET_S}
 deps['aggregate']=aggregate
 with pytest.raises(g.GuardError,match='72-hour'):
  g.main_profile(ckpt,data,freeze,parent/'budget','val','full',None,'cpu',root=root,deps=deps)
 rec=list((root/'logs/offline-phase-wall').glob('frozen-profile-*.json'))[0];assert json.loads(rec.read_text())['status']=='failed'
 assert not (root/'logs/model-resource.lock').exists() and not (root/'logs/offline-phase-wall/campaign.lock').exists()

def test_each_repeat_must_preserve_canonical_origin_order(tmp_path,monkeypatch):
 root,ckpt,data,freeze,parent,prof,deps=fixture(tmp_path,monkeypatch)
 original=prof.profile_run
 def reorder_repeat(*args,**kwargs):
  result=original(*args,**kwargs)
  path=Path(args[3])/'window_timings.csv'
  with path.open('r',newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f)); fields=list(rows[0])
  inds=[i for i,r in enumerate(rows) if int(r['repeat'])==2]
  rows[inds[0]],rows[inds[1]]=rows[inds[1]],rows[inds[0]]
  with path.open('w',newline='',encoding='utf-8') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
  return result
 monkeypatch.setattr(prof,'profile_run',reorder_repeat)
 with pytest.raises(g.GuardError,match='repeat 2 does not preserve'):
  g.main_profile(ckpt,data,freeze,parent/'reordered','val','full',None,'cpu',root=root,deps=deps)
 rec=list((root/'logs/offline-phase-wall').glob('frozen-profile-*.json'))[0]
 assert json.loads(rec.read_text())['status']=='failed'
 assert not (root/'logs/model-resource.lock').exists()


def _install_toy_blind_acceptance(root,freeze):
 import importlib.util
 helper_path=root/g.BLIND_HELPER_REL;helper_path.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(RUN/'edsgn/staging/blind_reference_cost_policy_v1.py',helper_path)
 spec=importlib.util.spec_from_file_location('toy_blind_helper',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 freeze.write_text(json.dumps({'extra':helper.freeze_extra_fields()}),encoding='utf-8')
 acceptance=root/g.BLIND_ACCEPTANCE_REL;acceptance.parent.mkdir(parents=True,exist_ok=True)
 acceptance.write_text(json.dumps({'accepted_by':'root','verdict':'pass','scope':'engineering-staging-only',
     'pipeline_advanced':False,'scientific_hypothesis_accepted':False,'actual_blind_control_run':False,
     'c07_experiment_closed':False,'evidence_sha256':{g.BLIND_HELPER_REL:helper.descriptor()['source_sha256']}}),encoding='utf-8')
 return helper

def test_periodic_control_requires_accepted_freeze_binding_and_records_descriptor(tmp_path,monkeypatch):
 root,ckpt,data,freeze,parent,prof,deps=fixture(tmp_path,monkeypatch)
 config={'kind':'periodic','graph_fraction':.5,'state_fraction':.5,'max_graph_age':8,'max_state_age':8,'audit_interval':0,'seed':7,'force_initialization':False,
         'cost_table':{'graph':[[1,1.0],[4,4.0]],'state':[[1,1.0],[4,4.0]],'base_ms':.25,'batch_size':1,'units':'toy'},'runtime_budget_ms':4.25}
 cfg=root/'policy.json';cfg.write_text(json.dumps(config),encoding='utf-8')
 helper=_install_toy_blind_acceptance(root,freeze)
 acceptance=root/g.BLIND_ACCEPTANCE_REL;acceptance.unlink()
 # A valid source/rule freeze alone is not root acceptance.
 with pytest.raises(g.GuardError,match='lacks independent root acceptance'):
  g.main_profile(ckpt,data,freeze,parent/'denied','val','periodic',cfg,'cpu',root=root,deps=deps)
 acceptance.write_text(json.dumps({'accepted_by':'root','verdict':'pass','scope':'engineering-staging-only',
     'pipeline_advanced':False,'scientific_hypothesis_accepted':False,'actual_blind_control_run':False,
     'c07_experiment_closed':False,'evidence_sha256':{g.BLIND_HELPER_REL:helper.descriptor()['source_sha256']}}),encoding='utf-8')
 result=g.main_profile(ckpt,data,freeze,parent/'accepted','val','periodic',cfg,'cpu',root=root,deps=deps)
 summary=json.loads((parent/'accepted/runtime_summary.json').read_text())
 assert summary['blind_reference_cost_policy']['descriptor']['source_sha256']==helper.descriptor()['source_sha256']
 assert summary['blind_reference_cost_policy']['frozen_binding']['rule_version']==helper.RULE_VERSION
 assert 'preprocessing_ms_per_window_including_guard' in summary
 assert result['status']=='completed'


