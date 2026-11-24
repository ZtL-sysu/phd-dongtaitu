from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, subprocess, sys, time, uuid
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PYTHON=Path(r'C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe')
PROFILE_SOURCE=ROOT/'edsgn/src/edsgn/profile.py'
VALIDATOR=ROOT/'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py'
VALIDATOR_HASHES={
 'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py':'5e88f56801955ded32e00fbc1b916ba4e9237b4c3eb76d70ac4c2cc6fba62525',
 'edsgn/scripts/frozen-pipeline-validator/scripts/citation_requirements.py':'efe945b9d14e9a24b9b18d2d2a9b318d4afff8b9d0683391ea6141017e46a069',
 'edsgn/scripts/frozen-pipeline-validator/scripts/delivery_readiness.py':'92ec18db25fb392ff55c9d164d89ffd0d3572de17c4e0622b515cc7ab5871988',
 'edsgn/scripts/frozen-pipeline-validator/assets/pipeline-definition.json':'1d4aadcea7a2e6eaaf24825c08dcec1433c79a3e6b6323d1424861fee7005e6e'}
WALL_BUDGET_S=72*60*60
BLIND_HELPER_REL='edsgn/staging/blind_reference_cost_policy_v1.py'
BLIND_ACCEPTANCE_REL='reviews/blind-reference-cost-policy-root-acceptance-v1.json'
NORMALIZER_HELPER_REL='edsgn/scripts/frozen_synthetic_stress.py'

class GuardError(RuntimeError): pass

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def strict_pairs(pairs):
 d={}
 for k,v in pairs:
  if k in d: raise GuardError('duplicate JSON key '+k)
  d[k]=v
 return d

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=strict_pairs)

def contained(root,raw,exists=True):
 p=Path(raw); p=(root/p).resolve() if not p.is_absolute() else p.resolve()
 if not p.is_relative_to(root.resolve()):raise GuardError('path outside run root: '+str(raw))
 if exists and not p.exists():raise GuardError('missing required path: '+str(p))
 return p

def file_map(root,path,source_py_only=False):
 path=Path(path)
 if path.is_symlink():raise GuardError('symlinks forbidden in frozen inputs: '+str(path))
 if path.is_file():return {path.relative_to(root).as_posix():sha(path)}
 if not path.is_dir():raise GuardError('not a file or directory: '+str(path))
 out={}
 for p in sorted(path.rglob('*')):
  if p.is_symlink():raise GuardError('symlinks forbidden in frozen input tree: '+str(p))
  if p.is_file() and (not source_py_only or p.suffix=='.py'):out[p.relative_to(root).as_posix()]=sha(p)
 return out

def resolve_artifacts(root,rows,label):
 if not isinstance(rows,list) or not rows:raise GuardError(label+' requires nonempty artifact records')
 result={}
 for row in rows:
  if not isinstance(row,dict) or not isinstance(row.get('path'),str):raise GuardError('malformed '+label+' entry')
  p=contained(root,row['path']); m=file_map(root,p)
  if row.get('sha256')!=sha(p):raise GuardError('stale '+label+' hash: '+row['path'])
  result.update(m)
 return result

def pipeline_snapshot(root):
 state_path=root/'pipeline-state.json'; raw=state_path.read_bytes(); state=read_json(state_path)
 rows=state.get('stages')
 if not isinstance(rows,list):raise GuardError('pipeline stages malformed')
 ids=[r.get('id') for r in rows if isinstance(r,dict)]
 if len(ids)!=len(rows) or len(set(ids))!=len(ids):raise GuardError('pipeline stages duplicate/malformed')
 by={r['id']:r for r in rows}
 for sid in ('P2','P3','P3A'):
  row=by.get(sid)
  if not row or row.get('status')!='passed' or row.get('gate_verdict')!='pass':raise GuardError(f'{sid} must be passed/pass')
 if by.get('P4',{}).get('status')!='running':raise GuardError('P4 must be running')
 files={'pipeline-state.json':hashlib.sha256(raw).hexdigest()}
 files.update(resolve_artifacts(root,state.get('contract'),'pipeline contract'))
 for row in rows:
  if row.get('status')=='passed':
   if row.get('gate_verdict')!='pass':raise GuardError('passed stage lacks pass verdict: '+row['id'])
   for label in ('inputs','evidence'): files.update(resolve_artifacts(root,row.get(label),f"{row['id']} {label}"))
 return files,raw

def run_validator(root,python=PYTHON,runner=subprocess.run):
 for rel,expected in VALIDATOR_HASHES.items():
  path=root/rel
  if not path.is_file() or sha(path)!=expected:raise GuardError('official validator copy hash mismatch: '+rel)
 cmd=[str(python),str(root/'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py'),str(root/'pipeline-state.json'),'--before','P4']
 proc=runner(cmd,cwd=str(root),shell=False,capture_output=True,text=True,check=False)
 if proc.returncode:raise GuardError('official P4 predecessor validator failed: '+(proc.stdout or '')+(proc.stderr or ''))
 return {'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}

def snapshot(root,checkpoint,dataset,freeze,policy_config,extra_paths=()):
 files,_=pipeline_snapshot(root)
 for rel,h in VALIDATOR_HASHES.items():files[rel]=sha(root/rel)
 inputs=[checkpoint,dataset,freeze,root/'edsgn/src/edsgn/profile.py',root/Path(__file__).resolve().relative_to(ROOT),root/'edsgn/scripts/aggregate_budget.py',root/'edsgn/scripts/candidate_gpu_fullpath_probe_v3.py']
 if policy_config is not None:inputs.append(policy_config)
 inputs.append(root/NORMALIZER_HELPER_REL)
 inputs.extend(extra_paths)
 for p in inputs:
  files.update(file_map(root,p,source_py_only=(p.is_dir() and p.name=='edsgn')))
 files.update(file_map(root,root/'edsgn/src/edsgn',source_py_only=True))
 return dict(sorted(files.items()))

def _normalization_contract(root,checkpoint,dataset_path,split):
 """Load PreparedData and force validation/test inputs onto training statistics."""
 import numpy as np
 import torch
 sys.path.insert(0,str(root/'edsgn/src'))
 sys.path.insert(0,str(root/'edsgn/scripts'))
 from edsgn import engine
 from edsgn.prepare import apply_normalizer
 from frozen_synthetic_stress import with_frozen_normalization
 payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
 provenance=payload.get('provenance',{})
 if provenance.get('scope')!='full_protocol':raise GuardError('profiling requires full_protocol checkpoint provenance')
 state=payload.get('state_dict')
 if not isinstance(state,dict) or any(torch.is_floating_point(x) and x.dtype!=torch.float32 for x in state.values() if torch.is_tensor(x)):
  raise GuardError('profiling requires an FP32 checkpoint state')
 config=payload.get('config',{}); stats=payload.get('statistics',{})
 train_path=config.get('dataset_path')
 if not train_path:raise GuardError('checkpoint config lacks its training dataset path')
 reference_rows=engine.datasets_at(train_path,'train')
 if not reference_rows:raise GuardError('checkpoint training PreparedData is unavailable')
 reference=reference_rows[0]
 expected_mean=np.asarray(stats.get('norm_mean'),dtype=np.float32)
 expected_std=np.asarray(stats.get('norm_std'),dtype=np.float32)
 expected_scale=np.asarray(stats.get('residual_scale'),dtype=np.float32)
 if expected_mean.shape!=(reference.values.shape[2],) or expected_std.shape!=expected_mean.shape:
  raise GuardError('checkpoint training normalization dimensions are malformed')
 if not np.array_equal(reference.mean,expected_mean) or not np.array_equal(reference.std,expected_std) or not np.array_equal(reference.scale,expected_scale):
  raise GuardError('training PreparedData statistics do not exactly match checkpoint statistics')
 selected=engine.datasets_at(dataset_path,split)
 if not selected:raise GuardError(f'no PreparedData for requested profile split {split}')
 normalized=[]
 for data in selected:
  if (data.values.shape[1:]!=reference.values.shape[1:] or data.history!=reference.history
      or data.hmax!=reference.hmax or data.count_data!=reference.count_data
      or not np.array_equal(data.node_ids,reference.node_ids) or tuple(data.horizons)!=tuple(reference.horizons)):
   raise GuardError('selected PreparedData task/node/horizon identity differs from checkpoint training data')
  if data.event_kind=='synthetic':
   item=with_frozen_normalization(data,payload,reference.node_ids)
  else:
   if not np.array_equal(data.mean,expected_mean) or not np.array_equal(data.std,expected_std) or not np.array_equal(data.scale,expected_scale):
    raise GuardError('non-synthetic PreparedData statistics differ from checkpoint training statistics')
   item=data
  calculated=apply_normalizer(item.raw_values,item.input_mask,expected_mean,expected_std,item.count_data)
  if not np.array_equal(item.values,calculated):
   raise GuardError('selected PreparedData values do not equal the frozen train-normalized raw observations')
  normalized.append(item)
 metadata={'normalization_source':'checkpoint_training_statistics',
           'normalization_helper_path':NORMALIZER_HELPER_REL,
           'normalization_helper_sha256':sha(root/NORMALIZER_HELPER_REL),
           'normalizer_impl_sha256':sha(root/'edsgn/src/edsgn/prepare.py'),
           'normalization_mean':expected_mean.tolist(),'normalization_std':expected_std.tolist(),
           'residual_scale':expected_scale.tolist(),
           'source_layout':'single_chronological_bundle' if Path(dataset_path).is_file() else 'independent_split_collection',
           'combined_bundle_materialized':bool(Path(dataset_path).is_file() and 'test' in selected[0].splits),
           'test_values_materialized':bool(split=='test' or (Path(dataset_path).is_file() and 'test' in selected[0].splits)),
           'test_origins_evaluated':split=='test','test_feedback_used':False}
 return payload,normalized,metadata

def _correct_profile_outputs(output,guard_calls,n_windows,repeats,warmup):
 summary_path=output/'runtime_summary.json'; csv_path=output/'window_timings.csv'
 summary=read_json(summary_path)
 if summary.get('scope')!='full_selected_split' or summary.get('smoke_limit') is not None:raise GuardError('profile runtime summary is not full-split')
 if summary.get('repeats')!=5 or summary.get('warmup_windows')!=10 or summary.get('dtype')!='float32' or summary.get('batch_size')!=1:raise GuardError('profile protocol fields do not match frozen B1/FP32/repeat/warmup')
 if len(guard_calls)<warmup+repeats*n_windows:raise GuardError('budget callback coverage shorter than warmup+replay')
 measured=guard_calls[warmup:warmup+repeats*n_windows]
 with csv_path.open('r',newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f)); fieldnames=list(rows[0].keys()) if rows else []
 if len(rows)!=repeats*n_windows:raise GuardError('window timing row count differs from 5 x origins')
 keys=[]; sums=[0.0]*repeats; corrected=[]
 for i,row in enumerate(rows):
  repeat=int(row['repeat']); key=(int(row['dataset_index']),int(row['origin']))
  if repeat not in range(1,repeats+1):raise GuardError('invalid repeat index')
  keys.append((repeat,*key)); g=measured[i]
  raw=float(row['preprocess_ms']); row['preprocess_ms_including_guard']=f'{raw:.9f}'
  row['guard_overhead_ms']=f'{g*1000:.9f}'; row['preprocess_ms']=f'{max(0.0,raw-g*1000):.9f}'
  corrected.append(float(row['preprocess_ms'])); sums[repeat-1]+=g*1000
  row['end_to_end_ms_excluding_preprocessing']=f"{float(row['end_to_end_ms_excluding_preprocessing']):.9f}"
 expected_pair_count=n_windows
 pairset={(int(row['dataset_index']),int(row['origin'])) for row in rows}
 if len(pairset)!=expected_pair_count:raise GuardError('duplicate or missing canonical pair entries')
 expected={(r,*p) for r in range(1,repeats+1) for p in pairset}
 if len(keys)!=len(set(keys)) or set(keys)!=expected:raise GuardError('window timings have duplicate/missing repeat-origin rows')
 if 'preprocess_ms_including_guard' not in fieldnames:fieldnames.append('preprocess_ms_including_guard')
 if 'guard_overhead_ms' not in fieldnames:fieldnames.append('guard_overhead_ms')
 with csv_path.open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
 # Keep raw side-by-side and expose corrected host prep/replay totals; forward wall samples are untouched.
 summary['preprocessing_ms_per_window_including_guard']=summary.get('preprocessing_ms_per_window')
 summary['preprocessing_ms_per_window']={'median':float(__import__('numpy').percentile(corrected,50)),'p95':float(__import__('numpy').percentile(corrected,95))}
 diagnostic_start=warmup+repeats*n_windows
 diagnostic_ms=sum(guard_calls[diagnostic_start:])*1000
 comp=summary.get('component_diagnostic',{})
 if isinstance(comp,dict) and isinstance(comp.get('diagnostic_replay_host_seconds'),(int,float)):
  comp['diagnostic_replay_host_seconds_including_guard']=comp['diagnostic_replay_host_seconds']
  comp['diagnostic_replay_host_seconds']=max(0,float(comp['diagnostic_replay_host_seconds'])-diagnostic_ms/1000)
 summary['replay_wall_total_ms_including_guard']=summary.get('replay_wall_total_ms')
 if isinstance(summary.get('repeat_summaries'),list):
  for i,row in enumerate(summary['repeat_summaries']):
   row['preprocess_total_ms_including_guard']=row.get('preprocess_total_ms')
   row['preprocess_total_ms']=max(0,float(row.get('preprocess_total_ms',0))-sums[i])
   row['replay_wall_total_ms_including_guard']=row.get('replay_wall_total_ms')
   row['replay_wall_total_ms']=max(0,float(row.get('replay_wall_total_ms',0))-sums[i])
 if isinstance(summary.get('replay_wall_total_ms'),dict) and isinstance(summary.get('replay_wall_total_ms_including_guard'),dict):
  raw=summary['replay_wall_total_ms_including_guard']; samples=raw.get('samples',[])
  corrected_samples=[max(0,float(x)-sums[i]) for i,x in enumerate(samples)]
  summary['replay_wall_total_ms']={'median':float(__import__('numpy').percentile(corrected_samples,50)) if corrected_samples else None,'p95':float(__import__('numpy').percentile(corrected_samples,95)) if corrected_samples else None,'samples':corrected_samples}
 write_atomic_json(summary_path,summary)
 return {'measured_guard_calls':len(measured),'measured_guard_overhead_ms':sum(sums),'corrected_preprocessing_ms':summary['preprocessing_ms_per_window'],'origin_pair_count':n_windows,'timing_rows':len(rows)}

def write_atomic_json(path,value):
 tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 with tmp.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def _blind_helper_context(root,policy,freeze):
 if getattr(policy,'kind',None) not in {'periodic','random'}:return None
 helper_path=root/BLIND_HELPER_REL; acceptance_path=root/BLIND_ACCEPTANCE_REL
 if not helper_path.is_file():raise GuardError('blind scheduler helper missing')
 spec=importlib.util.spec_from_file_location('_frozen_profile_blind_cost_v1',helper_path)
 if spec is None or spec.loader is None:raise GuardError('blind scheduler helper cannot be loaded')
 helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 freeze_doc=read_json(freeze); extra=freeze_doc.get('extra')
 try:binding=helper.validate_freeze_extra(extra)
 except Exception as exc:raise GuardError(f'frozen blind-helper binding missing/stale: {exc}') from exc
 descriptor=helper.descriptor()
 if descriptor.get('source_sha256')!=binding.get('source_sha256') or Path(descriptor.get('source_path','')).resolve()!=helper_path.resolve():raise GuardError('blind helper descriptor/source path mismatch')
 if not acceptance_path.is_file():raise GuardError('blind scheduler helper lacks independent root acceptance')
 acceptance=read_json(acceptance_path)
 evidence=acceptance.get('evidence_sha256')
 if (acceptance.get('accepted_by')!='root' or acceptance.get('verdict')!='pass'
     or acceptance.get('scope')!='engineering-staging-only'
     or acceptance.get('pipeline_advanced') is not False
     or acceptance.get('scientific_hypothesis_accepted') is not False
     or acceptance.get('actual_blind_control_run') is not False
     or acceptance.get('c07_experiment_closed') is not False
     or not isinstance(evidence,dict)
     or evidence.get(BLIND_HELPER_REL)!=binding['source_sha256']
     or helper.descriptor().get('rule_version')!=binding['rule_version']):
  raise GuardError('blind scheduler helper lacks the actual root engineering-staging acceptance record')
 helper.apply_blind_reference_cost(policy)
 return {'helper':helper,'descriptor':descriptor,'binding':binding,'acceptance_path':acceptance_path,
         'acceptance_sha256':sha(acceptance_path),'freeze_sha256':sha(freeze)}

def _revalidate_blind_context(context,freeze):
 if context is None:return
 helper=context['helper']
 binding=helper.validate_freeze_extra(read_json(freeze).get('extra'))
 descriptor=helper.descriptor()
 if binding!=context['binding'] or descriptor!=context['descriptor']:
  raise GuardError('frozen blind helper binding/descriptor changed after profile serialization')
 if sha(context['acceptance_path'])!=context['acceptance_sha256'] or sha(freeze)!=context['freeze_sha256']:
  raise GuardError('blind helper acceptance/freeze changed after profile serialization')

def main_profile(checkpoint,dataset_path,freeze_path,output,split,policy_kind,policy_config,device,*,root=ROOT,deps=None):
 root=Path(root).resolve();deps=deps or {}
 checkpoint=contained(root,checkpoint);dataset=contained(root,dataset_path);freeze=contained(root,freeze_path)
 policy_path=contained(root,policy_config) if policy_config else None
 output=contained(root,output,exists=False)
 if output.exists():raise FileExistsError('profile output must be a new directory')
 if split not in {'val','test'}:raise GuardError('split must be val or test')
 if device not in {'cpu','cuda'}:raise GuardError('device must be explicitly fixed to cpu or cuda')
 if split=='test' and freeze!=(checkpoint.parent/'test-freeze.json').resolve():raise GuardError('test profiling freeze must be checkpoint sibling test-freeze.json')
 phase_started=time.perf_counter()
 if not output.parent.is_dir():raise GuardError('output parent must already exist')
 import numpy as np
 sys.path.insert(0,str(root/'edsgn/src'));sys.path.insert(0,str(root/'edsgn/scripts'))
 from edsgn import profile as profile_module
 import aggregate_budget as budget_module
 original_datasets_at=profile_module.datasets_at
 if 'owned' in deps: _owned_default=None
 else:
  from candidate_gpu_fullpath_probe_v3 import _owned_model_processes as _owned_default
 policy=profile_module.build_policy(policy_kind,policy_path)
 blind_context=_blind_helper_context(root,policy,freeze)
 blind_paths=[] if blind_context is None else [root/BLIND_HELPER_REL,blind_context['acceptance_path']]
 before_files=snapshot(root,checkpoint,dataset,freeze,policy_path,blind_paths)
 _,statebytes=pipeline_snapshot(root)
 vbefore=run_validator(root,deps.get('python',PYTHON),deps.get('runner',subprocess.run))
 aggregate=deps.get('aggregate',budget_module.aggregate); owned=deps.get('owned',_owned_default)
 phase_dir=root/'logs/offline-phase-wall';phase_dir.mkdir(parents=True,exist_ok=True)
 phase_path=phase_dir/f"frozen-profile-{uuid.uuid4().hex}.json"; job_id=phase_path.stem
 started=phase_started;leases=[];guard_calls=[];pair_capture=[];profile_summary=None;status='running';error=None;final_stats={};original_prepare=profile_module.prepare_window;original_index=profile_module.index_origins
 receipt={'scope':'offline-experiment-phase','phase':'profiling','status':'running','job_id':job_id,'output_dir':str(output),'wall_seconds':0.0,'split':split,'device':device,'precision':'float32','batch_size':1,'repeats':5,'warmup_windows':10,'policy':asdict(policy)}
 write_atomic_json(phase_path,receipt)
 def check_budget(label):
  receipt['wall_seconds']=time.perf_counter()-started;write_atomic_json(phase_path,receipt)
  summary=aggregate(root,exclude_active_output=str(output))
  if not summary.get('coverage_complete'):raise GuardError('cumulative budget coverage incomplete')
  elapsed=time.perf_counter()-started
  if float(summary.get('spent_seconds',0))+elapsed>=float(summary.get('budget_seconds',WALL_BUDGET_S)):raise GuardError('72-hour cumulative stop-loss reached')
  return summary
 def prep_guarded(ds,origin):
  t=time.perf_counter();check_budget('before_prepare_window');guard_calls.append(time.perf_counter()-t)
  return original_prepare(ds,origin)
 def index_capture(datasets,split_name,limit=None,stride=1):
  pairs=original_index(datasets,split_name,limit=limit,stride=stride);pair_capture.extend((int(a),int(b)) for a,b in pairs);return pairs
 try:
  if owned(root):raise GuardError('owned model/resource process already active')
  check_budget('before-locks')
  for lock in (root/'logs/model-resource.lock',root/'logs/offline-phase-wall/campaign.lock'):
   lock.parent.mkdir(parents=True,exist_ok=True)
   with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'job_id':job_id,'phase':'frozen-profile-guarded-v2'},f)
   leases.append(lock)
  if owned(root):raise GuardError('owned model/resource process appeared after leases')
  payload,profile_datasets,normalization_metadata=_normalization_contract(root,checkpoint,dataset,split)
  def bound_datasets(path,requested_split):
   if Path(path).resolve()!=dataset or requested_split!=split:
    raise GuardError('profile attempted to load a split/path outside the normalization-bound PreparedData')
   return profile_datasets
  profile_module.datasets_at=bound_datasets
  profile_module.prepare_window=prep_guarded;profile_module.index_origins=index_capture
  profile_summary=profile_module.profile_run(checkpoint,dataset,split,output,policy,device_name=device,smoke_limit=None,repeats=5,warmup=10)
  if not pair_capture:raise GuardError('canonical full-split origin index was not captured')
  if len(pair_capture)!=len(set(pair_capture)):raise GuardError('canonical origin index contains duplicates')
  if len(set(pair_capture))!=int(profile_summary['n_windows']):raise GuardError('profile window count differs from canonical origin list')
  count=int(profile_summary['n_windows'])
  corrected=_correct_profile_outputs(output,guard_calls,count,5,10)
  # _correct_profile_outputs rewrites derived timing fields on disk; reload
  # before returning or appending the blind-control descriptor.
  profile_summary=read_json(output/'runtime_summary.json')
  profile_summary['normalization_provenance']=normalization_metadata
  write_atomic_json(output/'runtime_summary.json',profile_summary)
  receipt['profile_parameter_hash_before']=profile_summary.get('parameters_sha256_before')
  receipt['profile_parameter_hash_after']=profile_summary.get('parameters_sha256_after')
  if not receipt['profile_parameter_hash_before'] or receipt['profile_parameter_hash_before']!=receipt['profile_parameter_hash_after']:raise GuardError('profile parameter hash changed/missing')
  if profile_summary.get('device')!=device and not str(profile_summary.get('device','')).startswith(device+':'):raise GuardError('runtime device differs from fixed request')
  guard_doc={'scope':'guard-overhead-sidecar','guard_callback_scope':'before every profile.prepare_window call including warmup/measured/diagnostic windows','guard_calls_total':len(guard_calls),'guard_overhead_total_ms':sum(guard_calls)*1000,'warmup_calls':10,'measured_calls':5*count,'diagnostic_calls':len(guard_calls)-10-5*count,'measured_calls_corrected':corrected,'scientific_model_forward_timing_unchanged':True}
  write_atomic_json(output/'profile-guard-overhead.json',guard_doc)
  # Verify full repeated-origin coverage using the captured canonical origin pairs.
  with (output/'window_timings.csv').open('r',newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
  observed=[(int(r['dataset_index']),int(r['origin'])) for r in rows if int(r['repeat'])==1]
  if observed!=pair_capture:raise GuardError('window timing row origins/order differ from canonical origins')
  for repeat in range(1,6):
   repeated=[(int(r['dataset_index']),int(r['origin'])) for r in rows if int(r['repeat'])==repeat]
   if repeated!=pair_capture:raise GuardError(f'window timing repeat {repeat} does not preserve the full canonical origin order')
  required=['runtime_summary.json','window_timings.csv','trace.jsonl','profile-budget-accounting.json','profile-guard-overhead.json']
  if any(not (output/n).is_file() or (output/n).stat().st_size==0 for n in required):raise GuardError('profile final artifact missing/empty')
  if profile_summary.get('scope')!='full_selected_split' or profile_summary.get('n_windows')!=len(pair_capture):raise GuardError('full selected split was not profiled')
  if profile_summary.get('policy')!=asdict(policy):raise GuardError('profile policy differs from the frozen requested policy')
  if blind_context is not None:
   corrected_summary=read_json(output/'runtime_summary.json')
   corrected_summary['blind_reference_cost_policy']={'descriptor':blind_context['descriptor'],'frozen_binding':blind_context['binding'],'root_acceptance_sha256':blind_context['acceptance_sha256']}
   write_atomic_json(output/'runtime_summary.json',corrected_summary)
   profile_summary=corrected_summary
   _revalidate_blind_context(blind_context,freeze)
  hashes={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
  _revalidate_blind_context(blind_context,freeze)
  final_files=snapshot(root,checkpoint,dataset,freeze,policy_path,blind_paths)
  if final_files!=before_files:raise GuardError('frozen source/data/checkpoint/freeze/policy/P3A bytes or membership changed')
  if (root/'pipeline-state.json').read_bytes()!=statebytes:raise GuardError('canonical pipeline changed during profile')
  vafter=run_validator(root,deps.get('python',PYTHON),deps.get('runner',subprocess.run))
  # Re-snapshot after the official final gate, so the completed phase receipt
  # certifies one post-gate view of all authorities and serialized outputs.
  post_gate_files=snapshot(root,checkpoint,dataset,freeze,policy_path,blind_paths)
  if post_gate_files!=before_files:raise GuardError('frozen inputs changed after final official gate')
  if (root/'pipeline-state.json').read_bytes()!=statebytes:raise GuardError('canonical pipeline changed after final official gate')
  _revalidate_blind_context(blind_context,freeze)
  post_gate_hashes={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
  if post_gate_hashes!=hashes:raise GuardError('profile artifacts changed after final official gate')
  # The last budget check runs after final validation and all artifact/source
  # hashes; only its success permits a completed receipt.
  check_budget('final-after-all-files-and-gates')
  status='completed';receipt.update(status=status,wall_seconds=time.perf_counter()-started,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),canonical_origin_count=len(pair_capture),repeats=5,warmup_windows=10,output_sha256=hashes,source_input_manifest_sha256=hashlib.sha256(json.dumps(before_files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),validator_before=vbefore,validator_after=vafter,guard_overhead_ms=guard_doc['guard_overhead_total_ms'])
  write_atomic_json(phase_path,receipt)
  return {'status':status,'receipt':phase_path,'summary':profile_summary,'corrected_timing':corrected}
 except BaseException as exc:
  error=exc;status='failed';receipt.update(status=status,error=f'{type(exc).__name__}: {exc}',wall_seconds=time.perf_counter()-started,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  try:write_atomic_json(phase_path,receipt)
  except FileExistsError: pass
  raise
 finally:
  profile_module.prepare_window=original_prepare;profile_module.index_origins=original_index;profile_module.datasets_at=original_datasets_at
  for lock in reversed(leases):lock.unlink(missing_ok=True)

def main():
 p=argparse.ArgumentParser(description='Guarded full-origin frozen profiler; no smoke mode')
 p.add_argument('--checkpoint',required=True);p.add_argument('--dataset-path',required=True);p.add_argument('--freeze',required=True);p.add_argument('--output',required=True)
 p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--policy',choices=['full','periodic','random','adaptive','graph_only','state_only','off'],default='full');p.add_argument('--policy-config')
 p.add_argument('--device',choices=['cpu','cuda'],required=True)
 a=p.parse_args();result=main_profile(a.checkpoint,a.dataset_path,a.freeze,a.output,a.split,a.policy,a.policy_config,a.device)
 print(json.dumps({'status':result['status'],'receipt':str(result['receipt'])},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())



