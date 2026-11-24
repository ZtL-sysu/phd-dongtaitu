from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRUSTED_PYTHON = Path(r'C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe')
VALIDATOR = ROOT / 'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py'
VALIDATOR_FILES = {
 'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py':'5e88f56801955ded32e00fbc1b916ba4e9237b4c3eb76d70ac4c2cc6fba62525',
 'edsgn/scripts/frozen-pipeline-validator/scripts/citation_requirements.py':'efe945b9d14e9a24b9b18d2d2a9b318d4afff8b9d0683391ea6141017e46a069',
 'edsgn/scripts/frozen-pipeline-validator/scripts/delivery_readiness.py':'92ec18db25fb392ff55c9d164d89ffd0d3572de17c4e0622b515cc7ab5871988',
 'edsgn/scripts/frozen-pipeline-validator/assets/pipeline-definition.json':'1d4aadcea7a2e6eaaf24825c08dcec1433c79a3e6b6323d1424861fee7005e6e'}
DRIVERS = {
 'graph_v7':('edsgn/staging/frozen_graph_state_diagnostic_v7.py', {'--checkpoint','--dataset-path','--output','--policy','--policy-config','--device'}, {'--checkpoint','--dataset-path','--output'}),
 'h2_v4':('edsgn/staging/frozen_h2_matched_retention_v4.py', {'--checkpoint','--dataset-path','--split','--output','--device'}, {'--checkpoint','--dataset-path','--split','--output'}),
 'calibration_v9':('edsgn/staging/frozen_validation_calibration_v9.py', {'--checkpoint','--config','--queue','--epsilon-authority','--launch-binding','--binding','--output','--device','--root'}, {'--checkpoint','--config','--queue','--epsilon-authority','--launch-binding','--binding','--output','--device'}),
 'c06_v5':('edsgn/staging/frozen_split_history_robustness_v5.py', {'--checkpoint','--dataset-path','--domain-manifest','--recipe-json','--output','--split','--policy','--policy-config','--device','--batch-size'}, {'--checkpoint','--dataset-path','--domain-manifest','--recipe-json','--output','--split'}),
 'prediction_v5':('edsgn/staging/frozen_prediction_guarded_v5.py', {'--checkpoint','--dataset-path','--binding','--output','--split','--policy','--policy-config','--device','--root'}, {'--checkpoint','--dataset-path','--binding','--output','--split'}),
 'profile_v4':('edsgn/staging/frozen_profile_guarded_v4.py', {'--checkpoint','--dataset-path','--freeze','--output','--split','--policy','--policy-config','--device'}, {'--checkpoint','--dataset-path','--freeze','--output','--split'})}
PATH_FLAGS={'--checkpoint','--dataset-path','--config','--queue','--epsilon-evidence','--epsilon-authority','--launch-binding','--binding','--freeze','--domain-manifest','--recipe-json','--policy-config','--output'}

class DispatchError(RuntimeError): pass

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def strict_pairs(pairs):
 d={}
 for k,v in pairs:
  if k in d: raise DispatchError('duplicate JSON key: '+k)
  d[k]=v
 return d

def load_json(path):
 return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=strict_pairs)

def inroot(raw, root=ROOT):
 p=Path(raw)
 if not p.is_absolute(): p=root/p
 p=p.resolve(strict=True)
 if not p.is_relative_to(root.resolve()): raise DispatchError(f'path outside run root: {raw}')
 return p

def file_map(path, root=ROOT):
 path=Path(path)
 if path.is_symlink(): raise DispatchError(f'symlink forbidden: {path}')
 if path.is_file(): return {path.relative_to(root).as_posix():sha(path)}
 if not path.is_dir(): raise DispatchError(f'missing source: {path}')
 out={}
 for p in sorted(path.rglob('*')):
  if p.is_symlink(): raise DispatchError(f'symlink forbidden in source tree: {p}')
  if p.is_file(): out[p.relative_to(root).as_posix()]=sha(p)
 return out

def tree_digest(mapping):
 data=json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()
 return hashlib.sha256(data).hexdigest()

def _bound_record(root, raw, expected=None):
 p=inroot(raw,root)
 m=file_map(p,root)
 if expected is not None:
  got=sha(p) if p.is_file() else tree_digest(m)
  if got != expected: raise DispatchError(f'declared source hash mismatch: {raw}')
 return p,m

def _validate_validator(root):
 for rel,expected in VALIDATOR_FILES.items():
  p=root/rel
  if not p.is_file() or sha(p)!=expected: raise DispatchError('frozen validator bundle mismatch: '+rel)

def _argv(job, root, allow_existing_output=False):
 driver=job.get('driver'); argv=job.get('argv')
 if driver not in DRIVERS or not isinstance(argv,list) or not argv or any(not isinstance(x,str) or not x for x in argv): raise DispatchError('invalid driver/argv')
 rel,allowed,required=DRIVERS[driver]
 flags=[]; i=0
 while i<len(argv):
  flag=argv[i]
  if flag not in allowed or flag in flags or i+1>=len(argv) or argv[i+1].startswith('--'): raise DispatchError('unapproved, duplicate or valueless argv flag: '+flag)
  flags.append(flag); value=argv[i+1]
  if flag=='--root':
   candidate=Path(value); q=(root/candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
   if q!=root.resolve(): raise DispatchError('--root must name the canonical run root')
  if flag in PATH_FLAGS:
   candidate=Path(value)
   p=(root/candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
   if not p.is_relative_to(root.resolve()): raise DispatchError(f'path outside run root: {value}')
   if flag=='--output':
    if p.exists() and not allow_existing_output: raise DispatchError('output already exists')
    if not p.parent.is_dir(): raise DispatchError('output parent must already exist')
   elif not p.exists(): raise DispatchError('input path missing: '+value)
  if flag=='--split' and value not in {'validation','val','test'}: raise DispatchError('unsupported split')
  i+=2
 if not required.issubset(set(flags)): raise DispatchError(f'required flags missing; got {flags}')
 if job.get('phase_receipt_alias') is None: raise DispatchError('phase_receipt_alias is required')
 test_bound=any(argv[i]=='--split' and argv[i+1]=='test' for i in range(0,len(argv),2)) or job.get('driver') in {'graph_v7','prediction_v5'}
 if test_bound and not job.get('pretest_execution_bindings'):
  raise DispatchError('test-split execution requires at least one accepted pretest binding')
 return rel,argv

def _state_snapshot(root):
 state_path=root/'pipeline-state.json'
 raw=state_path.read_bytes(); state=load_json(state_path)
 stages=state.get('stages')
 if not isinstance(stages,list): raise DispatchError('pipeline stages malformed')
 ids=[r.get('id') for r in stages if isinstance(r,dict)]
 if len(ids)!=len(stages) or len(ids)!=len(set(ids)): raise DispatchError('duplicate/malformed stage rows')
 passed={}
 for item in state.get('contract', []):
  if not isinstance(item,dict) or not isinstance(item.get('path'),str): raise DispatchError('malformed contract artifact')
  _bound_record(root,item['path'],item.get('sha256'))
 for row in stages:
  if row.get('status')=='passed':
   if row.get('gate_verdict')!='pass': raise DispatchError('passed stage lacks pass verdict: '+str(row.get('id')))
   entries=[]
   for group in ('inputs','evidence'):
    arr=row.get(group)
    if not isinstance(arr,list) or not arr: raise DispatchError('passed stage has empty '+group)
    for item in arr:
     if not isinstance(item,dict) or not isinstance(item.get('path'),str): raise DispatchError('malformed stage artifact')
     p,m=_bound_record(root,item['path'],item.get('sha256'))
     entries.append((group,item['path'],m))
   passed[row['id']]=entries
 # Require P4 is the active boundary; the official validator checks ordered predecessor pass.
 p4=next((r for r in stages if r.get('id')=='P4'),None)
 if not p4 or p4.get('status')!='running': raise DispatchError('P4 must be running for frozen dispatch')
 return {'state_bytes_sha256':hashlib.sha256(raw).hexdigest(),'passed':passed,'state_path':state_path}

def _snapshot(job, root, job_path, driver_path):
 snap=_state_snapshot(root)
 items={}
 def add(label,raw,expected=None):
  p,m=_bound_record(root,raw,expected)
  for rel,h in m.items():
   if rel in items and items[rel]!=h: raise DispatchError('conflicting snapshot entry '+rel)
   items[rel]=h
  return {'path':str(p.relative_to(root).as_posix()),'files':m,'sha256':sha(p) if p.is_file() else tree_digest(m)}
 for rel,expected in VALIDATOR_FILES.items(): add('validator',rel,expected)
 add('pipeline-state','pipeline-state.json')
 for stage,entries in snap['passed'].items():
  for group,raw,m in entries:
   for rel,h in m.items(): items[rel]=h
 add('job',str(job_path.relative_to(root)))
 add('dispatcher',str(Path(__file__).resolve().relative_to(root)))
 add('driver',str(driver_path.relative_to(root)))
 for src in job.get('sources',[]):
  if not isinstance(src,dict) or set(src)-{'path','sha256'} or not isinstance(src.get('path'),str) or not isinstance(src.get('sha256'),str): raise DispatchError('malformed job source descriptor')
  add('declared-source',src['path'],src['sha256'])
 bindings=job.get('pretest_execution_bindings')
 if not isinstance(bindings,list): raise DispatchError('pretest_execution_bindings must be a list')
 seen_bindings=set()
 for item in bindings:
  if not isinstance(item,dict) or set(item)!={'path','sha256'}: raise DispatchError('malformed pretest execution binding descriptor')
  if item['path'] in seen_bindings: raise DispatchError('duplicate pretest execution binding path')
  seen_bindings.add(item['path'])
  record=add('pretest-execution-binding',item['path'],item['sha256'])
  _validate_execution_binding(root/record['path'])
 alias=job.get('phase_receipt_alias')
 alias_path=_validate_alias(root,alias,require_absent=False)
 _,argv=_argv(job,root,allow_existing_output=True)
 for i in range(0,len(argv),2):
  flag,value=argv[i],argv[i+1]
  if flag in PATH_FLAGS and flag!='--output': add('argument',value)
 return {'files':dict(sorted(items.items())),'pipeline_state_sha256':snap['state_bytes_sha256'],'passed_stage_ids':list(snap['passed']),
         'phase_receipt_alias_path':alias_path.relative_to(root).as_posix()}

def _validate_execution_binding(path):
 doc=load_json(path)
 if (doc.get('schema_version')!=2 or doc.get('status')!='accepted_by_root' or
     doc.get('scope')!='pretest_execution_binding' or doc.get('test_feedback_used') is not False):
  raise DispatchError('pretest execution binding lacks accepted schema/status/scope or has test feedback')
 if doc.get('test_artifacts_in_binding',False) is not False:
  raise DispatchError('pretest execution binding contains future test artifacts')
 def scan(value,key=''):
  if isinstance(value,dict):
   for k,v in value.items():
    low=str(k).lower()
    if ('test' in low and low not in {'test_feedback_used','test_artifacts_in_binding'} and v not in (False,None,'not_applicable')):
     raise DispatchError('pretest execution binding has a future test reference')
    scan(v,low)
  elif isinstance(value,list):
   for v in value: scan(v,key)
  elif isinstance(value,str) and (key.endswith('path') or key in {'path','output_path'}):
   low=value.replace('\\','/').lower()
   if '/test/' in '/'+low or 'test-output' in low or 'test_predictions' in low or 'test_artifact' in low:
    raise DispatchError('pretest execution binding references a future test output')
 scan(doc)
 return doc

def _validate_alias(root, raw, require_absent=True):
 if not isinstance(raw,str) or not raw: raise DispatchError('phase_receipt_alias must be a nonempty path')
 p=Path(raw); p=(root/p).resolve() if not p.is_absolute() else p.resolve()
 if not p.is_relative_to(root.resolve()) or p.is_symlink(): raise DispatchError('phase receipt alias escapes root or is symlinked')
 if not p.is_relative_to((root/'runs').resolve()) or p.parent.name!='evidence': raise DispatchError('phase receipt alias must be under runs/.../evidence/')
 if not p.parent.is_dir() or p.parent.is_symlink(): raise DispatchError('phase receipt alias parent must already exist and be nonsymlinked')
 cursor=p.parent
 while cursor!=root and cursor.is_relative_to(root):
  if cursor.is_symlink(): raise DispatchError('phase receipt alias has a symlinked parent')
  cursor=cursor.parent
 if require_absent and p.exists(): raise DispatchError('phase receipt alias already exists')
 return p

def _verify_child_completion(driver, output, phase_before, phase_dir, root):
 if output.is_symlink() or not output.is_dir(): raise DispatchError('child output directory missing or symlinked after zero exit')
 required_artifacts={
  'graph_v7':['predictions.npz','counter-rows.json','frozen-graph-state-diagnostic.json'],
  'h2_v4':['h2-matched-retention-report.json'],
  'calibration_v9':['calibration.json','validation-predictions.npz','calibration-validation-dependence-v2.json'],
  'c06_v5':['predictions.npz','metrics_by_horizon.csv','counter-rows.json','perturbation-manifest.json'],
  'prediction_v5':['predictions.npz','evaluation_manifest.json'],
  'profile_v4':['runtime_summary.json','window_timings.csv','trace.jsonl','profile-budget-accounting.json','profile-guard-overhead.json']}
 for name in required_artifacts[driver]:
  if (output/name).is_symlink() or not (output/name).is_file() or (output/name).stat().st_size==0: raise DispatchError('principal child artifact missing/empty: '+name)
 if driver=='h2_v4' and not list(output.glob('predictions-*.npz')): raise DispatchError('H2 arm predictions absent')
 candidates=[]
 for p in phase_dir.glob('*.json'):
  if p.resolve() in phase_before: continue
  if p.is_symlink(): raise DispatchError('child phase receipt symlink rejected')
  try: doc=load_json(p)
  except Exception: continue
  raw=doc.get('output_dir')
  if not isinstance(raw,str): continue
  q=Path(raw).resolve()
  if q==output.resolve(): candidates.append((p,doc))
 if len(candidates)!=1: raise DispatchError(f'expected exactly one completed child phase receipt for output; got {len(candidates)}')
 path,doc=candidates[0]
 current_output_hashes={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
 wanted={'calibration_v9':'calibration','profile_v4':'profiling'}.get(driver,'evaluation')
 if doc.get('scope')!='offline-experiment-phase' or doc.get('phase')!=wanted or doc.get('status')!='completed':
  raise DispatchError('child phase receipt is not a completed expected phase')
 if not isinstance(doc.get('job_id'),str) or not doc['job_id']: raise DispatchError('child receipt job_id missing')
 if driver in {'h2_v4','c06_v5'} and 'report_sha256' in doc:
  name='h2-matched-retention-report.json' if driver=='h2_v4' else None
  if name and doc['report_sha256']!=sha(output/name): raise DispatchError('child receipt report hash mismatch')
 if driver=='graph_v7':
  if not isinstance(doc.get('origin_count'),int) or doc['origin_count']<=0: raise DispatchError('graph child receipt has no completed origins')
  report=load_json(output/'frozen-graph-state-diagnostic.json')
  if report.get('predictions_file')!='predictions.npz' or not isinstance(report.get('full_test_origin_count'),int) or report['full_test_origin_count']!=doc['origin_count']:
   raise DispatchError('graph principal report/receipt mismatch')
 if driver=='h2_v4':
  report_path=output/'h2-matched-retention-report.json'
  if not isinstance(doc.get('origin_count'),int) or doc['origin_count']<=0 or doc.get('report_sha256')!=sha(report_path): raise DispatchError('H2 report/receipt binding mismatch')
  report=load_json(report_path)
  if report.get('origin_count')!=doc['origin_count']: raise DispatchError('H2 full-origin count differs between report and child receipt')
 if driver=='c06_v5':
  if not isinstance(doc.get('origins_completed'),int) or doc['origins_completed']<=0: raise DispatchError('C06 receipt has no completed origins')
  hashes=doc.get('outputs')
  if not isinstance(hashes,dict) or not hashes: raise DispatchError('C06 receipt output hashes missing')
  required=set(required_artifacts['c06_v5'])
  if not required.issubset(hashes): raise DispatchError('C06 receipt output hashes omit principal artifact(s): '+str(sorted(required-set(hashes))))
  if hashes!=current_output_hashes: raise DispatchError('C06 receipt output hash map differs from actual output directory')
  for name,digest in hashes.items():
   if Path(name).name!=name: raise DispatchError('C06 receipt output name escapes output directory')
   p=output/name
   if p.is_symlink() or not p.is_file() or sha(p)!=digest: raise DispatchError('C06 receipt output hash mismatch: '+str(name))
 if driver=='calibration_v9':
  if doc.get('input_hashes_match') is not True or doc.get('input_sha256_before')!=doc.get('input_sha256_after'):
   raise DispatchError('calibration child receipt input fence missing/failed')
  if doc.get('parameter_hash_before')!=doc.get('parameter_hash_after') or not doc.get('parameter_hash_before'):
   raise DispatchError('calibration child receipt parameter fence missing/failed')
  calibration=load_json(output/'calibration.json')
  with np.load(output/'validation-predictions.npz',allow_pickle=False) as archive:
   if calibration.get('validation_origin_count')!=len(archive['origins']): raise DispatchError('calibration validation origin count differs from saved predictions')
  if calibration.get('test_origins_evaluated') is not False or calibration.get('test_feedback_used') is not False:
   raise DispatchError('calibration child reports test feedback/access')
  if calibration.get('validation_prediction_sha256')!=sha(output/'validation-predictions.npz'):
   raise DispatchError('calibration report does not bind its validation prediction artifact')
 if driver=='prediction_v5':
  if doc.get('origin_count',0)<=0 or doc.get('test_accessed') is not True or doc.get('test_feedback_used') is not False:
   raise DispatchError('prediction child receipt lacks full test execution/no-feedback evidence')
  if doc.get('parameters_sha256') is None or not doc.get('input_hashes_match'):
   raise DispatchError('prediction child receipt lacks parameter/input integrity proof')
  output_records=doc.get('outputs')
  expected_records=[{'path':str((output/name).resolve()),'sha256':digest} for name,digest in sorted(current_output_hashes.items())]
  if output_records!=expected_records: raise DispatchError('prediction receipt output hashes differ from actual output directory')
  manifest=load_json(output/'evaluation_manifest.json')
  if manifest.get('n_windows')!=doc['origin_count'] or manifest.get('split')!='test':
   raise DispatchError('prediction manifest/receipt full-origin mismatch')
 if driver=='profile_v4':
  if doc.get('canonical_origin_count',0)<=0 or doc.get('repeats')!=5 or doc.get('warmup_windows')!=10:
   raise DispatchError('profile child receipt lacks full-origin repeated replay proof')
  if not doc.get('profile_parameter_hash_before') or doc.get('profile_parameter_hash_before')!=doc.get('profile_parameter_hash_after'):
   raise DispatchError('profile child receipt parameter hash mismatch')
  if doc.get('output_sha256')!=current_output_hashes: raise DispatchError('profile receipt output hash map differs from actual output directory')
  summary=load_json(output/'runtime_summary.json')
  if summary.get('scope')!='full_selected_split' or summary.get('smoke_limit') is not None or summary.get('n_windows')!=doc['canonical_origin_count']:
   raise DispatchError('profile summary is not a full unsmoked origin replay')
 return {'path':path.relative_to(root).as_posix(),'sha256':sha(path),'job_id':doc['job_id'],'phase':doc['phase'],'status':doc['status'],'output_dir':str(output),'principal_artifacts':{name:sha(output/name) for name in required_artifacts[driver]},'receipt_doc':doc}

def _run(cmd, root, runner=subprocess.run):
 return runner(cmd,cwd=str(root),shell=False,capture_output=True,text=True,check=False)

def dispatch(job_path, *, root=ROOT, python=TRUSTED_PYTHON, runner=subprocess.run):
 root=Path(root).resolve(); job_path=inroot(str(job_path),root)
 if not python.is_file(): raise DispatchError('trusted Conda Python missing')
 job=load_json(job_path)
 if set(job)!={'schema_version','job_id','driver','argv','sources','pretest_execution_bindings','phase_receipt_alias'} or job['schema_version']!=2: raise DispatchError('job schema invalid')
 jobid=job['job_id']
 if not isinstance(jobid,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',jobid): raise DispatchError('invalid job_id')
 if not isinstance(job['sources'],list): raise DispatchError('sources must be list')
 if not isinstance(job['pretest_execution_bindings'],list): raise DispatchError('pretest_execution_bindings must be list')
 alias_path=_validate_alias(root,job['phase_receipt_alias'],require_absent=True)
 _validate_validator(root)
 driver_rel,argv=_argv(job,root); driver=root/driver_rel
 output_arg=next(argv[i+1] for i in range(0,len(argv),2) if argv[i]=='--output')
 output_path=(root/Path(output_arg)).resolve() if not Path(output_arg).is_absolute() else Path(output_arg).resolve()
 if output_path.exists(): raise DispatchError('output already exists')
 logdir=root/'logs/verified-frozen-dispatch'; logdir.mkdir(parents=True,exist_ok=True)
 receipt=logdir/f'{jobid}.json'; stdoutp=logdir/f'{jobid}.stdout.log'; stderrp=logdir/f'{jobid}.stderr.log'
 if any(p.exists() for p in (receipt,stdoutp,stderrp)): raise DispatchError('dispatch outputs already exist')
 started=time.time(); before=None; after=None; error=None; status='failed'; childrc=None; vbefore=None; vafter=None; out=''; err=''; child_receipt=None
 output_arg=next(argv[i+1] for i in range(0,len(argv),2) if argv[i]=='--output')
 output_path=(root/Path(output_arg)).resolve() if not Path(output_arg).is_absolute() else Path(output_arg).resolve()
 phase_dir=root/'logs/offline-phase-wall'; phase_dir.mkdir(parents=True,exist_ok=True)
 phase_before={p.resolve() for p in phase_dir.glob('*.json')}
 try:
  before=_snapshot(job,root,job_path,driver)
  vcmd=[str(python),str(VALIDATOR),str(root/'pipeline-state.json'),'--before','P4']
  vr=_run(vcmd,root,runner)
  vbefore={'returncode':vr.returncode,'stdout':vr.stdout,'stderr':vr.stderr}
  if vr.returncode!=0: raise DispatchError('official before-P4 validator failed')
  result=_run([str(python),str(driver),*argv],root,runner)
  childrc=result.returncode; out=result.stdout or ''; err=result.stderr or ''
  after=_snapshot(job,root,job_path,driver)
  if childrc!=0: raise DispatchError(f'child returned {childrc}')
  child_receipt=_verify_child_completion(job['driver'],output_path,phase_before,phase_dir,root)
  if before!=after: raise DispatchError('frozen bytes/path membership changed during child')
  vr2=_run(vcmd,root,runner)
  vafter={'returncode':vr2.returncode,'stdout':vr2.stdout,'stderr':vr2.stderr}
  if vr2.returncode!=0: raise DispatchError('official post-child before-P4 validator failed')
  # Re-read principal artifacts and the exact child receipt after the final validator.
  final_child=_verify_child_completion(job['driver'],output_path,phase_before,phase_dir,root)
  if final_child!=child_receipt: raise DispatchError('child receipt or principal artifact hashes changed after final validator')
  # Validator bytes and state are also in before/after snapshots; run one final direct byte check.
  _validate_validator(root)
  if _snapshot(job,root,job_path,driver)!=before: raise DispatchError('final frozen snapshot changed')
  final_child=_verify_child_completion(job['driver'],output_path,phase_before,phase_dir,root)
  if final_child!=child_receipt: raise DispatchError('child receipt or principal artifact hashes changed after final frozen snapshot')
  original_receipt=root/child_receipt['path']; original_bytes=original_receipt.read_bytes(); original_sha=sha(original_receipt)
  if original_sha!=child_receipt['sha256']: raise DispatchError('original child receipt changed before aliasing')
  with alias_path.open('xb') as f: f.write(original_bytes); f.flush(); os.fsync(f.fileno())
  alias_sha=sha(alias_path)
  if alias_sha!=original_sha or original_receipt.read_bytes()!=alias_path.read_bytes(): raise DispatchError('phase receipt alias byte copy mismatch')
  if sha(original_receipt)!=original_sha or sha(alias_path)!=alias_sha: raise DispatchError('phase receipt bytes changed during final alias fence')
  final_child=_verify_child_completion(job['driver'],output_path,phase_before,phase_dir,root)
  if final_child!=child_receipt: raise DispatchError('child receipt or principal artifact changed after alias copy')
  if _snapshot(job,root,job_path,driver)!=before: raise DispatchError('final frozen snapshot changed after receipt alias')
  child_receipt.update(phase_receipt_original_path=child_receipt['path'],phase_receipt_original_sha256=original_sha,
                       phase_receipt_alias_path=alias_path.relative_to(root).as_posix(),phase_receipt_alias_sha256=alias_sha)
  alias_fields={k:child_receipt[k] for k in ('phase_receipt_alias_path','phase_receipt_alias_sha256','phase_receipt_original_path','phase_receipt_original_sha256')}
  status='accepted-output-pending-scientific-review'
 except Exception as exc:
  error=f'{type(exc).__name__}: {exc}'
  try: after=_snapshot(job,root,job_path,driver)
  except Exception as e: after={'snapshot_error':f'{type(e).__name__}: {e}'}
 finally:
  stdoutp.write_text(out,encoding='utf-8'); stderrp.write_text(err,encoding='utf-8')
  doc={'schema_version':2,'job_id':jobid,'driver':job.get('driver'),'status':status,'child_returncode':childrc,'error':error,'started_unix':started,'finished_unix':time.time(),'before':before,'after':after,'validator_before':vbefore,'validator_after':vafter,'child_phase_receipt':child_receipt,'stdout_path':stdoutp.relative_to(root).as_posix(),'stderr_path':stderrp.relative_to(root).as_posix(),'accounting':'No dispatcher phase-wall receipt; the child owns its existing phase receipt. Dispatcher wall is orchestration overhead and is not separately added to avoid double-counting.'}
  if status=='accepted-output-pending-scientific-review': doc.update(alias_fields)
  with receipt.open('x',encoding='utf-8') as f: json.dump(doc,f,indent=2,sort_keys=True)
 if status!='accepted-output-pending-scientific-review': raise DispatchError(error or 'dispatch failed; see receipt')
 return doc

def main():
 p=argparse.ArgumentParser(); p.add_argument('--job',required=True); a=p.parse_args()
 try:
  result=dispatch(a.job); print(json.dumps({'job_id':result['job_id'],'status':result['status'],'receipt':str((ROOT/'logs/verified-frozen-dispatch'/f"{result['job_id']}.json"))}))
 except Exception as e:
  print(f'dispatch failed: {type(e).__name__}: {e}',file=sys.stderr); return 2
 return 0
if __name__=='__main__': raise SystemExit(main())


