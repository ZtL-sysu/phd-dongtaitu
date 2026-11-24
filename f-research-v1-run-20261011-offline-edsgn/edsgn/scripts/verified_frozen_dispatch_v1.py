from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRUSTED_PYTHON = Path(r'C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe')
VALIDATOR = ROOT / 'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py'
VALIDATOR_FILES = {
 'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py':'5e88f56801955ded32e00fbc1b916ba4e9237b4c3eb76d70ac4c2cc6fba62525',
 'edsgn/scripts/frozen-pipeline-validator/scripts/citation_requirements.py':'efe945b9d14e9a24b9b18d2d2a9b318d4afff8b9d0683391ea6141017e46a069',
 'edsgn/scripts/frozen-pipeline-validator/scripts/delivery_readiness.py':'92ec18db25fb392ff55c9d164d89ffd0d3572de17c4e0622b515cc7ab5871988',
 'edsgn/scripts/frozen-pipeline-validator/assets/pipeline-definition.json':'1d4aadcea7a2e6eaaf24825c08dcec1433c79a3e6b6323d1424861fee7005e6e'}
DRIVERS = {
 'graph_v6':('edsgn/staging/frozen_graph_state_diagnostic_v6.py', {'--checkpoint','--dataset-path','--output','--policy','--policy-config','--device'}, {'--checkpoint','--dataset-path','--output'}),
 'h2_v4':('edsgn/staging/frozen_h2_matched_retention_v4.py', {'--checkpoint','--dataset-path','--split','--output','--device'}, {'--checkpoint','--dataset-path','--split','--output'}),
 'calibration_v5':('edsgn/staging/frozen_validation_calibration_v5.py', {'--checkpoint','--config','--queue','--epsilon-evidence','--binding','--output','--device','--root'}, {'--checkpoint','--config','--queue','--epsilon-evidence','--binding','--output','--device'}),
 'c06_v4':('edsgn/staging/frozen_split_history_robustness_v4.py', {'--checkpoint','--dataset-path','--domain-manifest','--recipe-json','--output','--split','--policy','--policy-config','--device','--batch-size'}, {'--checkpoint','--dataset-path','--domain-manifest','--recipe-json','--output','--split'})}
PATH_FLAGS={'--checkpoint','--dataset-path','--config','--queue','--epsilon-evidence','--binding','--domain-manifest','--recipe-json','--policy-config','--output','--root'}

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
 if set(flags)!=required: raise DispatchError(f'required flags mismatch; got {flags}')
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
 _,argv=_argv(job,root,allow_existing_output=True)
 for i in range(0,len(argv),2):
  flag,value=argv[i],argv[i+1]
  if flag in PATH_FLAGS and flag!='--output': add('argument',value)
 return {'files':dict(sorted(items.items())),'pipeline_state_sha256':snap['state_bytes_sha256'],'passed_stage_ids':list(snap['passed'])}

def _run(cmd, root, runner=subprocess.run):
 return runner(cmd,cwd=str(root),shell=False,capture_output=True,text=True,check=False)

def dispatch(job_path, *, root=ROOT, python=TRUSTED_PYTHON, runner=subprocess.run):
 root=Path(root).resolve(); job_path=inroot(str(job_path),root)
 if not python.is_file(): raise DispatchError('trusted Conda Python missing')
 job=load_json(job_path)
 if set(job)!={'schema_version','job_id','driver','argv','sources'} or job['schema_version']!=1: raise DispatchError('job schema invalid')
 jobid=job['job_id']
 if not isinstance(jobid,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',jobid): raise DispatchError('invalid job_id')
 if not isinstance(job['sources'],list): raise DispatchError('sources must be list')
 _validate_validator(root)
 driver_rel,argv=_argv(job,root); driver=root/driver_rel
 output_arg=next(argv[i+1] for i in range(0,len(argv),2) if argv[i]=='--output')
 output_path=(root/Path(output_arg)).resolve() if not Path(output_arg).is_absolute() else Path(output_arg).resolve()
 if output_path.exists(): raise DispatchError('output already exists')
 logdir=root/'logs/verified-frozen-dispatch'; logdir.mkdir(parents=True,exist_ok=True)
 receipt=logdir/f'{jobid}.json'; stdoutp=logdir/f'{jobid}.stdout.log'; stderrp=logdir/f'{jobid}.stderr.log'
 if any(p.exists() for p in (receipt,stdoutp,stderrp)): raise DispatchError('dispatch outputs already exist')
 started=time.time(); before=None; after=None; error=None; status='failed'; childrc=None; vbefore=None; vafter=None; out=''; err=''
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
  if before!=after: raise DispatchError('frozen bytes/path membership changed during child')
  vr2=_run(vcmd,root,runner)
  vafter={'returncode':vr2.returncode,'stdout':vr2.stdout,'stderr':vr2.stderr}
  if vr2.returncode!=0: raise DispatchError('official post-child before-P4 validator failed')
  # Validator bytes and state are also in before/after snapshots; run one final direct byte check.
  _validate_validator(root)
  if _snapshot(job,root,job_path,driver)!=before: raise DispatchError('final frozen snapshot changed')
  status='accepted-output-pending-scientific-review'
 except Exception as exc:
  error=f'{type(exc).__name__}: {exc}'
  try: after=_snapshot(job,root,job_path,driver)
  except Exception as e: after={'snapshot_error':f'{type(e).__name__}: {e}'}
 finally:
  stdoutp.write_text(out,encoding='utf-8'); stderrp.write_text(err,encoding='utf-8')
  doc={'schema_version':1,'job_id':jobid,'driver':job.get('driver'),'status':status,'child_returncode':childrc,'error':error,'started_unix':started,'finished_unix':time.time(),'before':before,'after':after,'validator_before':vbefore,'validator_after':vafter,'stdout_path':stdoutp.relative_to(root).as_posix(),'stderr_path':stderrp.relative_to(root).as_posix(),'accounting':'No dispatcher phase-wall receipt; the child owns its existing phase receipt. Dispatcher wall is orchestration overhead and is not separately added to avoid double-counting.'}
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
