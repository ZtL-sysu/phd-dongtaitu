from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRUSTED_PYTHON = Path(r'C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe')
VALIDATOR = ROOT / 'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py'
CANDIDATE_BINDING_HELPER = ROOT / 'edsgn/staging/prospective_training_candidate_binding_v1.py'
VALIDATOR_FILES = {
 'edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py':'5e88f56801955ded32e00fbc1b916ba4e9237b4c3eb76d70ac4c2cc6fba62525',
 'edsgn/scripts/frozen-pipeline-validator/scripts/citation_requirements.py':'efe945b9d14e9a24b9b18d2d2a9b318d4afff8b9d0683391ea6141017e46a069',
 'edsgn/scripts/frozen-pipeline-validator/scripts/delivery_readiness.py':'92ec18db25fb392ff55c9d164d89ffd0d3572de17c4e0622b515cc7ab5871988',
 'edsgn/scripts/frozen-pipeline-validator/assets/pipeline-definition.json':'1d4aadcea7a2e6eaaf24825c08dcec1433c79a3e6b6323d1424861fee7005e6e'}
DRIVERS = {
 'graph_v7':('edsgn/staging/frozen_graph_state_diagnostic_v7.py', {'--checkpoint','--dataset-path','--output','--policy','--policy-config','--device'}, {'--checkpoint','--dataset-path','--output'}),
 'graph_v8':('edsgn/staging/frozen_graph_state_diagnostic_v8.py', {'--checkpoint','--dataset-path','--output','--policy','--policy-config','--device'}, {'--checkpoint','--dataset-path','--output'}),
 'h2_v4':('edsgn/staging/frozen_h2_matched_retention_v4.py', {'--checkpoint','--dataset-path','--split','--output','--device'}, {'--checkpoint','--dataset-path','--split','--output'}),
 'calibration_v9':('edsgn/staging/frozen_validation_calibration_v9.py', {'--checkpoint','--config','--queue','--epsilon-authority','--launch-binding','--binding','--output','--device','--root'}, {'--checkpoint','--config','--queue','--epsilon-authority','--launch-binding','--binding','--output','--device'}),
 'c06_v5':('edsgn/staging/frozen_split_history_robustness_v5.py', {'--checkpoint','--dataset-path','--domain-manifest','--recipe-json','--output','--split','--policy','--policy-config','--device','--batch-size'}, {'--checkpoint','--dataset-path','--domain-manifest','--recipe-json','--output','--split'}),
 'prediction_v5':('edsgn/staging/frozen_prediction_guarded_v5.py', {'--checkpoint','--dataset-path','--binding','--output','--split','--policy','--policy-config','--device','--root'}, {'--checkpoint','--dataset-path','--binding','--output','--split'}),
 'profile_v5':('edsgn/staging/frozen_profile_guarded_v5.py', {'--checkpoint','--dataset-path','--freeze','--output','--split','--policy','--policy-config','--device'}, {'--checkpoint','--dataset-path','--freeze','--output','--split','--device'})}
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
  if flag=='--device' and driver in {'profile_v5','prediction_v5'} and value not in {'cpu','cuda'}:
   raise DispatchError('profile/prediction driver CLI accepts only cpu or cuda')
  i+=2
 if not required.issubset(set(flags)): raise DispatchError(f'required flags missing; got {flags}')
 if job.get('phase_receipt_alias') is None: raise DispatchError('phase_receipt_alias is required')
 test_bound=any(argv[i]=='--split' and argv[i+1]=='test' for i in range(0,len(argv),2)) or job.get('driver') in {'graph_v7','graph_v8','h2_v4','prediction_v5'}
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
 seen_bindings=set();validated_bindings=[]
 for item in bindings:
  if not isinstance(item,dict) or set(item)!={'path','sha256'}: raise DispatchError('malformed pretest execution binding descriptor')
  if item['path'] in seen_bindings: raise DispatchError('duplicate pretest execution binding path')
  seen_bindings.add(item['path'])
  record=add('pretest-execution-binding',item['path'],item['sha256'])
  binding_doc=_validate_execution_binding(root/record['path'],root);validated_bindings.append(binding_doc)
  for role,rel,digest in binding_doc['_verified_file_refs']:
   add('pretest-'+role,rel,digest)
 alias=job.get('phase_receipt_alias')
 alias_path=_validate_alias(root,alias,require_absent=False)
 _,argv=_argv(job,root,allow_existing_output=True)
 _validate_job_binding(job,argv,validated_bindings,root)
 schema4_bindings=[b for b in validated_bindings if b.get('_verified_plan_doc',{}).get('schema_version')==4]
 if schema4_bindings:
  queue_refs={json.dumps(b.get('global_test_queue'),sort_keys=True) for b in schema4_bindings}
  if len(queue_refs)!=1: raise DispatchError('one dispatcher job cannot mix global campaign queue bindings')
  queue_ref=schema4_bindings[0].get('global_test_queue')
  helper_path=ROOT/'edsgn/staging/global_frozen_test_completion_v1.py'
  spec=importlib.util.spec_from_file_location('_global_test_completion_for_dispatch_v8',helper_path)
  if spec is None or spec.loader is None: raise DispatchError('global test completion verifier is unavailable')
  helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
  try:
   queued=helper.verify_queued_job(root,queue_ref,job_path)
  except Exception as exc:
   raise DispatchError(f'job is not a member of the frozen global test queue: {exc}') from exc
  if not isinstance(queued,dict) or not queued: raise DispatchError('global queue verifier returned no byte refs')
  for rel,digest in queued.items(): add('global-queue-membership',rel,digest)
 for i in range(0,len(argv),2):
  flag,value=argv[i],argv[i+1]
  if flag in PATH_FLAGS and flag!='--output': add('argument',value)
 return {'files':dict(sorted(items.items())),'pipeline_state_sha256':snap['state_bytes_sha256'],'passed_stage_ids':list(snap['passed']),
         'phase_receipt_alias_path':alias_path.relative_to(root).as_posix()}

def _validate_execution_binding(path, root=None):
 root=Path(root or ROOT).resolve(); path=Path(path).resolve()
 doc=load_json(path)
 top={'schema_version','status','scope','endpoint_id','plan_path','plan_sha256','registry_path','registry_sha256','test_feedback_used','runs'}
 plan_doc=load_json(inroot(doc.get('plan_path',''),root)) if isinstance(doc.get('plan_path'),str) else {}
 schema_version=plan_doc.get('schema_version')
 if schema_version==4: top.add('global_test_queue')
 if (set(doc)!=top or doc.get('schema_version')!=2 or doc.get('status')!='accepted_by_root' or
     doc.get('scope')!='pretest_execution_binding' or doc.get('test_feedback_used') is not False):
  raise DispatchError('pretest execution binding must match typed endpoint v8 accepted input-only schema')
 if not isinstance(doc.get('endpoint_id'),str) or not doc['endpoint_id']:
  raise DispatchError('pretest execution binding endpoint_id missing')
 def checked_ref(ref,label):
  if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not isinstance(ref['path'],str):
   raise DispatchError('malformed pretest binding '+label+' reference')
  p=inroot(ref['path'],root)
  if not p.is_file() or not isinstance(ref['sha256'],str) or len(ref['sha256'])!=64 or sha(p)!=ref['sha256']:
   raise DispatchError('pretest binding '+label+' SHA-256 mismatch')
  return p.relative_to(root).as_posix(),ref['sha256']
 for label in ('plan','registry'):
  p,digest=checked_ref({'path':doc[label+'_path'],'sha256':doc[label+'_sha256']},label)
  if label=='plan' and p!=doc['plan_path']: raise DispatchError('pretest plan path mismatch')
  if label=='registry' and p!=doc['registry_path']: raise DispatchError('pretest registry path mismatch')
 plan_doc=load_json(root/doc['plan_path']); registry_doc=load_json(root/doc['registry_path'])
 schema_version=plan_doc.get('schema_version')
 candidate_helper=None
 if schema_version==4:
  spec=importlib.util.spec_from_file_location('_candidate_binding_helper_dispatch_v8',CANDIDATE_BINDING_HELPER)
  if spec is None or spec.loader is None: raise DispatchError('prospective candidate-binding helper is unavailable')
  candidate_helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate_helper;spec.loader.exec_module(candidate_helper)
 if plan_doc.get('endpoint_id')!=doc['endpoint_id']: raise DispatchError('pretest binding endpoint differs from frozen plan')
 registry_rows=registry_doc.get('endpoint_plans')
 if (registry_doc.get('artifact_status')!='accepted_pretest_registry' or registry_doc.get('split')!='test' or
     not isinstance(registry_rows,list) or len(registry_rows)!=49): raise DispatchError('pretest registry is not the accepted 49-row test registry')
 current=[r for r in registry_rows if isinstance(r,dict) and r.get('endpoint_id')==doc['endpoint_id']]
 if len(current)!=1 or current[0].get('path')!=doc['plan_path'] or current[0].get('sha256')!=doc['plan_sha256']:
  raise DispatchError('registry does not bind pretest endpoint plan')
 runs=doc.get('runs')
 if not isinstance(runs,list) or len(runs)!=10:
  raise DispatchError('pretest binding must enumerate both sides and all five seeds')
 roles=('checkpoint','freeze','input_binding','validation_authority','policy','queue_plan')
 allowed={'side','seed_id','config','training_binding',*roles,'block_length','acf_dependence_length','acf_authority_path','epsilon_status','epsilon_value_raw','epsilon_authority_path','test_feedback_used'}
 seen=set(); refs=[('plan',*checked_ref({'path':doc['plan_path'],'sha256':doc['plan_sha256']},'plan')),('registry',*checked_ref({'path':doc['registry_path'],'sha256':doc['registry_sha256']},'registry'))]
 for source in plan_doc.get('producer_sources',[]): refs.append(('producer-source',*checked_ref(source,'producer source')))
 domain=plan_doc.get('domain',{})
 if isinstance(domain,dict) and {'path','sha256'}<=set(domain): refs.append(('domain-manifest',*checked_ref({'path':domain['path'],'sha256':domain['sha256']},'domain manifest')))
 collection=plan_doc.get('dataset_collection',{})
 if isinstance(collection,dict):
  for item in collection.get('files',[]): refs.append(('dataset-member',*checked_ref(item,'dataset member')))
 acf_plan=plan_doc.get('acf_evidence',{})
 if isinstance(acf_plan,dict) and acf_plan.get('mode')=='fixed_validation_evidence': refs.append(('acf-authority',*checked_ref({'path':acf_plan['path'],'sha256':acf_plan['sha256']},'ACF authority')))
 eps_plan=plan_doc.get('epsilon_evidence')
 if isinstance(eps_plan,dict) and eps_plan.get('mode')=='fixed_validation_evidence': refs.append(('epsilon-authority',*checked_ref({'path':eps_plan['path'],'sha256':eps_plan['sha256']},'epsilon authority')))
 switch_plan=plan_doc.get('reporting_switch_authority')
 if switch_plan is not None:
  switch_path,switch_digest=checked_ref(switch_plan,'reporting switch authority')
  refs.append(('reporting-switch-authority',switch_path,switch_digest))
  switch_doc=load_json(root/switch_path)
  if (set(switch_doc)!={'schema','scope','dataset_id','join_rule','entries'}
      or switch_doc.get('schema')!='edsgn.reporting-switch-authority.v1'
      or switch_doc.get('scope')!='reporting_only_postrun'
      or switch_doc.get('join_rule')!='trajectory_index_exact_one'
      or not isinstance(switch_doc.get('entries'),list)):
   raise DispatchError('reporting switch authority has invalid report-only descriptor schema')
  for entry in switch_doc['entries']:
   if not isinstance(entry,dict) or set(entry)!={'trajectory_index','path','sha256','field'}:
    raise DispatchError('reporting switch authority has malformed source reference')
   refs.append(('reporting-switch-source',*checked_ref({'path':entry['path'],'sha256':entry['sha256']},'reporting switch source')))
 if schema_version==4:
  campaign=plan_doc.get('global_test_campaign')
  queue_ref=doc.get('global_test_queue')
  if not isinstance(campaign,dict) or set(campaign)!={'queue_path','completion_path'}:
   raise DispatchError('schema4 plan lacks frozen global test campaign paths')
  queue_path,queue_digest=checked_ref(queue_ref,'global test queue')
  if queue_path!=_resolve_rel(root,campaign['queue_path']): raise DispatchError('global test queue path differs from P3A plan')
  refs.append(('global-test-queue',queue_path,queue_digest))
 candidate_source=None
 if schema_version==4:
  candidate_source=next((x for x in plan_doc.get('producer_sources',[]) if x.get('path')=='edsgn/staging/prospective_training_candidate_binding_v1.py'),None)
  if candidate_source is None: raise DispatchError('schema4 plan omits pinned candidate-binding helper source')
  refs.append(('candidate-binding-helper',*checked_ref(candidate_source,'candidate-binding helper')))
  completion_helper=next((x for x in plan_doc.get('producer_sources',[]) if x.get('path')=='edsgn/staging/global_frozen_test_completion_v1.py'),None)
  if completion_helper is None: raise DispatchError('schema4 plan omits pinned global completion verifier source')
  refs.append(('global-completion-helper',*checked_ref(completion_helper,'global completion verifier')))
 for row in runs:
  row_allowed=set(allowed)
  if schema_version==4: row_allowed.add('candidate_selection')
  if not isinstance(row,dict) or set(row)!=row_allowed or type(row.get('seed_id')) is not int or row.get('side') not in {'left','right'}:
   raise DispatchError('pretest binding row is malformed or contains post-test artifacts')
  key=(row['side'],row['seed_id'])
  if key in seen or row['seed_id'] not in range(5): raise DispatchError('pretest binding has duplicate or invalid side/seed row')
  seen.add(key)
  side_desc=plan_doc.get(row['side']); seeds=side_desc.get('seeds') if isinstance(side_desc,dict) else None
  seed_desc=next((x for x in seeds or [] if isinstance(x,dict) and x.get('seed_id')==row['seed_id']),None)
  if not isinstance(seed_desc,dict): raise DispatchError('pretest binding seed is absent from frozen plan')
  if schema_version==3:
   if row.get('config')!=seed_desc.get('preparation',{}).get('config') or row.get('training_binding')!=seed_desc.get('preparation',{}).get('training_binding'):
    raise DispatchError('pretest config/training binding differs from P3A plan')
  elif schema_version==4:
   preparation_hashes=candidate_helper.validate_preparation(root,seed_desc['preparation'],row['seed_id'],allow_postrun_outputs=True)
   if not isinstance(preparation_hashes,dict) or not preparation_hashes: raise DispatchError('candidate helper returned no prospective preparation hashes')
   for rel,digest in preparation_hashes.items():
    p=inroot(rel,root)
    if not p.is_file() or sha(p)!=digest: raise DispatchError('prospective candidate preparation hash mismatch: '+rel)
    refs.append(('candidate-preparation',rel,digest))
   selected_refs={key:row.get(key) for key in ('candidate_selection','config','training_binding')}
   selected_hashes=candidate_helper.validate_selected_training(root,seed_desc['preparation'],row['seed_id'],selected_refs)
   if not isinstance(selected_hashes,dict) or not selected_hashes: raise DispatchError('candidate helper returned no selected provenance hashes')
   for rel,digest in selected_hashes.items():
    p=inroot(rel,root)
    if not p.is_file() or sha(p)!=digest: raise DispatchError('selected candidate provenance hash mismatch: '+rel)
    refs.append(('candidate-selection-proof',rel,digest))
  else: raise DispatchError('unsupported typed endpoint plan schema in dispatcher')
  planned=seed_desc.get('planned_paths',{})
  for role in roles:
   if not isinstance(planned,dict) or role not in planned or _resolve_rel(root,row[role]['path'])!=_resolve_rel(root,planned[role]):
    raise DispatchError('pretest '+role+' path differs from P3A prospective planned_paths')
  if row.get('test_feedback_used') is not False: raise DispatchError('pretest binding declares test feedback')
  for role in ('config','training_binding',*roles): refs.append((role,*checked_ref(row.get(role),role)))
  if schema_version==4:
   refs.append(('candidate-selection',*checked_ref(row.get('candidate_selection'),'candidate selection')))
  block=row.get('block_length'); dep=row.get('acf_dependence_length')
  if type(block) is not int or type(dep) is not int or block<1 or dep<1 or block!=max(plan_doc.get('history',0),plan_doc.get('max_horizon',0),dep):
   raise DispatchError('pretest binding block/dependence lengths violate the frozen history/horizon rule')
  domain=plan_doc.get('domain',{}); bounds=domain.get('bounds',[]) if isinstance(domain,dict) else []
  if not bounds or any(block>b.get('origin_count',0) for b in bounds): raise DispatchError('pretest binding block exceeds frozen origin support')
  acf=plan_doc.get('acf_evidence')
  if not isinstance(acf,dict): raise DispatchError('frozen plan ACF authority missing')
  if acf.get('mode')=='fixed_validation_evidence':
   if dep!=plan_doc.get('acf_dependence_length') or block!=plan_doc.get('block_length') or row.get('acf_authority_path')!=acf.get('path'):
    raise DispatchError('pretest binding changes frozen ACF/block authority')
  elif acf.get('mode')=='validation_derived':
   if row.get('acf_authority_path')!=acf.get('expected_record_path') or _resolve_rel(root,row.get('acf_authority_path'))!=_resolve_rel(root,row['validation_authority']['path']):
    raise DispatchError('derived ACF path is not covered by the row validation-authority hash')
  else: raise DispatchError('unsupported frozen ACF authority')
  eps=plan_doc.get('epsilon_evidence')
  if plan_doc.get('outcome_type')=='forecast_mae_noninferiority':
   if not isinstance(eps,dict): raise DispatchError('frozen NI endpoint lacks epsilon authority')
   status=row.get('epsilon_status'); value=row.get('epsilon_value_raw')
   if eps.get('mode')=='fixed_validation_evidence':
    if status!='accepted' or value!=eps.get('value_raw') or row.get('epsilon_authority_path')!=eps.get('path'): raise DispatchError('pretest binding changes fixed epsilon authority')
   elif eps.get('mode')=='validation_derived':
    if row.get('epsilon_authority_path')!=eps.get('expected_record_path') or _resolve_rel(root,row.get('epsilon_authority_path'))!=_resolve_rel(root,row['validation_authority']['path']): raise DispatchError('derived epsilon path is not covered by the row validation-authority hash')
   elif eps.get('mode')=='unidentifiable_by_rule':
    if status!='unidentifiable' or value is not None or row.get('epsilon_authority_path')!=eps.get('expected_record_path') or _resolve_rel(root,row.get('epsilon_authority_path'))!=_resolve_rel(root,row['validation_authority']['path']): raise DispatchError('unidentifiable epsilon rule is not covered by the row validation-authority hash')
   else: raise DispatchError('unsupported frozen epsilon authority')
  elif row.get('epsilon_status') is not None or row.get('epsilon_value_raw') is not None or row.get('epsilon_authority_path') is not None:
   raise DispatchError('non-NI pretest binding must not contain epsilon fields')
  if row.get('epsilon_status') not in {None,'accepted','unidentifiable'}: raise DispatchError('invalid pretest epsilon status')
  value=row.get('epsilon_value_raw')
  if row.get('epsilon_status')=='accepted' and (isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0):
   raise DispatchError('accepted pretest epsilon must be finite and positive')
  if row.get('epsilon_status') in {None,'unidentifiable'} and value is not None: raise DispatchError('nonaccepted pretest epsilon must be null')
 expected={(side,seed) for side in ('left','right') for seed in range(5)}
 if seen!=expected: raise DispatchError('pretest binding omits or adds side/seed rows')
 doc['_verified_file_refs']=refs;doc['_verified_plan_doc']=plan_doc;doc['_verified_binding_path']=path.relative_to(root).as_posix()
 return doc

def _resolve_rel(root, raw):
 p=Path(raw);p=(root/p).resolve() if not p.is_absolute() else p.resolve()
 if not p.is_relative_to(root.resolve()):raise DispatchError('frozen plan path escapes run root')
 return p.relative_to(root).as_posix()

def _validate_job_binding(job,argv,bindings,root):
 flags={argv[i]:argv[i+1] for i in range(0,len(argv),2)}
 is_test=flags.get('--split')=='test' or job.get('driver') in {'graph_v7','graph_v8','h2_v4','prediction_v5'}
 if not is_test:return
 if not bindings:raise DispatchError('test execution has no typed endpoint pretest binding')
 required_cli={'--checkpoint','--output','--dataset-path'}
 if job.get('driver')!='h2_v4':required_cli.add('--policy-config')
 if not required_cli.issubset(flags):raise DispatchError('test job omits checkpoint/policy/dataset/output binding flags')
 if '--split' in flags and flags['--split']!='test':raise DispatchError('test endpoint plan cannot dispatch a non-test split')
 output_rel=_resolve_rel(root,flags['--output']);alias=_resolve_rel(root,job.get('phase_receipt_alias'))
 source_items=job.get('sources')
 if not isinstance(source_items,list):raise DispatchError('test job source declarations must be a list')
 source_hashes={}
 for source in source_items:
  if not isinstance(source,dict) or set(source)!={'path','sha256'}:raise DispatchError('malformed declared producer source')
  rel,digest=source['path'],source['sha256']
  if not isinstance(rel,str) or not isinstance(digest,str) or not (root/rel).is_file() or sha(root/rel)!=digest:
   raise DispatchError('declared producer source hash mismatch')
  source_hashes[_resolve_rel(root,rel)]=digest
 for binding in bindings:
  plan=binding['_verified_plan_doc'];matches=[]
  if plan.get('domain',{}).get('split')!='test':raise DispatchError('test job binding plan is not for test split')
  collection=plan.get('dataset_collection',{});collection_root=collection.get('root')
  dataset_rel=_resolve_rel(root,flags['--dataset-path'])
  allowed_data={_resolve_rel(root,collection_root)} if isinstance(collection_root,str) else set()
  file_rows=collection.get('files',[]) if isinstance(collection,dict) else []
  if isinstance(file_rows,list):allowed_data.update(_resolve_rel(root,r['path']) for r in file_rows if isinstance(r,dict) and isinstance(r.get('path'),str))
  if dataset_rel not in allowed_data:raise DispatchError('test dataset path is not the bound collection root or member')
  if '--domain-manifest' in flags and _resolve_rel(root,flags['--domain-manifest'])!=_resolve_rel(root,plan['domain']['path']):
   raise DispatchError('test domain manifest differs from typed endpoint plan')
  for side in ('left','right'):
   for seed in plan.get(side,{}).get('seeds',[]):
    if not isinstance(seed,dict):continue
    seed_id=seed.get('seed_id');planned=seed.get('planned_paths',{})
    row=next((r for r in binding['runs'] if r['side']==side and r['seed_id']==seed_id),None)
    if row is None or _resolve_rel(root,row['checkpoint']['path'])!=_resolve_rel(root,flags['--checkpoint']):continue
    role_flags={'--policy-config':'policy','--binding':'input_binding','--freeze':'freeze','--config':'config','--queue':'queue_plan','--epsilon-authority':'validation_authority'}
    if any(flag in flags and _resolve_rel(root,flags[flag])!=_resolve_rel(root,row[role]['path']) for flag,role in role_flags.items()):continue
    if not all(role in planned and _resolve_rel(root,row[role]['path'])==_resolve_rel(root,planned[role]) for role in ('checkpoint','freeze','input_binding','validation_authority','policy','queue_plan')):continue
    if alias!=_resolve_rel(root,planned.get('phase_receipt','')):continue
    implicit_freeze=(root/Path(flags['--checkpoint'])).resolve().parent/'test-freeze.json'
    if not implicit_freeze.is_file() or _resolve_rel(root,row['freeze']['path'])!=implicit_freeze.relative_to(root).as_posix():continue
    if '--freeze' in flags and _resolve_rel(root,flags['--freeze'])!=implicit_freeze.relative_to(root).as_posix():continue
    output_roles=[role for role in ('driver_report','archive','csv','trace') if role in planned]
    if not output_roles:continue
    parents={_resolve_rel(root,planned[role]).rsplit('/',1)[0] for role in output_roles}
    if parents!={output_rel}:continue
    policy_doc=load_json(root/row['policy']['path'])
    effective_policy=flags.get('--policy','full' if job.get('driver')=='h2_v4' else policy_doc.get('kind'))
    if policy_doc.get('kind')!=effective_policy:continue
    producer_sha=plan.get(side,{}).get('producer_source_sha256')
    producer_rows=plan.get('producer_sources',[])
    driver_rel=DRIVERS[job.get('driver')][0]
    expected_driver=driver_rel
    if plan.get('schema_version')==4 and plan.get('outcome_type') in {'graph_tv_noise_superiority','graph_tv_response_superiority'}:
     expected_driver='edsgn/staging/frozen_graph_state_diagnostic_v8.py'
    elif plan.get('schema_version')==3 and plan.get('outcome_type') in {'graph_tv_noise_superiority','graph_tv_response_superiority'}:
     expected_driver='edsgn/staging/frozen_graph_state_diagnostic_v7.py'
    if driver_rel!=expected_driver: continue
    driver_path=root/driver_rel
    if not driver_path.is_file():continue
    driver_sha=sha(driver_path)
    exact_producer_rows=[x for x in producer_rows if isinstance(x,dict) and set(x)=={'path','sha256'}
                         and isinstance(x.get('path'),str) and _resolve_rel(root,x['path'])==driver_rel
                         and x.get('sha256')==driver_sha]
    if (producer_sha!=driver_sha or len(exact_producer_rows)!=1
            or source_hashes.get(driver_rel)!=driver_sha):continue
    if plan.get('schema_version')==4:
     helper_rel='edsgn/staging/prospective_training_candidate_binding_v1.py'
     helper_rows=[x for x in producer_rows if isinstance(x,dict) and x.get('path')==helper_rel]
     if len(helper_rows)!=1 or source_hashes.get(helper_rel)!=helper_rows[0].get('sha256'): continue
    if plan.get('outcome_type')=='paired_window_cost_superiority':
     expected_device=plan.get('profile_device')
     raw_device=flags.get('--device','cuda')
     normalized_device=('cpu' if raw_device=='cpu' else 'cuda' if raw_device in {'cuda','cuda:0'} else None)
     if expected_device not in {'cpu','cuda'} or normalized_device!=expected_device:continue
    matches.append((side,seed_id))
  if not matches:raise DispatchError('test job argv/source/output do not match a bound side/seed prospective endpoint run')
 return True

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
  'graph_v8':['predictions.npz','counter-rows.json','frozen-graph-state-diagnostic.json','trace-rows.jsonl.gz','metrics_by_horizon.csv'],
  'h2_v4':['h2-matched-retention-report.json'],
  'calibration_v9':['calibration.json','validation-predictions.npz','calibration-validation-dependence-v2.json'],
  'c06_v5':['predictions.npz','metrics_by_horizon.csv','counter-rows.json','perturbation-manifest.json'],
  'prediction_v5':['predictions.npz','evaluation_manifest.json'],
  'profile_v5':['runtime_summary.json','window_timings.csv','trace.jsonl','profile-budget-accounting.json','profile-guard-overhead.json']}
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
 wanted={'calibration_v9':'calibration','profile_v5':'profiling'}.get(driver,'evaluation')
 if doc.get('scope')!='offline-experiment-phase' or doc.get('phase')!=wanted or doc.get('status')!='completed':
  raise DispatchError('child phase receipt is not a completed expected phase')
 if not isinstance(doc.get('job_id'),str) or not doc['job_id']: raise DispatchError('child receipt job_id missing')
 if driver in {'h2_v4','c06_v5'} and 'report_sha256' in doc:
  name='h2-matched-retention-report.json' if driver=='h2_v4' else None
  if name and doc['report_sha256']!=sha(output/name): raise DispatchError('child receipt report hash mismatch')
 if driver in {'graph_v7','graph_v8'}:
  if not isinstance(doc.get('origin_count'),int) or doc['origin_count']<=0: raise DispatchError('graph child receipt has no completed origins')
  report=load_json(output/'frozen-graph-state-diagnostic.json')
  if report.get('predictions_file')!='predictions.npz' or not isinstance(report.get('full_test_origin_count'),int) or report['full_test_origin_count']!=doc['origin_count']:
   raise DispatchError('graph principal report/receipt mismatch')
  if driver=='graph_v8' and (report.get('scope')!='frozen_test_replay_oracle_free' or report.get('split')!='test'
      or report.get('oracle_accessed') is not False or report.get('oracle_npz_decoded') is not False
      or report.get('oracle_analysis_status')!='deferred_until_global_completion'):
   raise DispatchError('graph v8 report is not oracle-free/deferred full-test replay')
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
  dependence=load_json(output/'calibration-validation-dependence-v2.json')
  pred_path=output/'validation-predictions.npz'; pred_sha=sha(pred_path)
  with np.load(pred_path,allow_pickle=False) as archive:
   if 'origins' not in archive.files: raise DispatchError('calibration validation predictions omit origins')
   saved_origin_count=len(archive['origins'])
  if (type(dependence.get('validation_origin_count')) is not int or
      dependence['validation_origin_count']<=0 or dependence['validation_origin_count']!=saved_origin_count):
   raise DispatchError('calibration dependence sidecar origin count differs from saved validation predictions')
  if dependence.get('test_origins_evaluated') is not False or dependence.get('test_feedback_used') is not False:
   raise DispatchError('calibration dependence sidecar reports test feedback/access')
  if dependence.get('validation_prediction_sha256')!=pred_sha:
   raise DispatchError('calibration dependence sidecar does not bind its validation prediction artifact')
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
 if driver=='profile_v5':
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
  if _snapshot(job,root,job_path,driver)!=before:
   raise DispatchError('frozen source/input snapshot changed during official before-P4 validation')
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


