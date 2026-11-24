"""Fail-closed staging producer for one registered primary endpoint.

Staging only. The public entry requires canonical RUN and official metadata
validation; tests use private fixture helpers and never grant experiment authority.
"""
from __future__ import annotations
import csv, gzip, hashlib, importlib.util, json, math, os, re, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
TRUSTED_PYTHON=Path(r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe")
VALIDATOR_PATH=ROOT/"edsgn/staging/typed_endpoint_plan_validator_v9.py"
CANDIDATE_BINDING_HELPER=ROOT/"edsgn/staging/prospective_training_candidate_binding_v1.py"
CONVERTER_PATH=ROOT/"edsgn/staging/endpoint_sufficient_statistics_v4.py"
FORECAST_KERNEL_PATH=ROOT/"edsgn/staging/paired_forecast_bootstrap_v1.py"
SCALAR_KERNEL_PATH=ROOT/"edsgn/staging/paired_scalar_bootstrap_v1.py"
SOURCE_PATHS=(Path(__file__).resolve(),VALIDATOR_PATH,CONVERTER_PATH,FORECAST_KERNEL_PATH,SCALAR_KERNEL_PATH,
 ROOT/"edsgn/staging/frozen_graph_state_diagnostic_v7.py",
 ROOT/"edsgn/staging/frozen_graph_state_diagnostic_v8.py",
 ROOT/"edsgn/staging/frozen_h2_matched_retention_v4.py",
 ROOT/"edsgn/staging/frozen_prediction_guarded_v5.py",
 ROOT/"edsgn/staging/frozen_profile_guarded_v5.py",
 ROOT/"edsgn/scripts/verified_frozen_dispatch_v10.py", CANDIDATE_BINDING_HELPER,
 ROOT/"edsgn/staging/global_frozen_test_completion_v1.py",
 ROOT/"edsgn/scripts/bind_training_inputs.py",
 ROOT/"edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py")

class ProducerError(RuntimeError): pass

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def _strict_json(path):
 def pairs(items):
  out={}
  for k,v in items:
   if k in out: raise ProducerError(f"duplicate JSON key {k}: {path}")
   out[k]=v
  return out
 try: value=json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ProducerError(f"nonfinite JSON constant {x}")))
 except (OSError,json.JSONDecodeError) as exc: raise ProducerError(f"invalid JSON {path}: {exc}") from exc
 if not isinstance(value,dict): raise ProducerError(f"expected JSON object: {path}")
 return value

def _load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None: raise ProducerError(f"cannot load helper {path}")
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def _expected_pairs(plan):
 rows=[]
 for b in plan['domain']['bounds']:
  rows.extend((int(b['trajectory_index']),o) for o in range(int(b['first_origin']),int(b['last_origin'])+1))
 pairs=np.asarray(rows,dtype=np.int64)
 if not len(pairs): raise ProducerError('empty declared test-origin support')
 return pairs

def _artifact_index(root,manifest):
 out={}
 for row in manifest['run_artifacts']:
  key=(row['side'],int(row['seed_id']))
  if key in out: raise ProducerError('duplicate run artifact group')
  out[key]={}
  for role,ref in row['artifacts'].items(): out[key][role]=(root/ref['path'],ref['sha256'])
 return out

def _run_rows(root,manifest):
 """Postrun roles plus the exact pretest refs from the bound execution binding."""
 bp=_bound_ref(root,manifest,{})
 binding=_strict_json(bp)
 pre={(r['side'],int(r['seed_id'])):r for r in binding['runs']}
 rows={}
 for r in manifest['run_artifacts']:
  key=(r['side'],int(r['seed_id']))
  rows[key]={role:(root/ref['path'],ref['sha256']) for role,ref in r['artifacts'].items()}
  for role in ('config','training_binding','checkpoint','freeze','input_binding','validation_authority','policy','queue_plan'):
   ref=pre[key].get(role)
   if not isinstance(ref,dict) or set(ref)!={'path','sha256'}: raise ProducerError(f'pretest execution binding lacks {role}')
   rows[key][role]=(root/ref['path'],ref['sha256'])
  if 'candidate_selection' in pre[key]:
   ref=pre[key]['candidate_selection']
   if not isinstance(ref,dict) or set(ref)!={'path','sha256'}: raise ProducerError('pretest execution binding has malformed candidate selection ref')
   rows[key]['candidate_selection']=(root/ref['path'],ref['sha256'])
 return rows

def _bound_ref(root,manifest,key):
 ref=manifest['pretest_execution_binding']
 p=root/ref['path']
 if sha256(p)!=ref['sha256']: raise ProducerError('execution binding changed')
 return p

def _verify_h2_proof(path,plan,seed,pairs):
 arms={'informative','shuffled','constant','reset'}
 expected={(int(t),int(o),step,arm) for t,o in pairs for step in range(int(plan['history'])) for arm in arms}
 seen=set()
 with Path(path).open('r',encoding='utf-8') as stream:
  for line in stream:
   try: row=json.loads(line)
   except json.JSONDecodeError as exc: raise ProducerError('malformed H2 graph/action proof row') from exc
   key=(row.get('dataset_index'),row.get('origin'),row.get('history_step'),row.get('arm'))
   if (any(type(x)is not int for x in key[:3]) or key not in expected or key in seen
       or row.get('model_seed_id')!=seed or row.get('graph_match_proved') is not True
       or row.get('teacher_graph_action_sha256')!=row.get('arm_graph_action_sha256')):
    raise ProducerError('H2 proof has invalid, duplicate, incomplete, or mismatched graph/action rows')
   seen.add(key)
 return seen==expected

def _semantic_receipts(root,plan,manifest,artifact_index,plan_validator,proof):
 """Recheck postrun semantics not asserted by the path/hash validator."""
 binding_path=_bound_ref(root,manifest,proof)
 binding=_strict_json(binding_path)
 if binding.get('test_feedback_used') is not False: raise ProducerError('test feedback is declared in pretest binding')
 pairs=_expected_pairs(plan); expected_count=len(pairs)
 byrun={(r['side'],int(r['seed_id'])):r for r in binding['runs']}
 proof_rows=[]
 for side in ('left','right'):
  for seed in range(5):
   arts=artifact_index[(side,seed)]; row=byrun[(side,seed)]
   alias_p=arts['phase_receipt'][0]; original_p=arts['phase_receipt_original'][0]
   alias=_strict_json(alias_p); original=_strict_json(original_p)
   if alias_p.read_bytes()!=original_p.read_bytes(): raise ProducerError('phase receipt alias/original bytes differ')
   if alias.get('status')!='completed' or alias.get('job_id')!=original.get('job_id') or alias.get('output_dir')!=original.get('output_dir'):
    raise ProducerError('completed phase receipt identity/status mismatch')
   dispatch=_strict_json(arts['dispatch_receipt'][0])
   if dispatch.get('status')!='accepted-output-pending-scientific-review' or dispatch.get('child_returncode')!=0:
    raise ProducerError('dispatch child did not return successfully')
   if dispatch.get('before')!=dispatch.get('after'):
    raise ProducerError('dispatch before/after frozen input snapshots differ')
   for label in ('validator_before','validator_after'):
    if not isinstance(dispatch.get(label),dict) or dispatch[label].get('returncode')!=0:
     raise ProducerError(f'{label} official validator did not return zero')
   before=dispatch.get('before',{}).get('files',{})
   for role in ('config','training_binding','checkpoint','freeze','input_binding','validation_authority','policy','queue_plan'):
    ref=row.get(role)
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}: raise ProducerError(f'pretest binding lacks {role} before semantic receipt checks')
    if before.get(ref['path'])!=ref['sha256']: raise ProducerError(f'dispatch before snapshot omits pretest {role}')
   if plan.get('schema_version')==4:
    queue_ref=binding.get('global_test_queue')
    if not isinstance(queue_ref,dict) or before.get(queue_ref.get('path'))!=queue_ref.get('sha256'):
     raise ProducerError('dispatch before snapshot omits the root-frozen global test queue')
    prep=next(seed_row['preparation'] for seed_row in plan[side]['seeds'] if seed_row['seed_id']==seed)
    selected_refs={key:row.get(key) for key in ('candidate_selection','config','training_binding')}
    candidate_helper=plan_validator._candidate_helper()
    candidate_refs=candidate_helper.validate_selected_training(root,prep,seed,selected_refs)
    for rel,digest in candidate_refs.items():
     if before.get(rel)!=digest: raise ProducerError(f'dispatch before snapshot omits selected candidate provenance: {rel}')
    if (not CANDIDATE_BINDING_HELPER.is_file() or
        not any(x.get('path')=='edsgn/staging/prospective_training_candidate_binding_v1.py'
                and x.get('sha256')==sha256(CANDIDATE_BINDING_HELPER) for x in plan.get('producer_sources',[]))):
     raise ProducerError('schema4 plan does not source-pin the prospective candidate-binding helper')
   cp,cp_sha=arts['checkpoint']; freeze=_strict_json(arts['freeze'][0])
   if sha256(cp)!=cp_sha or freeze.get('checkpoint_sha256')!=cp_sha:
    raise ProducerError('freeze/checkpoint bytes do not match')
   report_path=arts['driver_report'][0]; report=_strict_json(report_path)
   if Path(alias['output_dir']).resolve()!=report_path.resolve().parent: raise ProducerError('driver report is outside the completed phase output directory')
   phase=alias.get('phase')
   outcome=plan['outcome_type']
   if outcome=='paired_window_cost_superiority':
    if (phase!='profiling' or report.get('scope')!='full_selected_split' or report.get('split')!='test'
        or report.get('smoke_limit') is not None or report.get('batch_size')!=1 or report.get('dtype')!='float32'
        or report.get('repeats')!=5 or report.get('warmup_windows')!=10 or report.get('weights_frozen') is not True):
     raise ProducerError('cost driver is not full B1/FP32/10-warmup/5-repeat test profile v4 scope')
    n=report.get('n_windows')
   elif outcome in ('graph_tv_noise_superiority','graph_tv_response_superiority'):
    expected_scope='frozen_test_replay_oracle_free' if plan.get('schema_version')==4 else 'frozen_test_diagnostic'
    if phase!='evaluation' or report.get('scope')!=expected_scope or report.get('split')!='test': raise ProducerError('graph diagnostic driver scope/split mismatch')
    if plan.get('schema_version')==4 and (report.get('oracle_accessed') is not False
       or report.get('oracle_npz_decoded') is not False
       or report.get('oracle_archive_bytes_hashed_only') is not True
       or report.get('oracle_analysis_status')!='deferred_until_global_completion'):
     raise ProducerError('graph8 replay must defer all oracle analysis until global completion')
    n=report.get('full_test_origin_count')
   elif _is_h2_matched(plan):
    if phase!='evaluation' or report.get('scope')!='frozen_h2_matched_retention_diagnostic' or report.get('split')!='test': raise ProducerError('H2 driver scope/split mismatch')
    if report.get('informative_teacher_parity_all_windows') is not True: raise ProducerError('H2 informative teacher parity was not proved for all windows')
    n=report.get('origin_count')
   else:
    if phase!='evaluation' or report.get('scope') not in {'full_protocol','frozen_full_origin_evaluation'} or report.get('split')!='test': raise ProducerError('forecast driver is not a full test replay')
    n=report.get('n_windows',report.get('full_test_origin_count'))
   if n!=expected_count: raise ProducerError('driver report does not certify exact full origin count')
   reported_cp=report.get('checkpoint_sha256',report.get('checkpoint'))
   if outcome=='paired_window_cost_superiority':
    reported_cp=freeze.get('checkpoint_sha256')
   if reported_cp!=cp_sha: raise ProducerError('driver report checkpoint bytes differ from frozen checkpoint')
   before_hash=next((report.get(k) for k in ('parameters_sha256_before','parameter_sha256_before','parameters_before','parameter_hash_before')),None)
   after_hash=next((report.get(k) for k in ('parameters_sha256_after','parameter_sha256_after','parameters_after','parameter_hash_after')),None)
   if not isinstance(before_hash,str) or not before_hash or before_hash!=after_hash:
    raise ProducerError('driver report lacks equal before/after parameter hashes')
   if freeze.get('parameters_sha256')!=before_hash:
    raise ProducerError('driver parameter hash differs from checkpoint freeze')
   if alias.get('returncode',0)!=0: raise ProducerError('completed phase receipt has nonzero return code')
   child=dispatch.get('child_phase_receipt')
   alias_ref=next(x['artifacts']['phase_receipt'] for x in manifest['run_artifacts'] if x['side']==side and int(x['seed_id'])==seed)
   original_ref=next(x['artifacts']['phase_receipt_original'] for x in manifest['run_artifacts'] if x['side']==side and int(x['seed_id'])==seed)
   if child.get('path')!=original_ref['path'] or child.get('sha256')!=original_ref['sha256'] or dispatch.get('phase_receipt_alias_path')!=alias_ref['path'] or dispatch.get('phase_receipt_alias_sha256')!=alias_ref['sha256'] or dispatch.get('phase_receipt_original_path')!=original_ref['path'] or dispatch.get('phase_receipt_original_sha256')!=original_ref['sha256']:
    raise ProducerError('dispatch child receipt path/hash differs from bound original receipt')
   if not isinstance(child,dict) or child.get('status')!='completed' or child.get('job_id')!=alias.get('job_id') or child.get('phase')!=alias.get('phase') or child.get('output_dir')!=alias.get('output_dir'):
    raise ProducerError('dispatch child phase receipt does not match completed alias')
   proof_rows.append({'side':side,'seed_id':seed,'checkpoint_sha256':cp_sha,'parameters_sha256':before_hash,
                      'phase_job_id':alias['job_id'],'driver_report_sha256':sha256(report_path)})
 # Parameter identity is per method/seed; don't incorrectly require two trained models to share weights.
 if plan['outcome_type']=='paired_window_cost_superiority':
  expected_device=plan.get('profile_device')
  if expected_device not in {'cpu','cuda'}: raise ProducerError('H3 plan does not freeze an exact CPU/CUDA device')
  canonical=lambda x: 'cpu' if x=='cpu' else ('cuda' if x in {'cuda','cuda:0'} else None)
  for seed in range(5):
   l=artifact_index[('left',seed)];r=artifact_index[('right',seed)]
   lf=_strict_json(l['freeze'][0]);rf=_strict_json(r['freeze'][0])
   if (l['checkpoint'][1]!=r['checkpoint'][1] or lf.get('checkpoint_sha256')!=rf.get('checkpoint_sha256')
       or lf.get('parameters_sha256')!=rf.get('parameters_sha256')):
    raise ProducerError('H3 profiling policies do not share the exact paired frozen checkpoint')
   lr=_strict_json(l['driver_report'][0]);rr=_strict_json(r['driver_report'][0])
   if canonical(lr.get('device'))!=canonical(rr.get('device')) or canonical(lr.get('device'))!=expected_device:
    raise ProducerError('H3 profile reports differ from the exact predeclared device')
 if _is_h2_matched(plan):
  arm_map={'V4-teacher-informative':'informative','V4-teacher-shuffled-c':'shuffled',
           'V4-teacher-constant-g':'constant','V4-teacher-unconditional-neighbor-reset':'reset'}
  for seed in range(5):
   for side in ('left','right'):
    arts=artifact_index[(side,seed)];report_path=_resolve_artifact(root,arts['driver_report']);report=_strict_json(report_path)
    proof_ref=arts.get('matched_graph_proof');proof_path=(report_path.parent/report.get('graph_proof_path','')).resolve()
    if (proof_ref is None or proof_path!=Path(proof_ref[0]).resolve() or not proof_path.is_file()
        or sha256(proof_path)!=proof_ref[1] or report.get('graph_proof_sha256')!=proof_ref[1]
        or report.get('scope')!='frozen_h2_matched_retention_diagnostic' or report.get('split')!='test'
        or report.get('oracle_accessed') is not False or report.get('targets_passed_to_kernel') is not False
        or report.get('graph_proof_lines')!=expected_count*int(plan['history'])*4):
     raise ProducerError('H2 full-window teacher/action proof is not bound before prediction arrays are opened')
    arm=arm_map[plan[side]['method_id']];rec=report.get('prediction_outputs',{}).get(arm);bound=arts.get('archive')
    if not isinstance(rec,dict) or bound is None or rec.get('path')!=Path(bound[0]).name or rec.get('sha256')!=bound[1]:
     raise ProducerError('H2 proof does not bind the exact declared prediction arm archive')
    if not _verify_h2_proof(proof_path,plan,seed,pairs): raise ProducerError('H2 proof stream is incomplete')
   left=artifact_index[('left',seed)];right=artifact_index[('right',seed)]
   if (left['checkpoint'][1]!=right['checkpoint'][1]
       or _strict_json(left['freeze'][0]).get('parameters_sha256')!=_strict_json(right['freeze'][0]).get('parameters_sha256')):
    raise ProducerError('H2 matched arms do not share the same frozen model weights')
 return {'full_pair_count':expected_count,'run_proofs':proof_rows,'pretest_binding_sha256':sha256(binding_path)}

def _is_h2_matched(plan):
 return (plan.get('claim')=='H2' and plan.get('outcome_type')=='forecast_mae_superiority'
         and isinstance(plan.get('left'),dict) and isinstance(plan.get('right'),dict)
         and plan['left'].get('method_id')=='V4-teacher-informative'
         and plan['right'].get('method_id') in {'V4-teacher-shuffled-c','V4-teacher-constant-g','V4-teacher-unconditional-neighbor-reset'})

def _selected_driver(plan):
 if plan.get('schema_version')==4 and plan.get('outcome_type') in {'graph_tv_noise_superiority','graph_tv_response_superiority'}:
  return 'edsgn/staging/frozen_graph_state_diagnostic_v8.py'
 if _is_h2_matched(plan): return 'edsgn/staging/frozen_h2_matched_retention_v4.py'
 return {'paired_window_cost_superiority':'edsgn/staging/frozen_profile_guarded_v5.py',
  'graph_tv_noise_superiority':'edsgn/staging/frozen_graph_state_diagnostic_v7.py',
  'graph_tv_response_superiority':'edsgn/staging/frozen_graph_state_diagnostic_v7.py',
  'forecast_mae_superiority':'edsgn/staging/frozen_prediction_guarded_v5.py',
  'forecast_mae_noninferiority':'edsgn/staging/frozen_prediction_guarded_v5.py'}[plan['outcome_type']]

def _resolve_artifact(root,ref):
 p=ref[0].resolve()
 if not p.is_relative_to(root.resolve()) or not p.is_file() or sha256(p)!=ref[1]: raise ProducerError(f'validated artifact changed or escaped RUN: {p}')
 return p

def _archive_ref(root,plan,rows,side,seed):
 arts=rows[(side,seed)]
 if _is_h2_matched(plan):
  report_path=_resolve_artifact(root,arts['driver_report']); report=_strict_json(report_path)
  outputs=report.get('prediction_outputs')
  arm={'V4-teacher-informative':'informative','V4-teacher-shuffled-c':'shuffled',
       'V4-teacher-constant-g':'constant','V4-teacher-unconditional-neighbor-reset':'reset'}.get(plan.get(side,{}).get('method_id'))
  if arm is None: raise ProducerError('H2 matched endpoint side is not a declared V4 teacher arm')
  record=outputs.get(arm) if isinstance(outputs,dict) else None
  if not isinstance(record,dict) or set(record)<{'path','sha256'}: raise ProducerError('H2 driver report omits exact requested prediction_outputs arm')
  archive_path=(report_path.parent/record['path']).resolve()
  bound=arts.get('archive')
  if bound is None or archive_path!=Path(bound[0]).resolve() or record['sha256']!=bound[1] or sha256(archive_path)!=bound[1]:
   raise ProducerError('H2 report arm archive path/hash differs from the exact planned artifact')
  proof_path=(report_path.parent/report.get('graph_proof_path','')).resolve()
  proof_ref=arts.get('matched_graph_proof')
  if (proof_ref is None or proof_path!=Path(proof_ref[0]).resolve() or
      not proof_path.is_file() or sha256(proof_path)!=report.get('graph_proof_sha256') or
      report.get('oracle_accessed') is not False or report.get('targets_passed_to_kernel') is not False):
   raise ProducerError('H2 graph proof path/hash is missing or changed')
  expected_lines=int(report.get('origin_count',0))*int(plan['history'])*4
  if report.get('graph_proof_lines')!=expected_lines: raise ProducerError('H2 graph proof lacks all origins/history/arms')
  count=0
  with proof_path.open('r',encoding='utf-8') as stream:
   for line in stream:
    obj=json.loads(line); count+=1
    if obj.get('graph_match_proved') is not True or obj.get('teacher_graph_action_sha256')!=obj.get('arm_graph_action_sha256') or obj.get('targets_passed_to_kernel') is True:
     raise ProducerError('H2 graph/action parity proof row failed or contains targets')
  if count!=expected_lines: raise ProducerError('H2 graph proof row count is incomplete')
  return (archive_path,bound[1])
 return arts['archive']

def _load_predictions(root,plan,artifact_index,pairs):
 left=[];right=[];targets=None;masks=None;horizons=None
 for side,slot in (('left',left),('right',right)):
  for seed in range(5):
   path=_resolve_artifact(root,_archive_ref(root,plan,artifact_index,side,seed))
   with np.load(path,allow_pickle=False) as z:
    need={'predictions','targets','target_mask','origins','seed_id'}
    if not need.issubset(z.files): raise ProducerError(f'prediction archive missing keys: {path}')
    pred=np.asarray(z['predictions']);target=np.asarray(z['targets']);mask=np.asarray(z['target_mask']);origin=np.asarray(z['origins']);sid=np.asarray(z['seed_id'])
    if sid.shape!=() or sid.dtype.kind not in 'iu' or int(sid)!=seed: raise ProducerError('archive model seed differs from bound run')
    if origin.dtype.kind not in 'iu' or origin.shape!=pairs.shape or not np.array_equal(origin,pairs): raise ProducerError('prediction archive lacks exact complete ordered test origins')
    if pred.ndim!=4 or target.shape!=pred.shape or mask.shape!=pred.shape or mask.dtype.kind!='b': raise ProducerError('prediction/target/mask axes or types mismatch')
    if not np.isfinite(pred).all() or not np.isfinite(target).all(): raise ProducerError('nonfinite prediction or target')
    if pred.shape[0]!=len(pairs) or pred.shape[1]<max(plan['required_horizons']): raise ProducerError('prediction full-origin/horizon support mismatch')
    if targets is None:
     targets=target.copy();masks=mask.copy();horizons=np.arange(1,pred.shape[1]+1,dtype=np.int64)
    elif not np.array_equal(targets,target) or not np.array_equal(masks,mask): raise ProducerError('paired runs do not contain identical targets/masks')
    slot.append(pred.copy())
 return np.stack(left),np.stack(right),targets,masks,horizons

def _parse_int(text,label):
 if not isinstance(text,str) or not re.fullmatch(r'(0|[1-9][0-9]*)',text): raise ProducerError(f'{label} must be a canonical nonnegative integer')
 return int(text)
def _parse_float(text,label):
 if not isinstance(text,str) or not re.fullmatch(r'-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',text): raise ProducerError(f'{label} must be an explicit finite decimal')
 v=float(text)
 if not math.isfinite(v): raise ProducerError(f'{label} must be finite')
 return v

def _csv_rows(path, seed_id):
 out=[]
 with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f)
  if not reader.fieldnames: raise ProducerError('timing CSV has no header')
  for raw in reader:
   row={k:raw[k] for k in raw}
   for k in ('repeat','dataset_index','origin'): row[k]=_parse_int(row[k],k)
   row['seed_id']=int(seed_id)
   row['end_to_end_ms_excluding_preprocessing']=_parse_float(row['end_to_end_ms_excluding_preprocessing'],'end_to_end_ms_excluding_preprocessing')
   out.append(row)
 return out

def _switches(plan,root=ROOT,expected_pairs=None):
 if plan.get('switch_indices') is not None: raise ProducerError('pretest plan must not contain decoded switch values')
 authority=plan.get('reporting_switch_authority')
 if not isinstance(authority,dict):raise ProducerError('separate reporting-only switch authority is absent')
 authority_path=(root/authority['path']).resolve()
 if not authority_path.is_file() or sha256(authority_path)!=authority['sha256']:
  raise ProducerError('report-only switch authority changed')
 authority_doc=_strict_json(authority_path)
 if authority_doc.get('schema')!='edsgn.reporting-switch-authority.v1' or authority_doc.get('scope')!='reporting_only_postrun':
  raise ProducerError('report-only switch authority schema/scope mismatch')
 observed={}
 for entry in authority_doc['entries']:
  source=(root/entry['path']).resolve()
  if not source.is_file() or sha256(source)!=entry['sha256']:raise ProducerError('bound switch source bytes changed')
  # Read only the scalar switch_index member; transition matrices and all other
  # oracle members remain unopened. This happens only after all ten receipts pass.
  with np.load(source,allow_pickle=False) as archive:
   if entry['field'] not in archive.files:raise ProducerError('switch source lacks frozen scalar field')
   value=np.asarray(archive[entry['field']])
   if value.shape!=() or value.dtype.kind not in 'iu':raise ProducerError('switch field must be scalar integer')
   observed[str(entry['trajectory_index'])]=int(value)
 if expected_pairs is not None:
  tids=sorted(set(int(x) for x in np.asarray(expected_pairs)[:,0]))
  if set(observed)!={str(i) for i in tids}:raise ProducerError('postrun switch authority does not map every expected trajectory')
 parsed={int(k):v for k,v in observed.items()}
 canonical=json.dumps({str(k):v for k,v in sorted(parsed.items())},sort_keys=True,separators=(',',':')).encode()
 return parsed,{'scope':'prepared_frozen_switch_mapping','source_sha256':authority['sha256'],
  'authority_scope':'separate_report_only_postrun_oracle_join','authority_sha256':authority['sha256'],
  'source_file_sha256_by_trajectory':{str(x['trajectory_index']):x['sha256'] for x in authority_doc['entries']},
  'mapping_sha256':hashlib.sha256(canonical).hexdigest()}

def _switch_support_reason(plan,pairs,switches):
    if plan['slice']=='full_split': return None
    rows=plan['domain']['bounds'];hist=plan['history']
    for row in rows:
        tid=row['trajectory_index'];s=switches[tid];first=row['first_origin'];last=row['last_origin']
        if plan['slice']=='full_post_switch' and not first<=s<=last:
            return f'trajectory {tid} has no full post-switch origin support'
        if plan['slice']=='post4L' and not (first<=s and s+4*hist-1<=last):
            return f'trajectory {tid} lacks the complete post-4L support'
        if plan['slice']=='paired_pre_post4L' and not (first<=s-4*hist and s+4*hist-1<=last):
            return f'trajectory {tid} lacks complete paired pre/post-4L support'
    return None

def _trace_stream(path):
 opener=gzip.open if path.suffix=='.gz' else open
 def iterator():
  with opener(path,'rt',encoding='utf-8') as f:
   for line in f:
    value=json.loads(line)
    if not isinstance(value,dict): raise ProducerError('trace row must be JSON object')
    yield value
 return iterator()

def _graph_stats(root,plan,artifact_index,pairs,converter):
 if plan['outcome_type']=='graph_tv_response_superiority':
  mode='response'
 else:
  mode={'full_split':'all','full_post_switch':'full_post_switch','post4L':'post'}.get(plan['slice'])
  if mode is None: raise ProducerError('graph endpoint has unsupported slice')
 switches,swproof=_switches(plan,root,pairs) if plan['slice'] in {'post4L','paired_pre_post4L','full_post_switch'} else (None,None)
 if switches is not None:
  reason=_switch_support_reason(plan,pairs,switches)
  if reason:return {'_unidentifiable':reason,'switch_indices':switches,'switch_mapping_proof':swproof}
 outputs={}
 sides=('left',) if mode=='response' else ('left','right')
 for side in sides:
  rows_by_seed={};nodes=None
  for seed in range(5):
   report=_strict_json(_resolve_artifact(root,artifact_index[(side,seed)]['driver_report']))
   trace_path=_resolve_artifact(root,artifact_index[(side,seed)]['trace'])
   declared=report.get('trace_rows_file')
   if declared is not None and Path(declared).name!=trace_path.name: raise ProducerError('graph report trace filename differs from manifest')
   # Node order comes from the complete first graph row and must match the P3A-bound domain node digest.
   iterator=_trace_stream(trace_path)
   try:first=next(iterator)
   except StopIteration as exc: raise ProducerError('empty trace stream') from exc
   grows=first.get('graph_rows')
   if not isinstance(grows,list) or not grows: raise ProducerError('trace lacks full graph rows')
   ids=[r.get('node_id') for r in grows]
   if any(type(x)is not int for x in ids) or len(set(ids))!=len(ids): raise ProducerError('trace node IDs invalid')
   candidate=np.asarray(ids,dtype=np.int64)
   ddoc=_strict_json(root/plan['domain']['path'])
   want=ddoc['dataset'].get('node_ids_sha256')
   if hashlib.sha256(np.ascontiguousarray(candidate).tobytes()).hexdigest()!=want: raise ProducerError('trace node order does not match frozen domain hash')
   if nodes is not None and not np.array_equal(nodes,candidate): raise ProducerError('node order differs between seeds')
   nodes=candidate
   def full_stream(first=first,rest=iterator):
    yield first
    yield from rest
   rows_by_seed[seed]=full_stream()
  outputs[side]=converter.graph_tv_sufficient_statistics_v7(rows_by_seed,expected_full_pairs=pairs,seed_ids=np.arange(5,dtype=np.int64),history_length=plan['history'],node_ids=nodes,slice_kind=mode,switch_indices=switches,switch_mapping_proof=swproof)
 outputs['_switch_indices']=switches
 outputs['_switch_mapping_proof']=swproof
 return outputs

def _analysis_fields(analysis, *, noninferiority=False, epsilon=None):
 pkey='centered_lower_tail_noninferiority_p' if noninferiority else ('centered_lower_tail_margin_p' if analysis.get('scope')=='paired_scalar_temporal_bootstrap_kernel' and epsilon is not None else 'centered_two_sided_p')
 raw=analysis.get(pkey)
 interval=analysis.get('percentile_95_interval')
 upper=analysis.get('upper_95')
 estimate=analysis.get('estimate')
 if noninferiority:
  directional=bool(upper is not None and epsilon is not None and upper < epsilon)
 else:
  directional=bool(estimate is not None and estimate < 0 and isinstance(interval,list) and len(interval)==2 and interval[1] < 0)
 return {'raw_p':raw,'raw_p_kind':pkey,'effect_estimate':estimate,'percentile_95_interval':interval,
         'upper_95':upper,'directional_gate_passed':directional}

def _compute(plan,manifest,artifact_index,root,validator,converter,forecast_kernel,scalar_kernel,proof):
 pairs=_expected_pairs(plan);block_rows=[];eps_status=None;eps_values=[]
 binding=_strict_json(_bound_ref(root,manifest,proof))
 eps_statuses=[]
 for r in binding['runs']:
  block_rows.append(r['block_length'])
  if plan['outcome_type']=='forecast_mae_noninferiority':
   eps_statuses.append(r['epsilon_status']);eps_values.append(r['epsilon_value_raw'])
 eps_status=eps_statuses[-1] if eps_statuses else None
 if eps_statuses and len(set(eps_statuses))!=1: raise ProducerError('NI epsilon identifiability status differs across bound runs')
 if len(set(block_rows))!=1: raise ProducerError('methods/seeds have different temporal block lengths')
 block=block_rows[0]
 out={'endpoint_id':plan['endpoint_id'],'claim':plan['claim'],'outcome_type':plan['outcome_type'],'split':'test',
      'n_full_origins':len(pairs),'block_length_windows':block,'seed_ids':[0,1,2,3,4],
      'interpretation':'kernel-derived endpoint only; no claim/family decision'}
 if plan['outcome_type'] in {'forecast_mae_superiority','forecast_mae_noninferiority'}:
  if plan['outcome_type']=='forecast_mae_noninferiority' and (eps_status!='accepted' or any(x is None for x in eps_values)):
   if eps_status!='unidentifiable' or any(x is not None for x in eps_values): raise ProducerError('NI epsilon status is inconsistent across bound seed runs')
   return {**out,'identifiable':False,'raw_p':None,'reason':'P3A/pretest-bound validation epsilon is unidentifiable; no primary NI p-value emitted'}
  if plan['outcome_type']=='forecast_mae_noninferiority' and len(set(eps_values))!=1: raise ProducerError('bound NI epsilon differs across seeds/methods')
  mode={'full_split':'all','post4L':'post','full_post_switch':'full_post_switch'}.get(plan['slice'])
  if mode is None: raise ProducerError('forecast producer does not accept paired response slice')
  sw,swproof=_switches(plan,root,pairs) if mode!='all' else (None,None)
  if sw is not None:
   reason=_switch_support_reason(plan,pairs,sw)
   if reason:return {**out,'identifiable':False,'raw_p':None,'reason':reason,'switch_indices':sw,'switch_authority_sha256':swproof['authority_sha256']}
  pred_l,pred_r,target,mask,hids=_load_predictions(root,plan,artifact_index,pairs)
  kwargs=dict(slice_kind=mode,history_length=plan['history'],switch_indices=sw,switch_mapping_proof=swproof)
  try:
   lstat=converter.forecast_sufficient_statistics(pred_l,target,mask,pairs,pairs,np.arange(5,dtype=np.int64),hids,np.asarray(plan['required_horizons'],dtype=np.int64),**kwargs)
   rstat=converter.forecast_sufficient_statistics(pred_r,target,mask,pairs,pairs,np.arange(5,dtype=np.int64),hids,np.asarray(plan['required_horizons'],dtype=np.int64),**kwargs)
  except ValueError as exc:
   if 'unidentifiable zero-support forecast endpoint' in str(exc):
    return {**out,'identifiable':False,'raw_p':None,'reason':str(exc)}
   raise
  epsilon=float(eps_values[0]) if plan['outcome_type']=='forecast_mae_noninferiority' else 0.
  try:
   analysis=forecast_kernel.paired_forecast_mae_bootstrap(lstat['left_abs_error_sum'],lstat['valid_count'],rstat['left_abs_error_sum'],rstat['valid_count'],lstat['pairs'],block_length=block,margin=epsilon)
  except ValueError as exc:
   if 'block length exceeds a trajectory' in str(exc): return {**out,'identifiable':False,'raw_p':None,'reason':str(exc)}
   raise
  return {**out,'identifiable':True,'switch_indices':sw,'switch_authority_sha256':swproof['authority_sha256'] if swproof else None,'epsilon_raw':epsilon if plan['outcome_type']=='forecast_mae_noninferiority' else None,**_analysis_fields(analysis,noninferiority=plan['outcome_type']=='forecast_mae_noninferiority',epsilon=epsilon),'analysis':analysis}
 if plan['outcome_type'] in {'graph_tv_noise_superiority','graph_tv_response_superiority'}:
  stats=_graph_stats(root,plan,artifact_index,pairs,converter)
  if '_unidentifiable' in stats:
   return {**out,'identifiable':False,'raw_p':None,'reason':stats['_unidentifiable'],'switch_indices':stats['switch_indices'],'switch_authority_sha256':stats['switch_mapping_proof']['authority_sha256']}
  try:
   if plan['outcome_type']=='graph_tv_response_superiority':
    s=stats['left'];pairs_rel=np.column_stack((s['trajectory_ids'],s['relative_origin_offsets'])).astype(np.int64)
    analysis=scalar_kernel.paired_scalar_bootstrap(s['negative_post_sum'],s['valid_row_count'],s['negative_pre_sum'],s['valid_row_count'],pairs_rel,block_length=block,margin=0.)
   else:
    l=stats['left'];r=stats['right']
    analysis=scalar_kernel.paired_scalar_bootstrap(l['row_tv_sum'],l['valid_row_count'],r['row_tv_sum'],r['valid_row_count'],l['pairs'],block_length=block,margin=0.)
  except ValueError as exc:
   if 'block length exceeds a trajectory' in str(exc): return {**out,'identifiable':False,'raw_p':None,'reason':str(exc)}
   raise
  return {**out,'identifiable':True,'switch_indices':stats['_switch_indices'],'switch_authority_sha256':stats['_switch_mapping_proof']['authority_sha256'] if stats['_switch_mapping_proof'] else None,**_analysis_fields(analysis),'analysis':analysis}
 if plan['outcome_type']=='paired_window_cost_superiority':
  rows_by_side={'left':[],'right':[]};summaries_by_side={'left':[],'right':[]}
  for side in ('left','right'):
   for seed in range(5):
    csv_path=_resolve_artifact(root,artifact_index[(side,seed)]['csv']);report=_strict_json(_resolve_artifact(root,artifact_index[(side,seed)]['driver_report']))
    report=dict(report);report['seed_id']=seed
    summaries_by_side[side].append(report);rows_by_side[side].extend(_csv_rows(csv_path,seed))
  l=converter.timing_sufficient_statistics(rows_by_side['left'],expected_full_pairs=pairs,seed_ids=np.arange(5,dtype=np.int64),runtime_summaries=summaries_by_side['left'],expected_device=plan['profile_device'])
  r=converter.timing_sufficient_statistics(rows_by_side['right'],expected_full_pairs=pairs,seed_ids=np.arange(5,dtype=np.int64),runtime_summaries=summaries_by_side['right'],expected_device=plan['profile_device'])
  analysis=scalar_kernel.paired_scalar_bootstrap(l['end_to_end_ms_sum'],l['valid_count'],r['end_to_end_ms_sum'],r['valid_count'],pairs,block_length=block,margin=0.)
  return {**out,'identifiable':True,**_analysis_fields(analysis),'analysis':analysis,'timing_scope':'median of five repeats per seed/origin; paired origins, no energy claim'}
 raise ProducerError(f'unsupported endpoint outcome {plan["outcome_type"]}')

def produce_endpoint(plan_path,registry_path,manifest_path,output_path,*,root=ROOT,official=True,_validator=None):
 """Validate metadata/receipts, compute one endpoint, revalidate, exclusive-write."""
 root=Path(root).resolve(); plan_path=Path(plan_path).resolve();registry_path=Path(registry_path).resolve();manifest_path=Path(manifest_path).resolve();output=Path(output_path).resolve()
 if root!=ROOT.resolve() or official is not True or _validator is not None: raise ProducerError('public producer requires canonical RUN and official metadata validator; injected roots/validators are test-only and unsupported')
 if not output.is_relative_to(root): raise ProducerError('output escapes RUN')
 if output.exists(): raise FileExistsError(output)
 validator=_validator or _load_module(VALIDATOR_PATH,'_typed_endpoint_validator_v9_for_producer')
 converter=_load_module(CONVERTER_PATH,'_endpoint_sufficient_statistics_v4_for_producer')
 forecast=_load_module(FORECAST_KERNEL_PATH,'_paired_forecast_bootstrap_v1_for_producer')
 scalar=_load_module(SCALAR_KERNEL_PATH,'_paired_scalar_bootstrap_v1_for_producer')
 plan=_strict_json(plan_path);manifest=_strict_json(manifest_path)
 # Metadata validator must complete first. It performs canonical gate/P3A checks before any output arrays load.
 proof=validator.validate(plan_path,registry_path,manifest_path=manifest_path,root=root,official=official)
 declared={row['path']:row['sha256'] for row in plan.get('producer_sources',[])}
 selected_driver=_selected_driver(plan)
 required_helpers=['edsgn/staging/typed_endpoint_producer_v6.py',selected_driver,
  'edsgn/staging/typed_endpoint_plan_validator_v9.py','edsgn/scripts/verified_frozen_dispatch_v10.py',
  'edsgn/scripts/bind_training_inputs.py','edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py',
  'edsgn/staging/endpoint_sufficient_statistics_v4.py','edsgn/staging/paired_forecast_bootstrap_v1.py',
  'edsgn/staging/paired_scalar_bootstrap_v1.py']
 if plan.get('schema_version')==4:
  required_helpers.append('edsgn/staging/prospective_training_candidate_binding_v1.py')
  required_helpers.append('edsgn/staging/global_frozen_test_completion_v1.py')
 for rel in required_helpers:
  path=(root/rel).resolve()
  if declared.get(rel)!=sha256(path): raise ProducerError(f'P3A-plan does not bind current required producer source: {rel}')
 plan_sha=sha256(plan_path);manifest_sha=sha256(manifest_path);registry_sha=sha256(registry_path)
 source_before={str(p.resolve()):sha256(p) for p in SOURCE_PATHS if p.is_file()}
 if str(Path(__file__).resolve()) not in source_before: source_before[str(Path(__file__).resolve())]=sha256(Path(__file__).resolve())
 artifact_index=_run_rows(root,manifest)
 semantic=_semantic_receipts(root,plan,manifest,artifact_index,validator,proof)
 # Semantic validation completes before numerical prediction/trace/timing artifacts are opened.
 result=_compute(plan,manifest,artifact_index,root,validator,converter,forecast,scalar,proof)
 result.update({'schema_version':1,'scope':'typed endpoint sufficient-statistic/bootstrap producer; no family claim decision',
   'plan_path':plan_path.relative_to(root).as_posix(),'plan_sha256':plan_sha,'registry_path':registry_path.relative_to(root).as_posix(),
   'registry_sha256':registry_sha,'postrun_manifest_path':manifest_path.relative_to(root).as_posix(),'postrun_manifest_sha256':manifest_sha,
   'canonical_pipeline_sha256':proof['canonical_pipeline_sha256_final'],'semantic_run_proofs':semantic,
   'producer_source_sha256':source_before,'consumed_hashes':proof['postrun_manifest']['consumed_hashes']})
 # Revalidate metadata authority, all consumed bytes, helpers, and canonical gate after array work.
 proof_after=validator.validate(plan_path,registry_path,manifest_path=manifest_path,root=root,official=official)
 if proof_after['canonical_pipeline_sha256_final']!=proof['canonical_pipeline_sha256_final'] or proof_after['postrun_manifest']['consumed_hashes']!=proof['postrun_manifest']['consumed_hashes']:
  raise ProducerError('canonical gate or consumed file set changed during endpoint production')
 if sha256(plan_path)!=plan_sha or sha256(registry_path)!=registry_sha or sha256(manifest_path)!=manifest_sha: raise ProducerError('plan/registry/postrun manifest changed during endpoint production')
 if {str(Path(k)):sha256(Path(k)) for k in source_before}!=source_before: raise ProducerError('producer/helper sources changed during endpoint production')
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x',encoding='utf-8',newline='\n') as f:
  json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 output_sha=sha256(output)
 # Final fence after serialization; failure leaves the unique output as an auditable failed artifact.
 proof_final=validator.validate(plan_path,registry_path,manifest_path=manifest_path,root=root,official=official)
 if proof_final['canonical_pipeline_sha256_final']!=proof['canonical_pipeline_sha256_final'] or sha256(plan_path)!=plan_sha or sha256(registry_path)!=registry_sha or sha256(manifest_path)!=manifest_sha:
  raise ProducerError('final plan/manifest/gate fence failed after output serialization')
 if {str(Path(k)):sha256(Path(k)) for k in source_before}!=source_before: raise ProducerError('source changed after endpoint output serialization')
 if sha256(output)!=output_sha: raise ProducerError('endpoint output changed after exclusive serialization')
 for rel,digest in proof['postrun_manifest']['consumed_hashes'].items():
  p=(root/rel).resolve()
  if not p.is_file() or sha256(p)!=digest: raise ProducerError(f'consumed input changed after serialization: {rel}')
 return result
