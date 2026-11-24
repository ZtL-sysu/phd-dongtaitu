"""Frozen no-refit evaluator for simple baselines (staging only).

This module deliberately has no fit routine. Production test execution remains
blocked until the root-owned P3/P3A authorization and a campaign dispatcher bind
this interface. The evaluator itself consumes only train-fitted coefficients.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Callable
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "edsgn" / "src"
VALIDATOR = ROOT / "edsgn" / "scripts" / "frozen-pipeline-validator" / "scripts" / "pipeline_state.py"
GUARD_SOURCE = ROOT / "edsgn" / "staging" / "simple_baseline_freeze_v1.py"
STATE = ROOT / "pipeline-state.json"

class FrozenSimpleError(RuntimeError): pass

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def canonical_hash(value: Any) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()

def _json(path: Path) -> dict:
    doc=json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(doc,dict): raise FrozenSimpleError(f'expected JSON object: {path}')
    return doc

def _npy_header(path: Path) -> dict:
    # Read only the NPY header. Do not deserialize coefficient or dataset arrays
    # for metadata-only inventory/admission.
    with path.open('rb') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        else: shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
    return {'shape':list(shape),'dtype':str(dtype),'fortran_order':bool(fortran)}

def inspect_existing_p3_fit(output_dir: str | Path) -> dict:
    """Metadata-only inventory of the legacy P3 simple output candidate."""
    out=Path(output_dir).resolve(); manifest_path=out/'simple-baseline-manifest.json'
    coef_path=out/'linear_coefficients.npy'
    if not manifest_path.is_file() or not coef_path.is_file():
        raise FileNotFoundError(f'P3 manifest/coefficient missing under {out}')
    manifest=_json(manifest_path)
    return {'manifest_path':str(manifest_path),'manifest_sha256':sha256_file(manifest_path),
        'coefficient_path':str(coef_path),'coefficient_sha256':sha256_file(coef_path),
        'coefficient_header':_npy_header(coef_path),'train_node_window_samples':manifest.get('train_node_window_samples'),
        'dataset_path':manifest.get('dataset_path'),'split':manifest.get('split'),
        'test_accessed':manifest.get('test_accessed'),'source_sha256':manifest.get('source_sha256'),
        'admission_ready':False,'missing_fit_bindings':['coefficient_sha256 in P3 fit manifest',
          'training_statistics_sha256','node_ids_sha256','completed P3 root acceptance']}

def _stage_state(root: Path) -> dict:
    state=_json(root/'pipeline-state.json')
    stages=state.get('stages')
    if not isinstance(stages,list) or len({x.get('id') for x in stages if isinstance(x,dict)})!=len(stages):
        raise FrozenSimpleError('malformed/duplicate pipeline stages')
    byid={x.get('id'):x for x in stages}
    for name in ('P2','P3','P3A'):
        row=byid.get(name)
        if not isinstance(row,dict) or row.get('status')!='passed' or row.get('gate_verdict')!='pass':
            raise FrozenSimpleError(f'requires {name}=passed/pass')
    if byid.get('P4',{}).get('status')!='running': raise FrozenSimpleError('requires P4=running')
    return state

def _passed_refs(root: Path, state: dict) -> dict[str,str]:
    refs={}
    for stage in state['stages']:
        if stage.get('status')=='passed':
            for entry in [*stage.get('inputs',[]),*stage.get('evidence',[])]:
                p=(root/entry['path']).resolve()
                if not p.is_file() or sha256_file(p)!=entry.get('sha256'):
                    raise FrozenSimpleError(f'stale passed-stage reference: {p}')
                refs[str(p)]=entry['sha256']
    refs[str((root/'pipeline-state.json').resolve())]=sha256_file(root/'pipeline-state.json')
    return refs


def _current_train_fingerprint(dataset_path: Path) -> dict[str,str]:
    root=dataset_path.resolve()
    if root.is_file():
        files=[root]
        files.extend(x.resolve() for x in (root.parent/'manifest.json',root.parent/'data-manifest.json') if x.is_file())
    elif root.is_dir():
        files=[x.resolve() for x in root.glob('train_seed*.npz') if x.is_file()]
        m=root/'manifest.json'
        if m.is_file(): files.append(m.resolve())
    else: raise FileNotFoundError(root)
    if not files: raise FrozenSimpleError('no train-source files found')
    return {p.name:sha256_file(p) for p in sorted(set(files))}

def _collection_snapshot(dataset_path: Path) -> dict[str,str]:
    root=dataset_path.resolve()
    if root.is_file(): files=[root,*(x for x in (root.parent/'data-manifest.json',root.parent/'manifest.json') if x.is_file())]
    elif root.is_dir(): files=[p.resolve() for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.npz','.json'}]
    else: raise FileNotFoundError(root)
    return {str(p):sha256_file(p) for p in sorted(set(files))}

def _input_snapshot(root: Path, dataset_path: Path, fit_path: Path, coeff_path: Path,
                    authorization_path: Path) -> dict[str,str]:
    state=_stage_state(root)
    refs=_passed_refs(root,state)
    for p in (fit_path,coeff_path,authorization_path,Path(__file__).resolve(),VALIDATOR,GUARD_SOURCE,SRC/'edsgn'/'engine.py',SRC/'edsgn'/'data.py'):
        if not p.is_file(): raise FrozenSimpleError(f'bound input missing: {p}')
        refs[str(p.resolve())]=sha256_file(p)
    refs.update(_collection_snapshot(dataset_path))
    auth=_json(authorization_path)
    dev_path=(root/auth['development_acceptance_path']).resolve()
    try: dev_path.relative_to(root)
    except ValueError as exc: raise FrozenSimpleError('development acceptance escapes RUN') from exc
    refs[str(dev_path)]=sha256_file(dev_path)
    for item in auth['accepted_validation_metrics']:
        metric=(root/item['path']).resolve()
        try: metric.relative_to(root)
        except ValueError as exc: raise FrozenSimpleError('validation metric escapes RUN') from exc
        refs[str(metric)]=sha256_file(metric)
    raw=auth.get('domain_descriptor_path')
    if not isinstance(raw,str) or not raw: raise FrozenSimpleError('authorization must bind a domain descriptor path')
    domain=(root/raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    try: domain.relative_to(root)
    except ValueError as exc: raise FrozenSimpleError('domain descriptor escapes RUN') from exc
    if not domain.is_file() or sha256_file(domain)!=auth.get('domain_descriptor_sha256'):
        raise FrozenSimpleError('domain descriptor path/hash mismatch')
    refs[str(domain)]=sha256_file(domain)
    return refs

def _official_before_p4(root: Path) -> None:
    result=subprocess.run([sys.executable,str(VALIDATOR),str(root/'pipeline-state.json'),'--before','P4'],
                          cwd=str(root),capture_output=True,text=True,check=False)
    if result.returncode!=0: raise FrozenSimpleError('official before-P4 validator rejected state: '+result.stdout[-1500:]+result.stderr[-1500:])

def validate_fit_authorization(root: Path, dataset_path: Path, fit_path: Path,
                               authorization_path: Path, method: str, output_dir: Path) -> tuple[dict,dict,Path]:
    """Metadata gate. Must run before any test dataset loader/NPZ decoder."""
    if method not in {'persistence','linear'}: raise ValueError(method)
    state=_stage_state(root)
    p3=next(x for x in state['stages'] if x['id']=='P3')
    p3a=next(x for x in state['stages'] if x['id']=='P3A')
    def stage_refs(stage):
        result={}
        for entry in [*stage.get('inputs',[]),*stage.get('evidence',[])]:
            if isinstance(entry,dict) and isinstance(entry.get('path'),str): result[str((root/entry['path']).resolve())]=entry.get('sha256')
        return result
    manifest=_json(fit_path)
    if manifest.get('artifact_role')!='train_only_simple_baseline_fit' or manifest.get('split')!='train' or manifest.get('test_accessed') is not False:
        raise FrozenSimpleError('fit manifest is not a completed train-only artifact')
    auth=_json(authorization_path)
    if stage_refs(p3a).get(str(authorization_path.resolve()))!=sha256_file(authorization_path):
        raise FrozenSimpleError('P3A does not hash-bind this exact pretest authorization')
    dev_path=(root/auth.get('development_acceptance_path','')).resolve()
    if not dev_path.is_file() or sha256_file(dev_path)!=auth.get('development_acceptance_sha256'):
        raise FrozenSimpleError('P3 development acceptance path/hash is not bound')
    try: dev_path.relative_to(root)
    except ValueError as exc: raise FrozenSimpleError('P3 acceptance escapes RUN') from exc
    dev=_json(dev_path)
    if dev.get('accepted_by_root') is not True or dev.get('verdict')!='pass' or dev.get('scope')!='simple-baseline-development-fit':
        raise FrozenSimpleError('missing root-accepted P3 development fit record')
    if stage_refs(p3).get(str(dev_path))!=sha256_file(dev_path):
        raise FrozenSimpleError('P3 does not hash-bind its development acceptance')
    if dev.get('fit_manifest_path')!=str(fit_path.resolve()) or dev.get('fit_manifest_sha256')!=sha256_file(fit_path):
        raise FrozenSimpleError('development acceptance differs from fit manifest')
    if stage_refs(p3).get(str(fit_path.resolve()))!=sha256_file(fit_path):
        raise FrozenSimpleError('P3 does not hash-bind this fit manifest')
    accepted_metrics=auth.get('accepted_validation_metrics')
    if not isinstance(accepted_metrics,list) or not accepted_metrics: raise FrozenSimpleError('pretest authorization must bind raw P3 validation metric files')
    if dev.get('accepted_validation_metrics')!=accepted_metrics:
        raise FrozenSimpleError('pretest authorization differs from metrics accepted by P3')
    for item in accepted_metrics:
        metric=(root/item['path']).resolve()
        try: metric.relative_to(root)
        except ValueError as exc: raise FrozenSimpleError('accepted validation metric escapes RUN') from exc
        if not metric.is_file() or sha256_file(metric)!=item.get('sha256'):
            raise FrozenSimpleError(f'accepted validation metric hash mismatch: {metric}')
        if stage_refs(p3).get(str(metric))!=item['sha256']:
            raise FrozenSimpleError(f'P3 does not hash-bind accepted validation metric: {metric}')
    current_train=_current_train_fingerprint(dataset_path)
    if canonical_hash(current_train)!=manifest.get('training_source_fingerprint_sha256') or current_train!=manifest.get('training_source_fingerprint'):
        raise FrozenSimpleError('current train-source bytes differ from frozen fit manifest')
    if type(manifest.get('train_node_window_samples')) is not int or manifest['train_node_window_samples']<=0:
        raise FrozenSimpleError('fit manifest lacks a positive train-window count')
    for k in ('training_source_fingerprint','training_source_fingerprint_sha256','source_hashes','source_hashes_sha256','training_statistics','training_statistics_sha256','node_ids','node_ids_sha256'):
        if k not in manifest: raise FrozenSimpleError(f'fit manifest missing train-only binding: {k}')
    if canonical_hash(manifest['training_source_fingerprint'])!=manifest['training_source_fingerprint_sha256'] or canonical_hash(manifest['source_hashes'])!=manifest['source_hashes_sha256'] or canonical_hash(manifest['training_statistics'])!=manifest['training_statistics_sha256']:
        raise FrozenSimpleError('fit manifest contains inconsistent train-only digests')
    expected={
      'schema_version':1,'status':'accepted_pretest_authorization','method':method,
      'dataset_path':str(dataset_path.resolve()),'fit_manifest_path':str(fit_path.resolve()),
      'fit_manifest_sha256':sha256_file(fit_path),'output_dir':str(output_dir.resolve()),
      'source_collection_fingerprint_sha256':canonical_hash(_collection_snapshot(dataset_path)),
      'development_acceptance_path':auth.get('development_acceptance_path'),
      'development_acceptance_sha256':auth.get('development_acceptance_sha256'),
      'domain_descriptor_path':auth.get('domain_descriptor_path')}
    domain_path=(root/auth['domain_descriptor_path']).resolve() if not Path(auth.get('domain_descriptor_path','')).is_absolute() else Path(auth['domain_descriptor_path']).resolve()
    try: domain_path.relative_to(root)
    except ValueError as exc: raise FrozenSimpleError('domain descriptor escapes RUN') from exc
    if not domain_path.is_file() or sha256_file(domain_path)!=auth.get('domain_descriptor_sha256'):
        raise FrozenSimpleError('domain descriptor path/hash mismatch')
    current_sources={'evaluator':sha256_file(Path(__file__).resolve()),'engine':sha256_file(SRC/'edsgn'/'engine.py'),'data':sha256_file(SRC/'edsgn'/'data.py')}
    if auth.get('source_current_sha256')!=current_sources: raise FrozenSimpleError('authorization current scientific source hash mismatch')
    for k,v in expected.items():
        if auth.get(k)!=v: raise FrozenSimpleError(f'pretest authorization mismatch: {k}')
    for key in ('training_source_fingerprint_sha256','source_hashes_sha256','training_statistics_sha256','node_ids_sha256'):
        if auth.get(key)!=manifest.get(key): raise FrozenSimpleError(f'authorization does not bind fit field {key}')
    if auth.get('accepted_by_root') is not True or auth.get('verdict')!='pass':
        raise FrozenSimpleError('pretest authorization lacks root acceptance')
    if method=='linear':
        cp=Path(manifest.get('linear_coefficients_path',''))
        if not cp.is_absolute(): cp=fit_path.parent/cp
        cp=cp.resolve()
        if not cp.is_file() or sha256_file(cp)!=manifest.get('linear_coefficients_sha256'):
            raise FrozenSimpleError('fit manifest does not bind coefficient bytes')
        if stage_refs(p3).get(str(cp))!=sha256_file(cp): raise FrozenSimpleError('P3 does not hash-bind coefficient bytes')
        if auth.get('coefficient_path')!=str(cp) or auth.get('coefficient_sha256')!=sha256_file(cp):
            raise FrozenSimpleError('authorization coefficient binding mismatch')
        if not manifest.get('training_statistics_sha256') or not manifest.get('node_ids_sha256'):
            raise FrozenSimpleError('fit manifest lacks train statistics/node-order bindings')
        header=_npy_header(cp)
        task=manifest['training_statistics'].get('task',{})
        expected_shape=[int(task.get('history',0))*int(task.get('n_features',0))+1,int(task.get('hmax',0))*int(task.get('n_features',0))]
        if header['shape']!=expected_shape or header['dtype'] not in {'float64','<f8'}:
            raise FrozenSimpleError(f'coefficient header does not match train-only task: {header}')
        if auth.get('training_statistics_sha256')!=manifest['training_statistics_sha256'] or auth.get('node_ids_sha256')!=manifest['node_ids_sha256']:
            raise FrozenSimpleError('authorization train statistics/node order mismatch')
        if auth.get('training_source_fingerprint_sha256')!=manifest['training_source_fingerprint_sha256'] or auth.get('source_hashes_sha256')!=manifest['source_hashes_sha256']:
            raise FrozenSimpleError('authorization train-source/scientific-source binding mismatch')
        node_array=np.asarray(manifest['node_ids'])
        node_digest=hashlib.sha256(node_array.tobytes(order='C')).hexdigest()
        if node_digest!=manifest['node_ids_sha256'] or manifest['training_statistics'].get('node_ids_sha256')!=node_digest:
            raise FrozenSimpleError('fit manifest node list differs from train statistics digest')
        source_basenames={Path(k).name:v for k,v in manifest['source_hashes'].items()}
        for rel,current in (('engine.py',SRC/'edsgn'/'engine.py'),('data.py',SRC/'edsgn'/'data.py')):
            if source_basenames.get(rel)!=sha256_file(current): raise FrozenSimpleError(f'fitted source differs from current {rel}')
    else:
        cp=Path(manifest.get('linear_coefficients_path',''))
        if auth.get('parameter_identity')!='persistence-repeat-last-observation/no-parameters':
            raise FrozenSimpleError('persistence requires its explicit no-parameter identity')
    return manifest,auth,cp

def _target(ds,origin):
    result=ds.target(int(origin))
    if isinstance(result,dict): return np.asarray(result.get('targets',result.get('values'))),np.asarray(result.get('target_mask',result.get('mask')),dtype=bool)
    return np.asarray(result[0]),np.asarray(result[1],dtype=bool)

def predict_dataset(ds, origins, method: str, coef=None, checkpoint: Callable|None=None):
    """Exact engine evaluation semantics; no fitting or preprocessing here."""
    if method not in {'persistence','linear'}: raise ValueError(method)
    if method=='linear' and coef is None: raise FrozenSimpleError('linear evaluation requires frozen coefficients')
    pred=[]; targets=[]; masks=[]
    for i,origin in enumerate(origins):
        if checkpoint is not None and i%32==0: checkpoint()
        w=ds.window(int(origin))
        if method=='persistence': p=np.repeat(w['raw_observations'][-1:,...],ds.hmax,axis=0)
        else:
            x=np.asarray(w['values']).transpose(1,0,2).reshape(-1,ds.history*ds.values.shape[-1])
            x=np.concatenate([x,np.ones((len(x),1),dtype=x.dtype)],axis=-1)
            p=(x@coef).reshape(-1,ds.hmax,ds.values.shape[-1]).transpose(1,0,2)
            if ds.count_data: p=np.maximum(p,0.)
        y,m=_target(ds,int(origin));pred.append(p);targets.append(y);masks.append(m)
    return {'predictions':np.asarray(pred),'targets':np.asarray(targets),'target_mask':np.asarray(masks),
            'origins':np.asarray(origins,dtype=np.int64)}

def evaluate_preloaded(datasets, method: str, coef=None, checkpoint=None):
    """Full-origin evaluator for already gate-authorized PreparedData fixtures."""
    from edsgn.engine import index_origins, metric_rows
    pairs=index_origins(datasets,'test')
    if not pairs: raise FrozenSimpleError('no legal test origins')
    chunks=[]
    for ds_index,ds in enumerate(datasets):
        local=[int(o) for idx,o in pairs if idx==ds_index]
        chunks.append((ds_index,local))
    preds=[];targets=[];masks=[]
    for ds_index,origins in chunks:
        if not origins: continue
        result=predict_dataset(datasets[ds_index],origins,method,coef,checkpoint)
        preds.extend(result['predictions']);targets.extend(result['targets']);masks.extend(result['target_mask'])
    result={'predictions':np.asarray(preds),'targets':np.asarray(targets),'target_mask':np.asarray(masks),'pairs':pairs}
    rows=metric_rows(result,datasets[0].horizons,method,0,'test')
    return result,rows

def write_result(output_dir: str|Path, result: dict, rows: list[dict], *, method: str, provenance: dict) -> dict:
    out=Path(output_dir).resolve()
    if out.exists(): raise FileExistsError(f'refusing overwrite: {out}')
    out.mkdir(parents=True,exist_ok=False)
    pred_path=out/'predictions.npz'
    with pred_path.open('xb') as f: np.savez_compressed(f,predictions=result['predictions'],targets=result['targets'],
        target_mask=result['target_mask'],origins=np.asarray(result['pairs'],dtype=np.int64),seed_id=np.asarray(0,dtype=np.int64))
    csv_path=out/'metrics_by_horizon.csv'
    fields=list(rows[0]) if rows else []
    with csv_path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    outputs={p.name:sha256_file(p) for p in (pred_path,csv_path)}
    manifest={'schema_version':1,'scope':'full_protocol','split':'test','method':method,'seed_id':0,
      'n_windows':int(result['predictions'].shape[0]),'output_hashes':outputs,'provenance':provenance,
      'scored_test_origins':True,'test_origins_evaluated':True,'test_feedback_used':False,
      'combined_bundle_materialized':bool(provenance.get('combined_bundle_materialized',False)),
      'oracle_accessed':False}
    mp=out/'evaluation_manifest.json'
    with mp.open('x',encoding='utf-8',newline='\n') as f: json.dump(manifest,f,sort_keys=True,indent=2);f.write('\n')
    return {'manifest_path':str(mp),'manifest_sha256':sha256_file(mp),'outputs':outputs,'n_windows':manifest['n_windows']}

def verify_result(output_dir: str|Path, *, expected_origins: int, method: str) -> dict:
    out=Path(output_dir).resolve(); mp=out/'evaluation_manifest.json'; doc=_json(mp)
    if doc.get('scope')!='full_protocol' or doc.get('split')!='test' or doc.get('method')!=method or doc.get('n_windows')!=expected_origins:
        raise FrozenSimpleError('manifest scope/method/full-origin count mismatch')
    for name,digest in doc.get('output_hashes',{}).items():
        p=out/name
        if not p.is_file() or sha256_file(p)!=digest: raise FrozenSimpleError(f'output hash mismatch: {p}')
    with np.load(out/'predictions.npz',allow_pickle=False) as z:
        if z['origins'].shape[0]!=expected_origins or z['predictions'].shape[0]!=expected_origins: raise FrozenSimpleError('saved output is incomplete')
    return {'manifest_sha256':sha256_file(mp),'verified_outputs':doc['output_hashes'],'n_windows':expected_origins}

def _check_guard(root: Path, checkpoint: Callable) -> None:
    scripts=ROOT/'edsgn'/'scripts'
    if str(scripts) not in sys.path: sys.path.insert(0,str(scripts))
    if str(ROOT/'edsgn'/'staging') not in sys.path: sys.path.insert(0,str(ROOT/'edsgn'/'staging'))
    try:
        import simple_baseline_freeze_v1 as guard_impl
        guard_impl._require_launch_guard(checkpoint,'simple frozen test evaluation')
        owner=checkpoint.__self__
        expected={str((root/'logs'/'model-resource.lock').resolve()),str((root/'logs'/'offline-phase-wall'/'campaign.lock').resolve())}
        actual={str(Path(p).resolve()) for p in owner.lock_paths}
        if actual!=expected: raise FrozenSimpleError('guard does not own both required model/campaign locks')
        checkpoint()
    except Exception as exc:
        if isinstance(exc,FrozenSimpleError): raise
        raise FrozenSimpleError(f'requires the owned dual-lock/72-hour phase guard: {exc}') from exc

def frozen_test_evaluate(root: str|Path, dataset_path: str|Path, fit_manifest_path: str|Path,
                         authorization_path: str|Path, method: str, output_dir: str|Path,
                         *, checkpoint: Callable, loader: Callable|None=None,
                         validator: Callable|None=None, before_write: Callable|None=None) -> dict:
    """Guarded evaluation entry. Caller must provide owned dual-lock/budget callback.

    Full campaign dispatcher/stable receipt/global-completion binding are not
    implemented in this preparation revision; no current campaign may launch it.
    """
    root=Path(root).resolve(); data=Path(dataset_path).resolve(); fitp=Path(fit_manifest_path).resolve(); authp=Path(authorization_path).resolve()
    _check_guard(root,checkpoint)
    manifest,auth,coef_path=validate_fit_authorization(root,data,fitp,authp,method,Path(output_dir))
    out=Path(output_dir).resolve()
    try: out.relative_to(root)
    except ValueError as exc: raise FrozenSimpleError('output directory must be contained in RUN root') from exc
    if out.exists(): raise FileExistsError(f'refusing overwrite: {out}')
    before=_input_snapshot(root,data,fitp,coef_path,authp)
    (validator or _official_before_p4)(root)
    if _input_snapshot(root,data,fitp,coef_path,authp)!=before: raise FrozenSimpleError('inputs changed across official before-P4 validator')
    # Only after gate and snapshots do we load coefficients/test arrays.
    if method=='linear': coef=np.load(coef_path,allow_pickle=False)
    else: coef=None
    checkpoint()
    if loader is None:
        from edsgn.engine import datasets_at
        datasets=datasets_at(data,'test')
    else: datasets=loader(data,'test')
    if _input_snapshot(root,data,fitp,coef_path,authp)!=before: raise FrozenSimpleError('inputs changed before test replay')
    expected_nodes=np.asarray(manifest['node_ids'])
    task=manifest['training_statistics']['task']
    for ds in datasets:
        if not np.array_equal(np.asarray(ds.node_ids),expected_nodes): raise FrozenSimpleError('test node order differs from train-fitted mapping')
        if (int(ds.history),int(ds.hmax),list(ds.horizons),bool(ds.count_data))!=(int(task['history']),int(task['hmax']),list(task['horizons']),bool(task['count_data'])):
            raise FrozenSimpleError('test task dimensions differ from train-fitted metadata')
        train_stats=manifest['training_statistics']
        for attr,key in (('mean','input_mean'),('std','input_std'),('scale','residual_scale')):
            if not np.array_equal(np.asarray(getattr(ds,attr)),np.asarray(train_stats[key])):
                raise FrozenSimpleError(f'test preprocessing {attr} differs from train-fitted statistics')
    result,rows=evaluate_preloaded(datasets,method,coef,checkpoint)
    checkpoint()
    provenance={'fit_manifest_sha256':sha256_file(fitp),'authorization_sha256':sha256_file(authp),
      'coefficient_sha256':sha256_file(coef_path) if method=='linear' else None,
      'training_statistics_sha256':manifest.get('training_statistics_sha256'),
      'node_ids_sha256':manifest.get('node_ids_sha256'),'pretest_snapshot_sha256':canonical_hash(before),
      'combined_bundle_materialized':data.is_file(),'test_origins_evaluated':True,'test_feedback_used':False}
    if before_write: before_write()
    receipt=write_result(output_dir,result,rows,method=method,provenance=provenance)
    checkpoint()
    if _input_snapshot(root,data,fitp,coef_path,authp)!=before: raise FrozenSimpleError('inputs changed after output serialization')
    verified=verify_result(output_dir,expected_origins=receipt['n_windows'],method=method)
    return {**receipt,'verification':verified,'scope':'staging_frozen_test_evaluator',
      'test_accessed':True,'source_array_may_materialize_other_partitions':data.is_file(),
      'test_origins_evaluated':True,'global_completion_dispatcher':'pending'}
