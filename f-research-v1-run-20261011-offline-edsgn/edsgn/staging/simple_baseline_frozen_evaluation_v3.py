"""Frozen no-refit evaluator for simple baselines (staging only).

This module deliberately has no fit routine. Production test execution remains
blocked until the root-owned P3/P3A authorization and a campaign dispatcher bind
this interface. The evaluator itself consumes only train-fitted coefficients.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, subprocess, sys, time, zipfile
from pathlib import Path
from typing import Any, Callable
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "edsgn" / "src"
VALIDATOR = ROOT / "edsgn" / "scripts" / "frozen-pipeline-validator" / "scripts" / "pipeline_state.py"
GUARD_SOURCE = ROOT / "edsgn" / "staging" / "simple_baseline_freeze_v2.py"
DOMAIN_PREFLIGHT = ROOT / "edsgn" / "staging" / "c06_domain_manifest_preflight_v4.py"
STATE = ROOT / "pipeline-state.json"
C06_DESCRIPTOR_SCHEMA = "edsgn.c06-domain-preflight.v1"
C06_DESCRIPTOR_SOURCES = [
    "edsgn/staging/c06_domain_manifest_preflight_v4.py",
    "edsgn/staging/split_history_robustness_v2.py",
    "edsgn/staging/frozen_split_history_robustness_v1.py",
    "edsgn/staging/frozen_split_history_robustness_v5.py",
    "edsgn/staging/blind_reference_cost_policy_v1.py",
    "reviews/blind-reference-cost-policy-root-acceptance-v1.json",
    "edsgn/src/edsgn/data.py", "edsgn/src/edsgn/engine.py",
    "edsgn/src/edsgn/robustness.py", "edsgn/src/edsgn/prepare.py",
]

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
    def reject_duplicates(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise FrozenSimpleError(f'duplicate JSON key {key!r} in {path}')
            result[key]=value
        return result
    def reject_constant(value): raise FrozenSimpleError(f'non-standard JSON constant {value!r} in {path}')
    doc=json.loads(path.read_text(encoding='utf-8-sig'),object_pairs_hook=reject_duplicates,parse_constant=reject_constant)
    if not isinstance(doc,dict): raise FrozenSimpleError(f'expected JSON object: {path}')
    return doc


def _valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _run_path(root: Path, raw: str, *, label: str) -> Path:
    if not isinstance(raw,str) or not raw or Path(raw).is_absolute():
        raise FrozenSimpleError(f'{label} must be a nonempty RUN-relative path')
    path=(root/raw).resolve()
    try: path.relative_to(root.resolve())
    except ValueError as exc: raise FrozenSimpleError(f'{label} escapes RUN') from exc
    return path


def _descriptor_sources(root: Path) -> dict[str,str]:
    return {rel:sha256_file(root/rel) for rel in C06_DESCRIPTOR_SOURCES}

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
            stage_seen=set()
            for entry in [*stage.get('inputs',[]),*stage.get('evidence',[])]:
                if not isinstance(entry,dict): raise FrozenSimpleError('malformed passed-stage reference')
                p=_run_path(root,entry.get('path'),label='passed-stage reference')
                digest=entry.get('sha256')
                if not _valid_sha(digest): raise FrozenSimpleError(f'invalid passed-stage SHA-256: {entry.get("path")}')
                key=str(p).casefold()
                if key in stage_seen: raise FrozenSimpleError(f'duplicate passed-stage path: {p}')
                stage_seen.add(key)
                if not p.is_file() or sha256_file(p)!=digest:
                    raise FrozenSimpleError(f'stale passed-stage reference: {p}')
                if str(p) in refs and refs[str(p)]!=digest:
                    raise FrozenSimpleError(f'conflicting passed-stage binding: {p}')
                refs[str(p)]=digest
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


def _c06_domain_bundle(root: Path, dataset_path: Path, auth: dict) -> tuple[dict,dict,Path,Path,list]:
    """Validate the real C06 v4 descriptor/domain using metadata-mask-only reads."""
    descriptor_path=_run_path(root,auth.get('domain_descriptor_path'),label='C06 descriptor')
    descriptor_sha=auth.get('domain_descriptor_sha256')
    if not _valid_sha(descriptor_sha) or not descriptor_path.is_file() or sha256_file(descriptor_path)!=descriptor_sha:
        raise FrozenSimpleError('C06 descriptor path/hash mismatch')
    descriptor=_json(descriptor_path)
    required={'schema','split','ordered_collection','domain_manifest_path','domain_manifest_sha256',
              'source_sha256','decoded_npz_members','header_only_npz_members',
              'forbidden_decoded_members','normalization_source','target_dataset_fingerprint_source'}
    if set(descriptor)!=required or descriptor.get('schema')!=C06_DESCRIPTOR_SCHEMA or descriptor.get('split')!='test':
        raise FrozenSimpleError('C06 descriptor schema/split mismatch')
    domain_path=_run_path(root,descriptor.get('domain_manifest_path'),label='C06 domain manifest')
    domain_sha=descriptor.get('domain_manifest_sha256')
    if not _valid_sha(domain_sha) or not domain_path.is_file() or sha256_file(domain_path)!=domain_sha:
        raise FrozenSimpleError('C06 domain manifest path/hash mismatch')
    expected_sources=_descriptor_sources(root)
    if descriptor.get('source_sha256')!=expected_sources:
        raise FrozenSimpleError('C06 descriptor source inventory differs from current pinned sources')
    if descriptor.get('header_only_npz_members')!=['values'] or set(descriptor.get('forbidden_decoded_members',[]))!={'raw_values','values','targets','target_mask'}:
        raise FrozenSimpleError('C06 descriptor does not declare the expected metadata-only access boundary')
    if not isinstance(descriptor.get('decoded_npz_members'),list) or set(descriptor['decoded_npz_members'])-{'metadata_json','node_ids','input_mask'}:
        raise FrozenSimpleError('C06 descriptor reports a forbidden decoded NPZ member')
    if descriptor.get('normalization_source')!='checkpoint.statistics bound from training-only checkpoint':
        raise FrozenSimpleError('C06 descriptor normalization source is not the frozen training transform')
    if descriptor.get('target_dataset_fingerprint_source')!='engine.dataset_fingerprint hashes original archive bytes without decoding members':
        raise FrozenSimpleError('C06 descriptor target-fingerprint rule mismatch')
    try:
        import c06_domain_manifest_preflight_v4 as preflight
        old_root=preflight.evaluator.ROOT
        preflight.evaluator.ROOT=root.resolve()
        try:
            paths,metadata_datasets,audit,expected_domain=preflight.domain_document(dataset_path,'test')
            actual_domain=_json(domain_path)
            if actual_domain!=expected_domain:
                raise FrozenSimpleError('C06 domain fields/origins differ from metadata-only collection')
            expected_descriptor=preflight.collection_descriptor(paths,metadata_datasets,audit,domain_path,'test')
        finally:
            preflight.evaluator.ROOT=old_root
    except FrozenSimpleError:
        raise
    except Exception as exc:
        raise FrozenSimpleError(f'C06 metadata-only descriptor validation failed: {exc}') from exc
    if descriptor!=expected_descriptor:
        raise FrozenSimpleError('C06 descriptor collection path/order/bytes/shape does not match current collection')
    trajectories=actual_domain.get('trajectories')
    if not isinstance(trajectories,list) or not trajectories:
        raise FrozenSimpleError('C06 domain has no trajectories')
    expected_pairs=[]
    for index,row in enumerate(trajectories):
        if not isinstance(row,dict) or row.get('trajectory_index')!=index or row.get('split')!='test':
            raise FrozenSimpleError('C06 domain trajectory index/split mismatch')
        first,last,count=(row.get('origin_first'),row.get('origin_last'),row.get('origin_count'))
        history=row.get('history_length')
        if any(type(x) is not int for x in (first,last,count,history)) or count<1 or history<1 or last-first+1!=count:
            raise FrozenSimpleError('C06 domain origin interval is malformed or incomplete')
        if row.get('history_union_start_inclusive')!=first-history+1 or row.get('history_union_end_exclusive')!=last+1:
            raise FrozenSimpleError('C06 domain history bounds do not match the full ordered origin interval')
        if row.get('domain_bounds_start_inclusive')!=first-history+1 or row.get('domain_bounds_end_exclusive')!=last+1:
            raise FrozenSimpleError('C06 domain declared bounds mismatch')
        expected_pairs.extend((index,o) for o in range(first,last+1))
    return descriptor,actual_domain,descriptor_path,domain_path,expected_pairs

def _input_snapshot(root: Path, dataset_path: Path, fit_path: Path, coeff_path: Path,
                    authorization_path: Path) -> dict[str,str]:
    state=_stage_state(root)
    refs=_passed_refs(root,state)
    auth=_json(authorization_path)
    descriptor,domain,descriptor_path,domain_path,_=_c06_domain_bundle(root,dataset_path,auth)
    for p in (fit_path,coeff_path,authorization_path,descriptor_path,domain_path,
              Path(__file__).resolve(),VALIDATOR,GUARD_SOURCE,DOMAIN_PREFLIGHT,
              SRC/'edsgn'/'engine.py',SRC/'edsgn'/'data.py'):
        if not p.is_file(): raise FrozenSimpleError(f'bound input missing: {p}')
        refs[str(p.resolve())]=sha256_file(p)
    for rel,digest in descriptor['source_sha256'].items():
        path=_run_path(root,rel,label='C06 transitive source')
        if not path.is_file() or sha256_file(path)!=digest: raise FrozenSimpleError(f'C06 source binding changed: {rel}')
        refs[str(path)]=digest
    refs.update(_collection_snapshot(dataset_path))
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
        seen=set()
        for entry in [*stage.get('inputs',[]),*stage.get('evidence',[])]:
            if not isinstance(entry,dict): raise FrozenSimpleError('malformed P3/P3A stage reference')
            path=_run_path(root,entry.get('path'),label='P3/P3A stage reference')
            digest=entry.get('sha256')
            if not _valid_sha(digest): raise FrozenSimpleError(f'invalid P3/P3A reference SHA-256: {path}')
            key=str(path).casefold()
            if key in seen: raise FrozenSimpleError(f'duplicate P3/P3A stage path: {path}')
            seen.add(key); result[str(path)]=digest
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
    descriptor,domain,descriptor_path,domain_path,expected_pairs=_c06_domain_bundle(root,dataset_path,auth)
    expected={
      'schema_version':1,'status':'accepted_pretest_authorization','method':method,
      'dataset_path':str(dataset_path.resolve()),'fit_manifest_path':str(fit_path.resolve()),
      'fit_manifest_sha256':sha256_file(fit_path),'output_dir':str(output_dir.resolve()),
      'source_collection_fingerprint_sha256':canonical_hash(_collection_snapshot(dataset_path)),
      'development_acceptance_path':auth.get('development_acceptance_path'),
      'development_acceptance_sha256':auth.get('development_acceptance_sha256'),
      'domain_descriptor_path':auth.get('domain_descriptor_path')}
    current_sources={'evaluator':sha256_file(Path(__file__).resolve()),'engine':sha256_file(SRC/'edsgn'/'engine.py'),
                     'data':sha256_file(SRC/'edsgn'/'data.py'),'guard':sha256_file(GUARD_SOURCE),
                     'domain_preflight':sha256_file(DOMAIN_PREFLIGHT)}
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


def _expected_metric_rows(result, horizons, method):
    y=np.asarray(result['targets']); pred=np.asarray(result['predictions']); mask=np.asarray(result['target_mask'],dtype=bool)
    rows=[]
    for h in horizons:
        for feature in range(y.shape[-1]):
            valid=mask[:,h-1,:,feature]
            error=pred[:,h-1,:,feature]-y[:,h-1,:,feature]
            n=int(valid.sum())
            rows.append({'method':method,'seed':0,'split':'test','horizon':int(h),'feature':int(feature),
                'stratum':'all','mae':float(np.abs(error)[valid].mean()) if n else None,
                'rmse':float(np.sqrt(np.square(error)[valid].mean())) if n else None,
                'n_scored':n,'n_windows':len(y)})
    return rows


def _assert_full_arrays(result, expected_pairs, *, hmax, n_nodes, n_features):
    expected_pairs=np.asarray(expected_pairs,dtype=np.int64)
    pairs=np.asarray(result.get('pairs'))
    pred=np.asarray(result.get('predictions')); target=np.asarray(result.get('targets'))
    mask=np.asarray(result.get('target_mask'))
    expected_shape=(len(expected_pairs),hmax,n_nodes,n_features)
    if pairs.dtype.kind not in 'iu' or pairs.shape!=expected_pairs.shape or not np.array_equal(pairs,expected_pairs):
        raise FrozenSimpleError('evaluated origin pairs are missing, reordered, duplicated, or out of domain')
    if pred.shape!=expected_shape or target.shape!=expected_shape or mask.shape!=expected_shape:
        raise FrozenSimpleError(f'prediction/target/mask shape is not the complete frozen task: {pred.shape,target.shape,mask.shape} != {expected_shape}')
    if mask.dtype!=np.bool_: raise FrozenSimpleError('clean target mask must be boolean')
    if not np.isfinite(pred).all() or not np.isfinite(target[mask]).all():
        raise FrozenSimpleError('predictions or valid clean targets contain nonfinite values')
    return expected_pairs

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

def verify_result(output_dir: str|Path, *, expected_result: dict, expected_rows: list[dict],
                  expected_pairs, method: str, horizons, task) -> dict:
    out=Path(output_dir).resolve(); mp=out/'evaluation_manifest.json'; doc=_json(mp)
    expected_pairs=np.asarray(expected_pairs,dtype=np.int64)
    expected_n=len(expected_pairs)
    if doc.get('scope')!='full_protocol' or doc.get('split')!='test' or doc.get('method')!=method or doc.get('n_windows')!=expected_n:
        raise FrozenSimpleError('manifest scope/method/full-origin count mismatch')
    output_hashes=doc.get('output_hashes')
    if not isinstance(output_hashes,dict) or set(output_hashes)!={'predictions.npz','metrics_by_horizon.csv'}:
        raise FrozenSimpleError('manifest must bind the exact prediction and raw-metric outputs')
    for name,digest in output_hashes.items():
        if not _valid_sha(digest): raise FrozenSimpleError(f'invalid output SHA-256: {name}')
        p=out/name
        if not p.is_file() or sha256_file(p)!=digest: raise FrozenSimpleError(f'output hash mismatch: {p}')
    with np.load(out/'predictions.npz',allow_pickle=False) as z:
        required={'predictions','targets','target_mask','origins','seed_id'}
        if set(z.files)!=required: raise FrozenSimpleError('saved prediction archive has missing/extra members')
        seed=np.asarray(z['seed_id'])
        if seed.shape!=() or seed.dtype.kind not in 'iu' or int(seed)!=0:
            raise FrozenSimpleError('saved predictions must carry seed_id=0')
        saved={key:np.asarray(z[key]) for key in ('predictions','targets','target_mask','origins')}
    _assert_full_arrays(expected_result,expected_pairs,hmax=task['hmax'],n_nodes=task['n_nodes'],n_features=task['n_features'])
    if saved['origins'].shape!=expected_pairs.shape or saved['origins'].dtype.kind not in 'iu' or not np.array_equal(saved['origins'],expected_pairs):
        raise FrozenSimpleError('saved origins have a gap, duplicate, reordering, or wrong trajectory ID')
    for key in ('predictions','targets','target_mask'):
        reference=np.asarray(expected_result[key])
        if saved[key].shape!=reference.shape or saved[key].dtype!=reference.dtype or not np.array_equal(saved[key],reference,equal_nan=True):
            raise FrozenSimpleError(f'saved {key} differs from evaluated complete arrays')
    expected_metrics=_expected_metric_rows(expected_result,horizons,method)
    if expected_rows!=expected_metrics: raise FrozenSimpleError('in-memory raw horizon metrics differ from recomputed values')
    with (out/'metrics_by_horizon.csv').open('r',newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; rows=list(reader)
    expected_fields=list(expected_metrics[0]) if expected_metrics else []
    if fields!=expected_fields or len(rows)!=len(expected_metrics):
        raise FrozenSimpleError('saved horizon-metric schema/row coverage mismatch')
    int_fields={'seed','horizon','feature','n_scored','n_windows'}
    float_fields={'mae','rmse'}
    for saved_row,expected_row in zip(rows,expected_metrics):
        for key in ('method','split','stratum'):
            if saved_row[key]!=str(expected_row[key]): raise FrozenSimpleError(f'saved metric {key} mismatch')
        for key in int_fields:
            if int(saved_row[key])!=expected_row[key]: raise FrozenSimpleError(f'saved metric {key} mismatch')
        for key in float_fields:
            expected_value=expected_row[key]
            if expected_value is None:
                if saved_row[key] not in ('','None'): raise FrozenSimpleError(f'saved metric {key} should be null')
            else:
                try: actual=float(saved_row[key])
                except ValueError as exc: raise FrozenSimpleError(f'saved metric {key} is not numeric') from exc
                if not math.isfinite(actual) or not math.isclose(actual,expected_value,rel_tol=1e-12,abs_tol=1e-12):
                    raise FrozenSimpleError(f'saved metric {key} differs from raw pooled cells')
    return {'manifest_sha256':sha256_file(mp),'verified_outputs':output_hashes,'n_windows':expected_n,
            'ordered_origins_sha256':hashlib.sha256(expected_pairs.tobytes()).hexdigest()}

def _check_guard(root: Path, checkpoint: Callable) -> None:
    scripts=ROOT/'edsgn'/'scripts'
    if str(scripts) not in sys.path: sys.path.insert(0,str(scripts))
    if str(ROOT/'edsgn'/'staging') not in sys.path: sys.path.insert(0,str(ROOT/'edsgn'/'staging'))
    try:
        import simple_baseline_freeze_v2 as guard_impl
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
                         validator: Callable|None=None, before_write: Callable|None=None,
                         managed_job_spec: str|Path, pretest_admission_binding: str|Path) -> dict:
    """Run a metadata-authorized, frozen simple baseline test evaluation.

    Mathematical prediction and metric code above is copied from evaluator2.
    No fitting occurs here; admission and all source/input fences precede test
    loading, and the managed CLI owns both leases until its completed receipt.
    """
    import simple_frozen_test_admission_v1 as admission
    root=Path(root).resolve(); data=Path(dataset_path).resolve(); fitp=Path(fit_manifest_path).resolve()
    jobp=Path(managed_job_spec).resolve(); bindingp=Path(pretest_admission_binding).resolve()
    job=_json(jobp)
    binding_ref={"path":str(bindingp.relative_to(root)).replace("\\","/"),"sha256":sha256_file(bindingp)}
    _check_guard(root,checkpoint)
    snapshot=admission.validate_admission(root,binding_ref,job,jobp)
    def refence():
        now={k:sha256_file(root/k) for k in snapshot}
        if now!=snapshot: raise FrozenSimpleError("managed admission input/source fence changed")
    (validator or _official_before_p4)(root)
    refence()
    checkpoint()
    refence()
    # Numeric artifacts are first opened after the official gate and final
    # callback. The helper has already checked their immutable metadata hashes.
    fitdoc=_json(fitp)
    coef_path=Path(fitdoc["linear_coefficients_path"])
    if not coef_path.is_absolute(): coef_path=fitp.parent/coef_path
    coef=np.load(coef_path,allow_pickle=False) if method=="linear" else None
    if loader is None:
        from edsgn.engine import datasets_at
        datasets=datasets_at(data,"test")
    else: datasets=loader(data,"test")
    refence()
    # Validate only declared test-domain geometry and train-fitted transforms.
    auth=_json(bindingp)
    descriptor_ref=auth["domain"]["descriptor"]
    domain_descriptor,domain_doc,descriptor_path,domain_path,expected_pairs=_c06_domain_bundle(
        root,data,{"domain_descriptor_path":descriptor_ref["path"],"domain_descriptor_sha256":descriptor_ref["sha256"]})
    from edsgn.engine import index_origins
    pairs=index_origins(datasets,"test")
    if pairs!=expected_pairs: raise FrozenSimpleError("loaded test origins differ from frozen domain")
    task=fitdoc["training_statistics"]["task"]
    for ds in datasets:
        if not np.array_equal(np.asarray(ds.node_ids),np.asarray(fitdoc["node_ids"])): raise FrozenSimpleError("test node ordering differs from train fit")
        for attr,key in (("mean","input_mean"),("std","input_std"),("scale","residual_scale")):
            if not np.array_equal(np.asarray(getattr(ds,attr)),np.asarray(fitdoc["training_statistics"][key])): raise FrozenSimpleError(f"test {attr} differs from train-fitted stats")
    result,rows=evaluate_preloaded(datasets,method,coef,checkpoint)
    if result["pairs"]!=expected_pairs: raise FrozenSimpleError("evaluation origins are not complete/frozen")
    _assert_full_arrays(result,expected_pairs,hmax=task["hmax"],n_nodes=task["n_nodes"],n_features=task["n_features"])
    if rows!=_expected_metric_rows(result,task["horizons"],method): raise FrozenSimpleError("metrics failed independent raw-unit recomputation")
    checkpoint()
    refence()
    provenance={"admission_sha256":sha256_file(bindingp),"fit_manifest_sha256":sha256_file(fitp),
      "coefficient_sha256":sha256_file(coef_path) if method=="linear" else None,
      "pretest_snapshot_sha256":canonical_hash(snapshot),"combined_bundle_materialized":data.is_file(),
      "scored_test_origins":True,"test_feedback_used":False,"oracle_accessed":False}
    if before_write: before_write()
    refence()
    receipt=write_result(output_dir,result,rows,method=method,provenance=provenance)
    checkpoint()
    refence()
    verify_result(output_dir,expected_result=result,expected_rows=rows,expected_pairs=expected_pairs,
                  method=method,horizons=task["horizons"],task=task)
    return {**receipt,"scope":"managed_simple_frozen_test","test_accessed":True,
      "source_array_may_materialize_other_partitions":data.is_file(),"test_origins_evaluated":True,
      "global_completion_dispatcher":"pending"}


def main(argv=None):
    import argparse, uuid
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path",required=True); parser.add_argument("--fit-manifest",required=True)
    parser.add_argument("--authorization",required=True); parser.add_argument("--method",choices=("persistence","linear"),required=True)
    parser.add_argument("--split",choices=("test",),required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--managed-job-spec",required=True); parser.add_argument("--pretest-admission-binding",required=True)
    args=parser.parse_args(argv); root=ROOT.resolve(); out=(root/args.output).resolve()
    job_id=f"simple-frozen-test-{uuid.uuid4().hex}"; started=time.perf_counter(); epoch=time.time()
    guard_impl=None; guard=None; status="failed"; error=None; outputs=None
    try:
        import simple_baseline_freeze_v2 as guard_impl
        command=[sys.executable,str(Path(__file__).resolve()),*(argv or sys.argv[1:])]
        job_doc=_json(Path(args.managed_job_spec).resolve())
        expected_cli={"--dataset-path":args.dataset_path,"--fit-manifest":args.fit_manifest,
            "--authorization":args.authorization,"--method":args.method,"--split":args.split,
            "--output":args.output,"--managed-job-spec":args.managed_job_spec,
            "--pretest-admission-binding":args.pretest_admission_binding}
        parsed_cli={}
        raw=list(argv) if argv is not None else list(sys.argv[1:])
        if len(raw)%2: raise FrozenSimpleError("managed simple CLI requires exact flag/value pairs")
        for i in range(0,len(raw),2):
            if raw[i] in parsed_cli: raise FrozenSimpleError("duplicate simple CLI flag")
            parsed_cli[raw[i]]=raw[i+1]
        if parsed_cli!=expected_cli or job_doc.get("argv")!=raw:
            raise FrozenSimpleError("CLI arguments differ from immutable managed job")
        guard=guard_impl._acquire_launch_guard(root,out,"evaluation",job_id,started,epoch,command)
        guard.checkpoint()
        result=frozen_test_evaluate(root,args.dataset_path,args.fit_manifest,args.authorization,args.method,out,
            checkpoint=guard.checkpoint,managed_job_spec=args.managed_job_spec,
            pretest_admission_binding=args.pretest_admission_binding)
        outputs={}
        for path in sorted(out.iterdir()):
            if not path.is_file(): raise FrozenSimpleError("unexpected nested output in simple test directory")
            outputs[path.name]=sha256_file(path)
        expected={"predictions.npz","metrics_by_horizon.csv","evaluation_manifest.json"}
        if set(outputs)!=expected: raise FrozenSimpleError("simple test outputs are incomplete/extra")
        guard.checkpoint()
        status="completed"
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except BaseException as exc:
        error=exc; status="stopped" if isinstance(exc,FrozenSimpleError) else "failed"; raise
    finally:
        if guard_impl is not None:
            try: guard_impl._phase_receipt(root,out,"evaluation",job_id,started,epoch,status,
                [sys.executable,str(Path(__file__).resolve()),*(argv or sys.argv[1:])],
                token=guard.token if guard else None,error=error,outputs=outputs if status=="completed" else None)
            finally:
                if guard is not None: guard.release()

if __name__=="__main__":
    raise SystemExit(main())
