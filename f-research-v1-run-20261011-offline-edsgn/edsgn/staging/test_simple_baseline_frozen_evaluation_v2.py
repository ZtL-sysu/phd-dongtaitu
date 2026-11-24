"""CPU-only temporary fixtures for the simple frozen test evaluator."""
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'edsgn'/'src'))
SPEC=importlib.util.spec_from_file_location('simple_frozen_v2',Path(__file__).with_name('simple_baseline_frozen_evaluation_v2.py'))
mod=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(mod)

class ToyDS:
    def __init__(self, count=True, future_delta=0):
        self.history=2;self.hmax=2;self.horizons=[1,2];self.count_data=count
        self.values=np.array([[[0.],[1.]],[[1.],[0.]],[[2.],[1.]],[[3.],[2.]],[[4.],[3.]],[[5.],[4.]],[[6.],[5.]],[[7.],[6.]]],dtype=np.float32)
        self.raw=self.values+10
        self.targets=np.zeros((8,2,2,1),dtype=np.float32)
        for t in range(8): self.targets[t,:, :,0]=t+np.array([[1,2],[3,4]])
        self.targets[7]+=future_delta
        self.target_mask=np.ones_like(self.targets,dtype=bool)
        self.input_mask=np.ones_like(self.values,dtype=bool)
        self.splits={'train':(0,4),'validation':(4,6),'test':(4,8)}
        self.node_ids=np.array([10,11]);self.mean=np.array([0.]);self.std=np.array([1.]);self.scale=np.array([[1.],[1.]])
        self.metadata={'source':'toy','seed':0};self.partition='chronological';self.event_kind='synthetic'
        self.target_calls=[]
    def origins(self,split):
        a,b=self.splits['test' if split=='test' else 'train' if split=='train' else 'validation']
        first=max(self.history-1,a-1);last=b-self.hmax-1
        return np.arange(first,last+1) if last>=first else np.empty(0,dtype=np.int64)
    def window(self,origin):
        sl=slice(origin-self.history+1,origin+1)
        return {'values':self.values[sl], 'raw_observations':self.raw[sl]}
    def target(self,origin):
        self.target_calls.append(origin)
        return self.targets[origin],self.target_mask[origin]

class Guard:
    def checkpoint(self): pass

class Caller:
    def checkpoint(self): pass

def _owned_guard(run, output, monkeypatch):
    import simple_baseline_freeze_v1 as guard_impl
    monkeypatch.setattr(guard_impl,'_owned_model_workers',lambda _root:[])
    monkeypatch.setattr(guard_impl,'_budget_report',lambda *_a,**_k:{'coverage_complete':True,'spent_seconds':1.0})
    locks=[run/'logs/model-resource.lock',run/'logs/offline-phase-wall/campaign.lock']
    for lock in locks:
        lock.parent.mkdir(parents=True,exist_ok=True);lock.write_text(json.dumps({'token':'test-token'}),encoding='utf-8')
    return guard_impl._LaunchGuard(run,output,'evaluation','toy',time.perf_counter(),time.time(),['python'],locks,'test-token')


def test_persistence_linear_match_legacy_math_clip_and_future_invariance():
    ds=ToyDS(count=True)
    origins=ds.origins('test')
    coef=np.array([[1.0,0.0],[0.0,0.0],[0.5,2.0]],dtype=np.float64)
    linear=mod.predict_dataset(ds,origins,'linear',coef)
    manual=[]
    for o in origins:
        x=ds.window(o)['values'].transpose(1,0,2).reshape(2,-1)
        x=np.concatenate([x,np.ones((2,1))],axis=1)
        manual.append(np.maximum((x@coef).reshape(-1,2,1).transpose(1,0,2),0))
    assert np.array_equal(linear['predictions'],np.asarray(manual))
    pers=mod.predict_dataset(ds,origins,'persistence')
    expected=np.repeat(ds.window(origins[0])['raw_observations'][-1:],2,axis=0)
    assert np.array_equal(pers['predictions'][0],expected)
    import simple_baseline_freeze_v1 as legacy
    for name,coef_arg in [('persistence',None),('linear',coef)]:
        old,_=legacy._evaluate_method([ds],'test',name,coef_arg)
        new=mod.predict_dataset(ds,origins,name,coef_arg)
        assert np.array_equal(new['predictions'],old['predictions'])
        assert np.array_equal(new['targets'],old['targets'])
        assert np.array_equal(new['target_mask'],old['target_mask'])
    assert ds.target_calls==[3,4,5]*6
    changed=ToyDS(count=True,future_delta=999)
    assert np.array_equal(mod.predict_dataset(ds,origins,'linear',coef)['predictions'],
                          mod.predict_dataset(changed,origins,'linear',coef)['predictions'])
    signed=ToyDS(count=False)
    negcoef=np.array([[0.0,0.0],[0.0,0.0],[-1.0,-1.0]],dtype=np.float64)
    assert np.any(mod.predict_dataset(signed,origins,'linear',negcoef)['predictions']<0)
    assert np.all(mod.predict_dataset(ds,origins,'linear',negcoef)['predictions']>=0)


def _write_authorized_fixture(tmp_path, monkeypatch):
    run=tmp_path/'run';run.mkdir()
    data=run/'data.npz';toy=ToyDS()
    metadata={'splits':{k:list(v) for k,v in toy.splits.items()},'count_data':True,'horizons':[1,2],
        'history':2,'hmax':2,'event_kind':'synthetic','metadata':toy.metadata,'partition':toy.partition}
    np.savez_compressed(data,metadata_json=np.asarray(json.dumps(metadata)),values=toy.values,
        input_mask=toy.input_mask,node_ids=toy.node_ids)
    out=run/'p3';out.mkdir()
    cp=out/'linear_coefficients.npy'
    np.save(cp,np.array([[1.,0.],[0.,0.],[.5,2.]],dtype=np.float64),allow_pickle=False)
    nodes=np.asarray([10,11],dtype=np.int64);nodehash=hashlib.sha256(nodes.tobytes(order='C')).hexdigest()
    stats={'node_ids_sha256':nodehash,'input_mean':[0.0],'input_std':[1.0],'residual_scale':[[1.0],[1.0]],'task':{'n_nodes':2,'n_features':1,'history':2,'hmax':2,'horizons':[1,2],'count_data':True}}
    fpgr={'data.npz':mod.sha256_file(data)}
    src={'engine.py':mod.sha256_file(ROOT/'edsgn/src/edsgn/engine.py'),'data.py':mod.sha256_file(ROOT/'edsgn/src/edsgn/data.py')}
    fit={'schema_version':1,'artifact_role':'train_only_simple_baseline_fit','dataset_path':str(data.resolve()),'linear_coefficients_path':str(cp.resolve()),
       'linear_coefficients_sha256':mod.sha256_file(cp),'training_statistics':stats,'training_statistics_sha256':mod.canonical_hash(stats),
       'training_source_fingerprint':fpgr,'training_source_fingerprint_sha256':mod.canonical_hash(fpgr),
       'source_hashes':src,'source_hashes_sha256':mod.canonical_hash(src),
       'node_ids':nodes.tolist(),'node_ids_sha256':nodehash,'train_node_window_samples':100,'split':'train','test_accessed':False}
    fp=out/'fit.json';fp.write_text(json.dumps(fit),encoding='utf-8')
    # Copy the pinned C06 source inventory into this temporary RUN so the
    # descriptor uses the production v4 schema and normal path rules.
    for rel in mod.C06_DESCRIPTOR_SOURCES:
        dst=run/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/rel).read_bytes())
    import c06_domain_manifest_preflight_v4 as preflight
    old_root=preflight.evaluator.ROOT;preflight.evaluator.ROOT=run
    try:
        paths,metadata_ds,audit,domain_doc=preflight.domain_document(data,'test')
        domain=run/'domain.json';domain.write_text(json.dumps(domain_doc,ensure_ascii=False,indent=2),encoding='utf-8')
        descriptor=preflight.collection_descriptor(paths,metadata_ds,audit,domain,'test')
        desc_path=run/'descriptor.json';desc_path.write_text(json.dumps(descriptor,ensure_ascii=False,indent=2),encoding='utf-8')
    finally: preflight.evaluator.ROOT=old_root
    metric=out/'linear_val_metrics.csv';metric.write_text('horizon,mae\n1,1.0\n',encoding='utf-8')
    dev={'accepted_by_root':True,'verdict':'pass','scope':'simple-baseline-development-fit',
         'fit_manifest_path':str(fp.resolve()),'fit_manifest_sha256':mod.sha256_file(fp),
         'accepted_validation_metrics':[{'path':metric.relative_to(run).as_posix(),'sha256':mod.sha256_file(metric)}]}
    devp=out/'development-acceptance.json';devp.write_text(json.dumps(dev),encoding='utf-8')
    state={'schema_version':1,'pipeline':'f-research-v1','stages':[
      {'id':'P2','status':'passed','gate_verdict':'pass','inputs':[],'evidence':[]},
      {'id':'P3','status':'passed','gate_verdict':'pass','inputs':[], 'evidence':[
          {'path':fp.relative_to(run).as_posix(),'sha256':mod.sha256_file(fp)},
          {'path':cp.relative_to(run).as_posix(),'sha256':mod.sha256_file(cp)},
          {'path':metric.relative_to(run).as_posix(),'sha256':mod.sha256_file(metric)},
          {'path':devp.relative_to(run).as_posix(),'sha256':mod.sha256_file(devp)}]},
      {'id':'P3A','status':'passed','gate_verdict':'pass','inputs':[],'evidence':[]},
      {'id':'P4','status':'running','gate_verdict':'pending','inputs':[],'evidence':[]}]}
    current_sources={'evaluator':mod.sha256_file(Path(mod.__file__).resolve()),
        'engine':mod.sha256_file(ROOT/'edsgn/src/edsgn/engine.py'),'data':mod.sha256_file(ROOT/'edsgn/src/edsgn/data.py'),
        'guard':mod.sha256_file(mod.GUARD_SOURCE),'domain_preflight':mod.sha256_file(mod.DOMAIN_PREFLIGHT)}
    auth={'schema_version':1,'status':'accepted_pretest_authorization','accepted_by_root':True,'verdict':'pass',
       'method':'linear','dataset_path':str(data.resolve()),'fit_manifest_path':str(fp.resolve()),
       'fit_manifest_sha256':mod.sha256_file(fp),'coefficient_path':str(cp.resolve()),
       'coefficient_sha256':mod.sha256_file(cp),'training_statistics_sha256':fit['training_statistics_sha256'],'node_ids_sha256':nodehash,
       'training_source_fingerprint_sha256':fit['training_source_fingerprint_sha256'],'source_hashes_sha256':fit['source_hashes_sha256'],
       'output_dir':str((run/'test-output').resolve()),
       'source_collection_fingerprint_sha256':mod.canonical_hash(mod._collection_snapshot(data)),
       'development_acceptance_path':devp.relative_to(run).as_posix(),'development_acceptance_sha256':mod.sha256_file(devp),
       'accepted_validation_metrics':dev['accepted_validation_metrics'],
       'domain_descriptor_path':'descriptor.json','domain_descriptor_sha256':mod.sha256_file(desc_path),
       'source_current_sha256':current_sources}
    ap=out/'authorization.json';ap.write_text(json.dumps(auth),encoding='utf-8')
    state['stages'][2]['evidence']=[{'path':ap.relative_to(run).as_posix(),'sha256':mod.sha256_file(ap)}]
    (run/'pipeline-state.json').write_text(json.dumps(state),encoding='utf-8')
    return run,data,fp,ap,cp,state


def test_gate_runs_before_loader_and_full_result_hashes_verify(tmp_path,monkeypatch):
    run,data,fp,ap,cp,state=_write_authorized_fixture(tmp_path,monkeypatch)
    order=[];ds=ToyDS();output=run/'test-output'
    guard=_owned_guard(run,output,monkeypatch)
    def validator(root): order.append('gate')
    def loader(path,split): order.append('loader');assert split=='test';return [ds]
    result=mod.frozen_test_evaluate(run,data,fp,ap,'linear',output,checkpoint=guard.checkpoint,
         loader=loader,validator=validator)
    assert order==['gate','loader']
    assert result['n_windows']==3 and result['test_accessed'] is True
    assert result['verification']['n_windows']==3
    with np.load(output/'predictions.npz',allow_pickle=False) as z:
        assert np.array_equal(z['origins'],np.array([[0,3],[0,4],[0,5]]))
        assert np.array_equal(z['target_mask'],np.ones((3,2,2,1),dtype=bool))
    saved_manifest=json.loads((output/'evaluation_manifest.json').read_text())
    assert saved_manifest['combined_bundle_materialized'] is True and saved_manifest['test_feedback_used'] is False
    assert mod.sha256_file(output/'metrics_by_horizon.csv')
    assert len(list(__import__('csv').DictReader((output/'metrics_by_horizon.csv').open())))==2
    assert mod.sha256_file(cp)==json.loads(fp.read_text())['linear_coefficients_sha256']
    guard.release()


def test_pre_validator_mutation_refuses_before_loader(tmp_path,monkeypatch):
    run,data,fp,ap,cp,state=_write_authorized_fixture(tmp_path,None)
    guard=_owned_guard(run,run/'test-output',monkeypatch)
    called=[]
    def validator(root): cp.write_bytes(b'changed coefficient')
    with pytest.raises(mod.FrozenSimpleError,match='stale passed-stage reference'):
        mod.frozen_test_evaluate(run,data,fp,ap,'linear',run/'test-output',checkpoint=guard.checkpoint,
          loader=lambda *_:(called.append(True) or [ToyDS()]),validator=validator)
    assert not called


def test_final_budget_callback_mutation_refuses_before_test_loader(tmp_path,monkeypatch):
    run,data,fp,ap,cp,state=_write_authorized_fixture(tmp_path,None)
    guard=_owned_guard(run,run/'test-output',monkeypatch)
    import simple_baseline_freeze_v1 as guard_impl
    calls=[]; loader_calls=[]
    def budget_report(*_args,**_kwargs):
        calls.append(1)
        # _check_guard is the first checkpoint; mutate at the explicit final
        # checkpoint after coefficient load, before the test loader.
        if len(calls)==2: data.write_bytes(data.read_bytes()+b'changed-after-budget')
        return {'coverage_complete':True,'spent_seconds':1.0}
    monkeypatch.setattr(guard_impl,'_budget_report',budget_report)
    # Collection membership/descriptor validation itself detects this archive
    # byte mutation during the immediate pre-load full snapshot.
    with pytest.raises(mod.FrozenSimpleError,match='C06 descriptor collection|inputs changed after final pre-load budget check'):
        mod.frozen_test_evaluate(run,data,fp,ap,'linear',run/'test-output',checkpoint=guard.checkpoint,
          loader=lambda *_:(loader_calls.append(True) or [ToyDS()]),validator=lambda _root:None)
    assert len(calls)==2 and loader_calls==[]
    guard.release()


def test_actual_p3_inventory_is_metadata_only_and_not_admitted():
    rows=[]
    for domain in ('synthetic','email','mooc','email-episode'):
        out=ROOT/'runs'/'baselines'/domain/'simple'
        item=mod.inspect_existing_p3_fit(out);rows.append(item)
        assert item['split']=='val' and item['test_accessed'] is False
        assert item['coefficient_header']['shape'] and item['coefficient_sha256']
        assert item['admission_ready'] is False
        assert 'training_statistics_sha256' in item['missing_fit_bindings']
    assert len(rows)==4

def test_actual_legacy_p3_manifest_is_not_admissible_fit_bundle(tmp_path):
    path=ROOT/'runs/baselines/email/simple/simple-baseline-manifest.json'
    run=tmp_path/'run';run.mkdir()
    state={'stages':[{'id':'P2','status':'passed','gate_verdict':'pass'},
      {'id':'P3','status':'passed','gate_verdict':'pass'},
      {'id':'P3A','status':'passed','gate_verdict':'pass'},
      {'id':'P4','status':'running'}]}
    (run/'pipeline-state.json').write_text(json.dumps(state),encoding='utf-8')
    data=run/'fake.npz';data.write_bytes(b'opaque')
    auth=run/'auth.json';auth.write_text('{}',encoding='utf-8')
    with pytest.raises(mod.FrozenSimpleError,match='not a completed train-only artifact'):
        mod.validate_fit_authorization(run,data,path,auth,'linear',run/'out')


def test_output_verifier_rejects_prediction_tampering(tmp_path):
    result={'predictions':np.zeros((1,2,2,1)), 'targets':np.zeros((1,2,2,1)),
       'target_mask':np.ones((1,2,2,1),dtype=bool),'pairs':[(0,1)]}
    receipt=mod.write_result(tmp_path/'out',result,[{'method':'linear','seed':0,'split':'test','horizon':1,'feature':0,'stratum':'all','mae':0.,'rmse':0.,'n_scored':2,'n_windows':1}],method='linear',provenance={})
    expected={'predictions':result['predictions'],'targets':result['targets'],'target_mask':result['target_mask'],'pairs':result['pairs']}
    rows=[{'method':'linear','seed':0,'split':'test','horizon':1,'feature':0,'stratum':'all','mae':0.,'rmse':0.,'n_scored':2,'n_windows':1}]
    task={'hmax':2,'n_nodes':2,'n_features':1}
    assert mod.verify_result(tmp_path/'out',expected_result=expected,expected_rows=rows,expected_pairs=[[0,1]],
        method='linear',horizons=[1],task=task)['n_windows']==1
    (tmp_path/'out'/'predictions.npz').write_bytes(b'changed')
    with pytest.raises(mod.FrozenSimpleError,match='output hash mismatch'):
        mod.verify_result(tmp_path/'out',expected_result=expected,expected_rows=rows,expected_pairs=[[0,1]],
            method='linear',horizons=[1],task=task)


def _write_verifiable_result(out):
    result={'predictions':np.ones((2,2,2,1),dtype=np.float32),
       'targets':np.zeros((2,2,2,1),dtype=np.float32),
       'target_mask':np.ones((2,2,2,1),dtype=bool),'pairs':[(0,3),(0,4)]}
    task={'hmax':2,'n_nodes':2,'n_features':1}; horizons=[1,2]
    rows=mod._expected_metric_rows(result,horizons,'persistence')
    mod.write_result(out,result,rows,method='persistence',provenance={})
    return result,rows,task,horizons


def _refresh_output_hash(out,name):
    manifest=json.loads((out/'evaluation_manifest.json').read_text(encoding='utf-8'))
    manifest['output_hashes'][name]=mod.sha256_file(out/name)
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')


def test_verifier_rejects_reordered_or_duplicate_origins_even_with_matching_hash(tmp_path):
    for variant in ('reordered','duplicate'):
        out=tmp_path/variant; result,rows,task,horizons=_write_verifiable_result(out)
        origins=np.asarray([[0,4],[0,3]] if variant=='reordered' else [[0,3],[0,3]],dtype=np.int64)
        with np.load(out/'predictions.npz',allow_pickle=False) as z:
            members={k:np.asarray(z[k]) for k in z.files}
        members['origins']=origins
        np.savez_compressed(out/'predictions.npz',**members)
        _refresh_output_hash(out,'predictions.npz')
        with pytest.raises(mod.FrozenSimpleError,match='saved origins'):
            mod.verify_result(out,expected_result=result,expected_rows=rows,expected_pairs=[[0,3],[0,4]],
              method='persistence',horizons=horizons,task=task)


def test_verifier_recomputes_metrics_even_if_manifest_hash_is_refreshed(tmp_path):
    out=tmp_path/'out'; result,rows,task,horizons=_write_verifiable_result(out)
    text=(out/'metrics_by_horizon.csv').read_text(encoding='utf-8')
    (out/'metrics_by_horizon.csv').write_text(text.replace(',1.0,1.0,',',999.0,1.0,'),encoding='utf-8')
    _refresh_output_hash(out,'metrics_by_horizon.csv')
    with pytest.raises(mod.FrozenSimpleError,match='differs from raw pooled cells'):
        mod.verify_result(out,expected_result=result,expected_rows=rows,expected_pairs=[[0,3],[0,4]],
          method='persistence',horizons=horizons,task=task)


def test_verifier_rejects_wrong_scope_and_saved_mask_dtype(tmp_path):
    out=tmp_path/'scope'; result,rows,task,horizons=_write_verifiable_result(out)
    manifest=json.loads((out/'evaluation_manifest.json').read_text(encoding='utf-8'))
    manifest['split']='validation'
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(mod.FrozenSimpleError,match='manifest scope'):
        mod.verify_result(out,expected_result=result,expected_rows=rows,expected_pairs=[[0,3],[0,4]],
          method='persistence',horizons=horizons,task=task)

    out=tmp_path/'mask'; result,rows,task,horizons=_write_verifiable_result(out)
    with np.load(out/'predictions.npz',allow_pickle=False) as z:
        members={k:np.asarray(z[k]) for k in z.files}
    members['target_mask']=members['target_mask'].astype(np.int8)
    np.savez_compressed(out/'predictions.npz',**members)
    _refresh_output_hash(out,'predictions.npz')
    with pytest.raises(mod.FrozenSimpleError,match='target_mask'):
        mod.verify_result(out,expected_result=result,expected_rows=rows,expected_pairs=[[0,3],[0,4]],
          method='persistence',horizons=horizons,task=task)


def test_c06_descriptor_split_tamper_rejected_after_rehash(tmp_path,monkeypatch):
    run,data,fp,ap,cp,state=_write_authorized_fixture(tmp_path,None)
    auth=mod._json(ap); descriptor_path=run/'descriptor.json'
    descriptor=mod._json(descriptor_path); descriptor['split']='validation'
    descriptor_path.write_text(json.dumps(descriptor),encoding='utf-8')
    auth['domain_descriptor_sha256']=mod.sha256_file(descriptor_path)
    with pytest.raises(mod.FrozenSimpleError,match='schema/split mismatch'):
        mod._c06_domain_bundle(run,data,auth)


def test_duplicate_json_keys_and_invalid_or_duplicate_stage_refs_rejected(tmp_path):
    dup=tmp_path/'dup.json'; dup.write_text('{"x":1,"x":2}',encoding='utf-8')
    with pytest.raises(mod.FrozenSimpleError,match='duplicate JSON key'):
        mod._json(dup)
    run=tmp_path/'run';run.mkdir(); item=run/'item.json';item.write_text('{}',encoding='utf-8')
    digest=mod.sha256_file(item)
    state={'stages':[{'id':'P2','status':'passed','inputs':[
        {'path':'item.json','sha256':digest},{'path':'item.json','sha256':digest}],'evidence':[]}]}
    with pytest.raises(mod.FrozenSimpleError,match='duplicate passed-stage path'):
        mod._passed_refs(run,state)
    state['stages'][0]['inputs']=[{'path':'../outside.json','sha256':digest}]
    with pytest.raises(mod.FrozenSimpleError,match='escapes RUN'):
        mod._passed_refs(run,state)


def test_validator_rejection_precedes_test_loader(tmp_path,monkeypatch):
    run,data,fp,ap,cp,state=_write_authorized_fixture(tmp_path,None)
    guard=_owned_guard(run,run/'test-output',monkeypatch)
    calls=[]
    def reject(_root): raise mod.FrozenSimpleError('mock official gate rejected')
    with pytest.raises(mod.FrozenSimpleError,match='mock official gate rejected'):
        mod.frozen_test_evaluate(run,data,fp,ap,'linear',run/'test-output',checkpoint=guard.checkpoint,
          loader=lambda *_:(calls.append('test loader') or [ToyDS()]),validator=reject)
    assert calls==[]
    assert not (run/'test-output').exists()
    guard.release()
