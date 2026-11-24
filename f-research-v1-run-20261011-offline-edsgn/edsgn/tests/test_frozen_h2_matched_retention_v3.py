import importlib, json, shutil, sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'edsgn'/'staging'));sys.path.insert(0,str(ROOT/'edsgn'/'src'))
import frozen_h2_matched_retention_v3 as h2
from edsgn import engine
from edsgn.data import PreparedData
from edsgn.model import ForecastModel
from edsgn.scheduling import Policy


def make_npz(path, seed, split='validation'):
    rng=np.random.default_rng(seed);t,n,f=13,5,2
    raw=rng.poisson(1.2,(t,n,f)).astype(np.float32); mask=np.ones((t,n,f),bool);mask[3,1,0]=False
    vals=(np.log1p(raw)-np.array([.4,.5],np.float32))/np.array([.7,.8],np.float32);vals=np.where(mask,vals,0).astype(np.float32)
    metadata={'splits':{'train':[0,8],'validation':[9,13],'test':[9,13]},'count_data':True,'horizons':[1,3],
      'history':6,'hmax':3,'event_kind':'email','partition':'chronological','metadata':{'seed':seed}}
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,metadata_json=np.asarray(json.dumps(metadata)),values=vals,targets=raw,input_mask=mask,
      target_mask=mask,obs_age=np.zeros_like(vals),interval_end=np.arange(t,dtype=np.float64),node_ids=np.arange(n),
      mean=np.asarray([.4,.5],np.float32),std=np.asarray([.7,.8],np.float32),scale=np.ones((n,f),np.float32),
      raw_values=raw,events=np.empty((0,3),np.int64))
    return path


def make_temp_run(tmp,monkeypatch):
    root=tmp/'run'; (root/'edsgn/staging').mkdir(parents=True);(root/'edsgn/scripts').mkdir(parents=True);(root/'edsgn/src/edsgn').mkdir(parents=True)
    (root/'logs').mkdir(); (root/'checkpoints').mkdir()
    shutil.copy2(h2.KERNEL,root/'edsgn/staging/h2_matched_graph_retention_v2.py')
    shutil.copy2(Path(h2.__file__),root/'edsgn/staging/frozen_h2_matched_retention_v3.py')
    for rel in ('edsgn/scripts/aggregate_budget.py','edsgn/scripts/candidate_gpu_fullpath_probe_v3.py','edsgn/scripts/frozen_synthetic_stress.py','edsgn/src/edsgn/prepare.py'):
      shutil.copy2(ROOT/rel,root/rel)
    ev=root/'evidence.json';ev.write_text('{"accepted":true}',encoding='utf-8')
    state={'stages':[{'id':'P2','status':'passed','gate_verdict':'pass'},{'id':'P3','status':'passed','gate_verdict':'pass'},
      {'id':'P3A','status':'passed','gate_verdict':'pass','evidence':[{'path':str(ev),'sha256':h2.sha256(ev)}]}, {'id':'P4','status':'running'}]}
    (root/'pipeline-state.json').write_text(json.dumps(state),encoding='utf-8')
    main=make_npz(root/'data/main.npz',11); test=make_npz(root/'data/stress-test.npz',19)
    monkeypatch.setattr(h2,'ROOT',root);monkeypatch.setattr(h2,'KERNEL',root/'edsgn/staging/h2_matched_graph_retention_v2.py')
    monkeypatch.setattr(h2,'__file__',str(root/'edsgn/staging/frozen_h2_matched_retention_v3.py'))
    ds=engine.datasets_at(main,'train')[0]
    cfg={'dataset_path':str(main),'variant':'V4','seed':7,'device':'cpu','threads':1,
      'model':{'d_total':16,'encoder_width':8,'readout_hidden':8,'k':2,'candidate_cap':4,'retention_mode':'informative'}}
    stats={'norm_mean':[.4,.5],'norm_std':[.7,.8],'residual_scale':np.ones((5,2),np.float32).tolist(),
      'output_scale':[1.,1.],'output_mean':[1.,1.]}
    model=engine.build_model(cfg,ds,stats).eval()
    payload={'config':cfg,'task':{'n_nodes':5,'n_features':2,'history':6,'hmax':3,'count_data':True},
      'statistics':stats,'state_dict':model.state_dict(),'provenance':{'scope':'full_protocol','source_sha256':engine.source_manifest(),
      'dependency_source_sha256':{},'dataset_path':str(main)}}
    checkpoint=root/'checkpoints/checkpoint.pt';torch.save(payload,checkpoint)
    return root,main,test,checkpoint,payload


def test_engine_checkpoint_schema_uses_task_and_prepareddata_shape():
    p={'config':{'dataset_path':'d.npz','variant':'V4','seed':0},'task':{'n_nodes':5,'n_features':2,'history':6,'hmax':3,'count_data':True}}
    assert h2.validate_checkpoint_payload(p)==p['task']
    bad=json.loads(json.dumps(p));bad['config']['history']=6
    with pytest.raises(RuntimeError,match='fake config'):
      h2.validate_checkpoint_payload(bad)
    assert not hasattr(PreparedData,'n_nodes') and not hasattr(PreparedData,'n_features')

def test_metadata_and_byte_collection_authorization_does_not_rebind_main_data(tmp_path,monkeypatch):
    root,main,test,checkpoint,payload=make_temp_run(tmp_path,monkeypatch)
    allowed=h2.h2_freeze_collection_authorization([main,test],engine,root)
    assert set(allowed)=={str(main.resolve()),str(test.resolve())}
    assert allowed[str(main.resolve())]['collection_sha256']!=allowed[str(test.resolve())]['collection_sha256']
    extra=h2.h2_freeze_extras([test],engine,'V4',root)
    top={'dataset_fingerprint':engine.dataset_fingerprint(main)}
    freeze={**top,'checkpoint_sha256':h2.sha256(checkpoint),'parameters_sha256':engine.parameter_hash(engine.load_checkpoint(checkpoint,'cpu')[0]),'extra':extra}
    _,dep,_,_=h2._canonical_source_fingerprints(root,engine,'V4')
    h2.validate_test_freeze(freeze,checkpoint_sha=freeze['checkpoint_sha256'],parameter_sha=freeze['parameters_sha256'],
      collection_path=test,dataset_fingerprint=engine.dataset_fingerprint(test),collection_sha=h2.collection_sha256(test),
      driver_sha=extra['h2_matched_retention_driver_sha256'],kernel_sha=extra['h2_matched_retention_kernel_sha256'],
      source_sha=extra['h2_matched_retention_source_manifest_sha256'],dependency_sha=dep)
    assert freeze['dataset_fingerprint']==engine.dataset_fingerprint(main)
    bad=json.loads(json.dumps(freeze));bad['extra']['h2_matched_retention_allowed_collections'].pop(str(test.resolve()))
    with pytest.raises(RuntimeError,match='exact metadata-and-byte'):
      h2.validate_test_freeze(bad,checkpoint_sha=freeze['checkpoint_sha256'],parameter_sha=freeze['parameters_sha256'],
       collection_path=test,dataset_fingerprint=engine.dataset_fingerprint(test),collection_sha=h2.collection_sha256(test),
       driver_sha=extra['h2_matched_retention_driver_sha256'],kernel_sha=extra['h2_matched_retention_kernel_sha256'],source_sha=extra['h2_matched_retention_source_manifest_sha256'],dependency_sha=dep)

def test_actual_engine_freeze_and_toy_prepareddata_two_origin_full_kernel_replay(tmp_path,monkeypatch):
    root,main,test,checkpoint,payload=make_temp_run(tmp_path,monkeypatch)
    policy=Policy(kind='full');extra=h2.h2_freeze_extras([test],engine,'V4',root)
    evidence=root/'selection.json';evidence.write_text('{"selected_from":"validation"}',encoding='utf-8')
    freeze=engine.freeze_checkpoint(checkpoint,{'full':policy},[evidence],block_length=6,epsilon=0.1,extra=extra)
    assert freeze['dataset_fingerprint']==engine.dataset_fingerprint(main)
    model,loaded=engine.load_checkpoint(checkpoint,'cpu')
    assert 'history' not in loaded['config'] and loaded['task']['history']==6
    prepared=engine.datasets_at(test,'test')
    assert isinstance(prepared[0],PreparedData) and not hasattr(prepared[0],'n_nodes') and not hasattr(prepared[0],'n_features')
    assert tuple(prepared[0].values.shape[1:])==(5,2)
    report=h2.run_checkpoint_collection(checkpoint,test,'test',root/'outputs/h2','cpu',_root=root)
    assert report['origin_count']==2 and report['history']==6 and report['hmax']==3 and report['batch_size']==1
    assert report['model_seed_id']==7 and report['trajectory_seed_ids']==[19,19]
    assert report['informative_teacher_parity_all_windows'] and report['targets_passed_to_kernel'] is False
    assert report['oracle_accessed'] is False and report['artifact_disk_upper_bound_bytes']>0
    for name in ('teacher',*h2.ARMS):
      z=np.load(root/'outputs/h2'/f'predictions-{name}.npz',allow_pickle=False)
      assert z['predictions'].shape==(2,3,5,2) and z['origins'].tolist()==[[0,8],[0,9]]
      assert z['seed_id'].item()==7 and z['trajectory_seed_id'].tolist()==[19,19]
    assert not list((root/'outputs/h2').glob('.*.work.npy'))

def test_canonical_pipeline_tamper_and_output_immutability_fail_closed(tmp_path,monkeypatch):
    root,main,test,checkpoint,payload=make_temp_run(tmp_path,monkeypatch)
    # Existing output is rejected before artifacts can be touched.
    out=root/'outputs/existing';out.mkdir(parents=True);sentinel=out/'keep.txt';sentinel.write_text('locked')
    with pytest.raises(FileExistsError):h2.run_checkpoint_collection(checkpoint,test,'test',out,'cpu',_root=root)
    assert sentinel.read_text()=='locked'
    # A failed canonical gate cannot be replaced by private fixture status.
    pipe=root/'pipeline-state.json';state=json.loads(pipe.read_text());state['stages'][1]['status']='running';pipe.write_text(json.dumps(state))
    with pytest.raises(RuntimeError,match='P2/P3'):
      h2.run_checkpoint_collection(checkpoint,test,'test',root/'outputs/blocked','cpu',_root=root)

def test_both_locks_budget_and_disk_boundaries(tmp_path):
    good=lambda root,exclude_active_output=None:{'coverage_complete':True,'spent_seconds':3,'budget_seconds':100}
    out=tmp_path/'o'
    for rel in ('logs/model-resource.lock','logs/offline-phase-wall/campaign.lock'):
      p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
      with pytest.raises(RuntimeError,match='lock'):h2.resource_preflight(tmp_path,out,good,lambda root:[])
      p.unlink()
    with pytest.raises(RuntimeError,match='active'):h2.resource_preflight(tmp_path,out,good,lambda root:['python edsgn.train'])
    with pytest.raises(RuntimeError,match='exhausted'):
      h2.resource_preflight(tmp_path,out,lambda *a,**k:{'coverage_complete':True,'spent_seconds':100,'budget_seconds':100},lambda root:[])
    assert h2.estimate_artifact_bytes(2,3,5,2)>0

def test_pipeline_toc_tou_after_kernel_is_rejected_and_failed_output_retained(tmp_path,monkeypatch):
    root,main,test,checkpoint,payload=make_temp_run(tmp_path,monkeypatch)
    extra=h2.h2_freeze_extras([test],engine,'V4',root);policy=Policy(kind='full')
    evidence=root/'selection.json';evidence.write_text('{"selected_from":"validation"}',encoding='utf-8')
    engine.freeze_checkpoint(checkpoint,{'full':policy},[evidence],6,.1,extra=extra)
    model,loaded=engine.load_checkpoint(checkpoint,'cpu');kernel_path=root/'edsgn/staging/h2_matched_graph_retention_v2.py'
    import importlib.util
    spec=importlib.util.spec_from_file_location('h2_mutation_test_kernel',kernel_path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    changed={'once':False}
    def mutate_after_kernel(*args,**kwargs):
      result=mod.run_matched_retention_probe(*args,**kwargs)
      if not changed['once']:
        pipe=root/'pipeline-state.json';pipe.write_bytes(pipe.read_bytes()+b' ');changed['once']=True
      return result
    out=root/'outputs/toc-failure'
    with pytest.raises(RuntimeError,match='pipeline bytes changed'):
      h2.run_checkpoint_collection(checkpoint,test,'test',out,'cpu',_root=root,
        _services={'engine':engine,'model':model,'payload':loaded,'kernel':mutate_after_kernel})
    assert changed['once'] and out.exists() and not (out/'h2-matched-retention-report.json').exists()

def test_required_normalization_sources_fail_closed(tmp_path,monkeypatch):
    root,main,test,checkpoint,payload=make_temp_run(tmp_path,monkeypatch)
    (root/'edsgn/src/edsgn/prepare.py').unlink()
    with pytest.raises(RuntimeError,match='normalization source missing'):
      h2._source_hashes(root,engine)


def test_serialization_budget_exhaustion_never_marks_completed(tmp_path, monkeypatch):
    import frozen_h2_matched_retention_v3 as h2
    report=tmp_path/"final-report.json"; calls={"n":0}
    def guard():
      calls["n"]+=1
      if calls["n"]==2: raise RuntimeError("phase exceeded remaining cumulative budget")
    def final_checks(): assert report.is_file()
    with pytest.raises(RuntimeError,match="budget"):
      h2._write_report_with_final_guard(report,{"ok":True},guard,final_checks)
    assert report.is_file() and json.loads(report.read_text())["ok"] is True
    # Outer receipt completion is ordered after the final guard in run_with_guards.
    source=(Path(h2.__file__).read_text(encoding="utf-8"))
    assert source.index("        guard()\n        report_path=") < source.index('receipt.update(status="completed"')


def test_outer_final_budget_failure_keeps_failed_receipt_after_serialization(tmp_path,monkeypatch):
    import frozen_h2_matched_retention_v3 as h2
    root,main,test,checkpoint,payload=make_temp_run(tmp_path,monkeypatch)
    (root/'logs/offline-phase-wall').mkdir(parents=True,exist_ok=True)
    calls={'n':0}
    def aggregate(*args,**kwargs):
      calls['n']+=1
      return {'coverage_complete':True,'spent_seconds':0 if calls['n']==1 else 100,'budget_seconds':100}
    def fake_run(*args,**kwargs):
      out=Path(args[3]);out.mkdir(parents=True,exist_ok=True)
      (out/'h2-matched-retention-report.json').write_text('{"serialized":true}',encoding='utf8')
      return {'origin_count':2}
    monkeypatch.setattr(h2,'run_checkpoint_collection',fake_run)
    out=root/'outputs/final-guard-failure'
    with pytest.raises(RuntimeError,match='budget exhausted'):
      h2.run_with_guards(checkpoint,test,'test',out,'cpu',_root=root,
          _services={'state':json.loads((root/'pipeline-state.json').read_text()),'aggregate':aggregate,'owned':lambda root:[]})
    receipts=list((root/'logs/offline-phase-wall').glob('h2-matched-retention-*.json'))
    assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text())
    assert receipt['status']=='failed' and (out/'h2-matched-retention-report.json').is_file()


def test_v3_source_descriptor_names_v3_and_keeps_v2_kernel(tmp_path,monkeypatch):
    import frozen_h2_matched_retention_v3 as h2
    root,main,test,checkpoint,payload=make_temp_run(tmp_path,monkeypatch)
    mapping=h2._source_hashes(root,engine)
    assert mapping['edsgn/staging/frozen_h2_matched_retention_v3.py']==h2.sha256(Path(h2.__file__))
    assert mapping['edsgn/staging/h2_matched_graph_retention_v2.py']==h2.sha256(root/'edsgn/staging/h2_matched_graph_retention_v2.py')


