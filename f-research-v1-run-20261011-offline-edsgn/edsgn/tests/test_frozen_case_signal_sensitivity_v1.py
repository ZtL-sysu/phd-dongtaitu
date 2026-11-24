import json, shutil, sys
from pathlib import Path
import numpy as np
import pytest
import torch

RUN=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(RUN/'edsgn'/'staging'));sys.path.insert(0,str(RUN/'edsgn'/'src'))
import frozen_case_signal_sensitivity_v1 as case
from edsgn import engine
from edsgn.data import PreparedData
from edsgn.scheduling import Policy


def _write(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n",encoding="utf-8");return path

def _make_npz(path,seed):
    rng=np.random.default_rng(seed);t,n,f=13,5,2
    raw=rng.poisson(1.2,(t,n,f)).astype(np.float32);mask=np.ones((t,n,f),bool);mask[3,1,0]=False
    mean=np.asarray([.4,.5],np.float32);std=np.asarray([.7,.8],np.float32)
    vals=np.where(mask,(np.log1p(raw)-mean)/std,0).astype(np.float32)
    metadata={'splits':{'train':[0,8],'validation':[9,13],'test':[9,13]},'count_data':True,'horizons':[1,3],
      'history':6,'hmax':3,'event_kind':'email','partition':'chronological','metadata':{'seed':seed}}
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,metadata_json=np.asarray(json.dumps(metadata)),values=vals,targets=raw,input_mask=mask,
      target_mask=mask,obs_age=np.zeros_like(vals),interval_end=np.arange(t,dtype=np.float64),node_ids=np.arange(n),
      mean=mean,std=std,scale=np.ones((n,f),np.float32),raw_values=raw,events=np.empty((0,3),np.int64))
    return path

def _source_tree(root):
    (root/'edsgn/staging').mkdir(parents=True,exist_ok=True);(root/'edsgn/scripts').mkdir(parents=True,exist_ok=True);(root/'edsgn/src/edsgn').mkdir(parents=True,exist_ok=True)
    for p in (RUN/'edsgn/src/edsgn').glob('*.py'):shutil.copy2(p,root/'edsgn/src/edsgn'/p.name)
    for rel in ('edsgn/scripts/aggregate_budget.py','edsgn/scripts/candidate_gpu_fullpath_probe_v3.py',
                'edsgn/scripts/frozen_synthetic_stress.py','edsgn/src/edsgn/prepare.py',
                'edsgn/staging/case_signal_sensitivity_kernel_v2.py','edsgn/staging/h2_matched_graph_retention_v2.py',
                'reviews/case-signal-sensitivity-kernel-v2-root-engineering-v1.json'):
        target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(RUN/rel,target)
    shutil.copy2(Path(case.__file__),root/'edsgn/staging/frozen_case_signal_sensitivity_v1.py')

def make_temp_case_run(tmp_path,monkeypatch):
    root=tmp_path/'run';_source_tree(root);(root/'logs').mkdir(exist_ok=True);(root/'checkpoints').mkdir(exist_ok=True)
    ev=_write(root/'evidence.json',{'evidence':'temporary accepted gate fixture'})
    state={'stages':[{'id':'P2','status':'passed','gate_verdict':'pass'},{'id':'P3','status':'passed','gate_verdict':'pass'},
      {'id':'P3A','status':'passed','gate_verdict':'pass','evidence':[{'path':str(ev),'sha256':case.sha256(ev)}]},
      {'id':'P4','status':'running'}]}
    (root/'pipeline-state.json').write_text(json.dumps(state),encoding='utf-8')
    main=_make_npz(root/'data/email/email.npz',11);test=_make_npz(root/'collections/email/test.npz',19)
    cfg={'dataset_path':str(main.resolve()),'variant':'V4','seed':0,'device':'cpu','threads':1,
      'model':{'d_total':16,'encoder_width':8,'readout_hidden':8,'k':2,'candidate_cap':4,'retention_mode':'informative'}}
    ds=engine.datasets_at(main,'train')[0]
    stats={'norm_mean':[.4,.5],'norm_std':[.7,.8],'residual_scale':np.ones((5,2),np.float32).tolist(),
      'output_scale':[1.,1.],'output_mean':[1.,1.]}
    model=engine.build_model(cfg,ds,stats).eval()
    payload={'config':cfg,'task':{'n_nodes':5,'n_features':2,'history':6,'hmax':3,'count_data':True},
      'statistics':stats,'state_dict':model.state_dict(),'provenance':{'scope':'full_protocol','source_sha256':engine.source_manifest(),
      'dependency_source_sha256':{},'dataset_path':str(main.resolve())}}
    checkpoint=root/'checkpoints/checkpoint.pt';torch.save(payload,checkpoint)
    monkeypatch.setattr(case,'ROOT',root)
    monkeypatch.setattr(case,'KERNEL',root/'edsgn/staging/case_signal_sensitivity_kernel_v2.py')
    monkeypatch.setattr(case,'H2_KERNEL',root/'edsgn/staging/h2_matched_graph_retention_v2.py')
    monkeypatch.setattr(case,'KERNEL_ACCEPTANCE',root/'reviews/case-signal-sensitivity-kernel-v2-root-engineering-v1.json')
    monkeypatch.setattr(case,'__file__',str(root/'edsgn/staging/frozen_case_signal_sensitivity_v1.py'))
    source_sha=case._canonical_source_fingerprints(root,engine,'V4')[0]
    train_fp=engine.dataset_fingerprint(main);node_ids=ds.node_ids
    val_record=_write(root/'selection/validation-origin-record.json',{'scope':'case_signal_full_validation_origin_record',
      'split':'validation','origins_complete_and_ordered':True,'full_validation_origin_count':2,
      'checkpoint_sha256':case.sha256(checkpoint),'source_manifest_sha256':source_sha,
      'dataset_fingerprint':train_fp,'node_order_sha256':case.node_order_sha256(node_ids)})
    selection=_write(root/'selection/case-signal-selection.json',{'schema_version':1,
      'scope':case.CASE_SELECTION_SCOPE,'status':'accepted','accepted_by':'root','test_feedback_used':False,
      'dataset_id':'email','training_dataset_path':str(main.resolve()),'dataset_fingerprint':train_fp,
      'variant':'V4','model_seed':0,'split':'validation','checkpoint_sha256':case.sha256(checkpoint),
      'source_manifest_sha256':source_sha,'node_order_sha256':case.node_order_sha256(node_ids),
      'full_validation_origin_count':2,
      'validation_origin_record':{'path':val_record.relative_to(root).as_posix(),'sha256':case.sha256(val_record)},
      'selection_rule':{'focal':'largest_training_activity_score_then_node_id',
                        'neighbor':'largest_mean_valid_neighbor_weight_in_full_validation_then_node_id'},
      'focal_node':{'index':0,'node_id':0},'neighbor_node':{'index':1,'node_id':1}})
    extra=case.case_freeze_extras([test],engine,'V4',selection,root)
    engine.freeze_checkpoint(checkpoint,{'full':Policy(kind='full')},[selection,val_record],6,.1,extra=extra)
    model2,payload2=engine.load_checkpoint(checkpoint,'cpu')
    # Use the real engine-loader payload and Frozen model on the complete toy collection.
    return root,main,test,checkpoint,selection,model2,payload2,state

def test_real_temp_engine_freeze_full_origin_three_output_semantic_proof(tmp_path,monkeypatch):
    root,main,test,checkpoint,selection,model,payload,state=make_temp_case_run(tmp_path,monkeypatch)
    ds=engine.datasets_at(test,'test')[0]
    assert isinstance(ds,PreparedData) and not hasattr(ds,'n_nodes') and not hasattr(ds,'n_features')
    result=case.run_case_collection(checkpoint,test,selection,'email','test',root/'outputs/case','cpu',_root=root,
      _services={'toy_fixture':True,'engine':engine,'model':model,'payload':payload})
    out=root/'outputs/case';report=json.loads((out/'case-signal-sensitivity-report.json').read_text())
    closure=json.loads((out/'output-closure.json').read_text())
    assert report['scope']=='frozen_case_signal_sensitivity_diagnostic'
    assert report['origin_count']==2 and report['history']==6 and report['batch_size']==1
    assert report['informative_teacher_parity_all_windows'] and report['retention_graph_actions_exact_all_windows']
    assert report['relationship_candidate_cap_preserved_all_windows']
    assert report['targets_passed_to_kernel'] is False and report['oracle_accessed'] is False
    assert report['managed_global_dispatcher_integrated'] is False and report['real_campaign_launchable'] is False
    assert report['semantic_proof_lines']==report['semantic_proof_expected_lines']==12
    assert closure['closure_sha256']==case.canonical_sha(closure['files'])
    assert set(closure['files'])=={p.name for p in out.iterdir() if p.is_file() and p.name!='output-closure.json'}
    manifest=json.loads((out/'evaluation_manifest.json').read_text())
    assert manifest['files']['case-signal-semantic-proof.jsonl']==case.sha256(out/'case-signal-semantic-proof.jsonl')
    for name in ('teacher',*case.ARMS):
        with np.load(out/f'predictions-{name}.npz',allow_pickle=False) as z:
            assert z['predictions'].shape==(2,3,5,2) and z['origins'].shape==(2,2)
            assert z['seed_id'].item()==0
        metrics=list(csv for csv in (out/f'metrics-{name}-by-horizon-feature.csv').read_text().splitlines())
        assert len(metrics)==1+3*2
    proof=(out/'case-signal-semantic-proof.jsonl').read_text().splitlines()
    assert len(proof)==12 and all(json.loads(row)['candidate_cap_preserved'] for row in proof)
    assert not list(out.glob('.*.work.npy'))

def test_non_root_selection_or_wrong_node_identity_fails_before_test_windows(tmp_path,monkeypatch):
    root,main,test,checkpoint,selection,model,payload,state=make_temp_case_run(tmp_path,monkeypatch)
    doc=json.loads(selection.read_text());doc['accepted_by']='not-root';selection.write_text(json.dumps(doc),encoding='utf-8')
    with pytest.raises(RuntimeError,match='accepted validation-only'):
      case.run_case_collection(checkpoint,test,selection,'email','test',root/'outputs/reject','cpu',_root=root,
       _services={'toy_fixture':True,'engine':engine,'model':model,'payload':payload})
    assert not (root/'outputs/reject').exists()

def test_public_entrypoint_requires_managed_global_case_admission(tmp_path,monkeypatch):
    root,main,test,checkpoint,selection,model,payload,state=make_temp_case_run(tmp_path,monkeypatch)
    with pytest.raises(RuntimeError,match='global prospective case-job inventory'):
      case.run_with_guards(checkpoint,test,selection,'email','test',root/'outputs/no-authority','cpu',_root=root)
    assert not (root/'outputs/no-authority').exists()

def test_freeze_rejects_selection_hash_or_real_node_id_mismatch(tmp_path,monkeypatch):
    root,main,test,checkpoint,selection,model,payload,state=make_temp_case_run(tmp_path,monkeypatch)
    # The accepted JSON must still match the selection bytes authorized by the
    # checkpoint sibling freeze; changing the selected real ID is not a new arm.
    doc=json.loads(selection.read_text());doc['neighbor_node']['node_id']=4
    selection.write_text(json.dumps(doc),encoding='utf-8')
    with pytest.raises(RuntimeError,match='selection differs|ID differs|node order'):
      case.run_case_collection(checkpoint,test,selection,'email','test',root/'outputs/freeze-reject','cpu',_root=root,
       _services={'toy_fixture':True,'engine':engine,'model':model,'payload':payload})
    assert not (root/'outputs/freeze-reject').exists()

def test_guarded_toy_phase_writes_completion_and_releases_owned_locks(tmp_path,monkeypatch):
    root,main,test,checkpoint,selection,model,payload,state=make_temp_case_run(tmp_path,monkeypatch)
    _write(root/'logs/budget.json',{'toy':True})
    result=case.run_with_guards(checkpoint,test,selection,'email','test',root/'outputs/guarded','cpu',_root=root,
      _services={'toy_fixture':True,'engine':engine,'model':model,'payload':payload,'state':state,
        'aggregate':lambda *a,**k:{'coverage_complete':True,'spent_seconds':1,'budget_seconds':10000},
        'owned':lambda _root:[]})
    receipt=list((root/'logs/offline-phase-wall').glob('case-signal-sensitivity-*.json'))
    assert len(receipt)==1
    doc=json.loads(receipt[0].read_text())
    assert doc['status']=='completed' and doc['phase']=='evaluation' and doc['output_closure_sha256']
    assert result['origin_count']==2
    assert not (root/'logs/model-resource.lock').exists()
    assert not (root/'logs/offline-phase-wall/campaign.lock').exists()

def test_final_budget_flip_after_serialization_leaves_failed_not_completed_receipt(tmp_path,monkeypatch):
    root,main,test,checkpoint,selection,model,payload,state=make_temp_case_run(tmp_path,monkeypatch)
    calls={'n':0}
    def aggregate(*args,**kwargs):
        calls['n']+=1
        # Permit launch and replay checks; refuse at the outer post-serialization
        # guard immediately before the successful phase receipt.
        spent=100 if calls['n']>=7 else 1
        return {'coverage_complete':True,'spent_seconds':spent,'budget_seconds':100}
    with pytest.raises(RuntimeError,match='budget exhausted'):
      case.run_with_guards(checkpoint,test,selection,'email','test',root/'outputs/budget-failure','cpu',_root=root,
        _services={'toy_fixture':True,'engine':engine,'model':model,'payload':payload,'state':state,
          'aggregate':aggregate,'owned':lambda _root:[]})
    receipts=list((root/'logs/offline-phase-wall').glob('case-signal-sensitivity-*.json'))
    assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text())
    assert receipt['status']=='failed' and 'report_sha256' not in receipt
    assert (root/'outputs/budget-failure/case-signal-sensitivity-report.json').is_file()
    assert not (root/'logs/model-resource.lock').exists()
    assert not (root/'logs/offline-phase-wall/campaign.lock').exists()

