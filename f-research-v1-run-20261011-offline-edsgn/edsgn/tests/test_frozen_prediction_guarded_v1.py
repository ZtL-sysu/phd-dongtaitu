import copy, hashlib, importlib.util, json, shutil, sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pytest
import torch

RUN=Path(r"C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn")
sys.path[:0]=[str(RUN/"edsgn/src"),str(RUN/"edsgn/scripts"),str(RUN/"edsgn/staging")]
from edsgn import engine

SPEC=importlib.util.spec_from_file_location("frozen_prediction_guarded_v1",RUN/"edsgn/staging/frozen_prediction_guarded_v1.py")
bridge=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(bridge)


def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def _write_npz(path):
    T,N,F=16,3,1; raw=np.arange(T*N*F,dtype=np.float32).reshape(T,N,F)/10+1
    meta={"partition":"chronological","count_data":False,"event_kind":"synthetic","history":2,"hmax":2,"horizons":[1,2],"splits":{"train":[0,8],"validation":[8,12],"test":[12,16]},"metadata":{"n_bins_total":T,"seed":9}}
    np.savez_compressed(path,metadata_json=np.asarray(json.dumps(meta)),values=raw.copy(),raw_values=raw,targets=raw.copy(),input_mask=np.ones_like(raw,bool),target_mask=np.ones_like(raw,bool),obs_age=np.zeros_like(raw),interval_end=np.arange(T,dtype=np.float64)+1,node_ids=np.arange(N),mean=np.zeros(F,np.float32),std=np.ones(F,np.float32),scale=np.ones((N,F),np.float32),events=np.empty((0,3),np.int64))

def _fixture(tmp_path,monkeypatch,kind="full"):
    root=tmp_path.resolve(); (root/"edsgn/src/edsgn").mkdir(parents=True);(root/"edsgn/scripts").mkdir(parents=True);(root/"edsgn/staging").mkdir(parents=True)
    for p in (RUN/"edsgn/src/edsgn").glob("*.py"):shutil.copy2(p,root/"edsgn/src/edsgn"/p.name)
    for name in ("bind_training_inputs.py","frozen_synthetic_stress.py","aggregate_budget.py","candidate_gpu_fullpath_probe_v3.py","baseline_campaign.py"):
        src=(RUN/"edsgn/scripts"/name)
        if src.exists():shutil.copy2(src,root/"edsgn/scripts"/name)
    for name in ("blind_reference_cost_policy_v1.py","paired_statistics_shared_v4.py"):
        shutil.copy2(RUN/"edsgn/staging"/name,root/"edsgn/staging"/name)
    data=root/"data/main.npz";data.parent.mkdir();_write_npz(data)
    variant="V4"; config={"dataset_path":str(data),"variant":variant,"seed":0,"device":"cpu","threads":1,"model":{"d_total":8,"k":2,"candidate_cap":2,"encoder_width":4,"readout_hidden":4}}
    cfg=root/"config.json";cfg.write_text(json.dumps(config),encoding="utf8")
    ds=engine.datasets_at(data,"train")[0]; stats=engine.task_statistics(ds)
    model=engine.build_model(config,ds,stats)
    payload={"config":config,"task":{"n_nodes":3,"n_features":1,"history":2,"hmax":2,"count_data":False},"statistics":stats,"state_dict":model.state_dict(),"train_activity":np.ones(3).tolist(),"provenance":{"scope":"full_protocol","source_sha256":engine.source_manifest(),"dependency_source_sha256":engine.dependency_manifest(variant),"dataset_path":str(data)}}
    ckpt=root/"checkpoint/checkpoint.pt";ckpt.parent.mkdir();torch.save(payload,ckpt)
    costs={"graph":[[1,.1],[3,.3]],"state":[[1,.1],[3,.3]],"base_ms":.01,"batch_size":1,"units":"toy measured envelope"}
    policies={"main": (engine.Policy() if kind=="full" else engine.Policy(kind=kind,graph_fraction=.5,state_fraction=.5,max_graph_age=5,max_state_age=5,audit_interval=0,force_initialization=False,cost_table=costs,runtime_budget_ms=.5))}
    extra={}
    if kind in {"periodic","random"}:
        hs=importlib.util.spec_from_file_location("blind_temp",root/"edsgn/staging/blind_reference_cost_policy_v1.py");blind=importlib.util.module_from_spec(hs);hs.loader.exec_module(blind);extra=blind.freeze_extra_fields()
    evidence=root/"val-selection.json";evidence.write_text('{"validation":true}',encoding="utf8")
    engine.freeze_checkpoint(ckpt,policies,[evidence],block_length=2,epsilon=.1,extra=extra)
    binding_dir=root/"binding";binding_dir.mkdir()
    bs=importlib.util.spec_from_file_location("bind_training_inputs_setup_toy",root/"edsgn/scripts/bind_training_inputs.py")
    binder=importlib.util.module_from_spec(bs);bs.loader.exec_module(binder);binder.ROOT=root
    binding=binder.prepare_binding(cfg,output_dir=binding_dir)
    p3in=root/"gate-input.json";p3in.write_text("input",encoding="utf8");p3ev=root/"gate-evidence.json";p3ev.write_text("evidence",encoding="utf8")
    stages=[{"id":"P2","status":"passed","gate_verdict":"pass"},{"id":"P3","status":"passed","gate_verdict":"pass"},{"id":"P3A","status":"passed","gate_verdict":"pass","inputs":[{"path":str(p3in.relative_to(root)),"sha256":_sha(p3in)}],"evidence":[{"path":str(p3ev.relative_to(root)),"sha256":_sha(p3ev)}]},{"id":"P4","status":"running"}]
    (root/"pipeline-state.json").write_text(json.dumps({"stages":stages}),encoding="utf8")
    monkeypatch.setattr(bridge,"ROOT",root)
    def explicit_root_binding(path, run_root):
        spec=importlib.util.spec_from_file_location("bind_training_inputs_prediction_toy",root/"edsgn/scripts/bind_training_inputs.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=root
        module.verify_binding(path)
        return module
    monkeypatch.setattr(bridge,"_verify_binding",explicit_root_binding)
    policy_cfg=None
    if kind!='full':
        policy_cfg=root/'policy.json'; policy_cfg.write_text(json.dumps(asdict(policies['main'])),encoding='utf8')
    return root,data,ckpt,cfg,binding,policies['main'],payload,policy_cfg


def _budget(root,output,receipt,doc,started): pass


def test_real_engine_full_cpu_prediction_exports_paired_stats_schema(tmp_path,monkeypatch):
    root,data,ckpt,cfg,binding,policy,payload,policy_cfg=_fixture(tmp_path,monkeypatch,"full")
    calls=[]
    result=bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=root/"runs/predict",split="test",policy_kind="full",device="cpu",root=root,budget_fn=lambda *a:calls.append("guard"),process_scan=lambda:[])
    assert result["pairs"]==[(0,11),(0,12),(0,13)]
    receipt=json.loads(next((root/"logs/offline-phase-wall").glob("frozen-prediction-*.json")).read_text())
    assert receipt["phase"]=="evaluation" and receipt["status"]=="completed" and receipt["test_accessed"] is True
    assert "--split" in receipt["command"] and receipt["command"][receipt["command"].index("--split")+1]=="test"
    assert receipt["command"][receipt["command"].index("--checkpoint")+1]==str(ckpt)
    assert len(receipt["outputs"])>=4 and calls.count("guard")>=4
    # Contract expected by paired_statistics_shared_v4._validate_test_artifacts.
    outputs={Path(x["path"]).resolve():x["sha256"] for x in receipt["outputs"]}
    for name in ("evaluation_manifest.json","predictions.npz"):
        path=(root/"runs/predict"/name).resolve();assert outputs[path]==_sha(path)
    with np.load(root/"runs/predict/predictions.npz",allow_pickle=False) as z:
        assert z["origins"].tolist()==[[0,11],[0,12],[0,13]] and z["seed_id"].item()==0


def test_periodic_without_bound_blind_helper_is_rejected(tmp_path,monkeypatch):
    root,data,ckpt,cfg,binding,policy,payload,policy_cfg=_fixture(tmp_path,monkeypatch,"periodic")
    freeze_path=ckpt.parent/"test-freeze.json";freeze=json.loads(freeze_path.read_text());freeze["extra"]={};freeze_path.write_text(json.dumps(freeze),encoding="utf8")
    with pytest.raises(ValueError,match="source hash"):
        bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=root/"runs/predict",split="test",policy_kind="periodic",policy_config_path=policy_cfg,device="cpu",root=root,budget_fn=_budget,process_scan=lambda:[])


@pytest.mark.parametrize("kind",["periodic","random"])
def test_blind_controls_apply_frozen_helper_and_complete_full_prediction(kind,tmp_path,monkeypatch):
    root,data,ckpt,cfg,binding,policy,payload,policy_cfg=_fixture(tmp_path,monkeypatch,kind)
    result=bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=root/"runs/predict",split="test",policy_kind=kind,policy_config_path=policy_cfg,device="cpu",root=root,budget_fn=_budget,process_scan=lambda:[])
    assert len(result["pairs"])==3
    receipt=json.loads(next((root/"logs/offline-phase-wall").glob("frozen-prediction-*.json")).read_text())
    assert receipt["status"]=="completed" and receipt["policy_source"]


def test_final_pipeline_change_after_serialization_leaves_failed_receipt(tmp_path,monkeypatch):
    root,data,ckpt,cfg,binding,policy,payload,policy_cfg=_fixture(tmp_path,monkeypatch,"full")
    original=engine.evaluate_run
    def mutate(*a,**kw):
        result=original(*a,**kw)
        p=root/"pipeline-state.json";doc=json.loads(p.read_text());doc["stages"][0]["gate_verdict"]="stale";p.write_text(json.dumps(doc))
        return result
    monkeypatch.setattr(engine,"evaluate_run",mutate)
    out=root/"runs/predict"
    with pytest.raises(RuntimeError,match="passed/pass"):
        bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=out,split="test",policy_kind="full",device="cpu",root=root,budget_fn=_budget,process_scan=lambda:[])
    assert (out/"predictions.npz").is_file()
    receipt=json.loads(next((root/"logs/offline-phase-wall").glob("frozen-prediction-*.json")).read_text())
    assert receipt["status"]=="failed"


def test_final_budget_failure_preserves_artifacts_and_failed_receipt(tmp_path,monkeypatch):
    root,data,ckpt,cfg,binding,policy,payload,policy_cfg=_fixture(tmp_path,monkeypatch,"full")
    state={"n":0}
    def budget(*args):
        state["n"]+=1
        if state["n"]==5: raise RuntimeError("toy final budget exhausted")
    out=root/"runs/predict"
    with pytest.raises(RuntimeError,match="budget exhausted"):
        bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=out,split="test",policy_kind="full",device="cpu",root=root,budget_fn=budget,process_scan=lambda:[])
    assert (out/"predictions.npz").is_file() and (out/"evaluation_manifest.json").is_file()
    receipt=json.loads(next((root/"logs/offline-phase-wall").glob("frozen-prediction-*.json")).read_text())
    assert receipt["status"]=="failed" and "budget exhausted" in receipt["error"]







