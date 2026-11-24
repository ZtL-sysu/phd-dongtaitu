import importlib.util, json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("v4_writer_tests_for_independent_review",ROOT/"edsgn/tests/test_frozen_prediction_guarded_v4.py")
writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
bridge=writer.bridge;engine=writer.engine

def test_access_is_persisted_before_test_loader_failure(tmp_path,monkeypatch):
    root,data,ckpt,cfg,binding,policy,payload,policy_cfg=writer._fixture(tmp_path,monkeypatch,"full")
    def fail_loader(*args,**kwargs): raise RuntimeError("toy loader failure")
    monkeypatch.setattr(bridge,"_frozen_normalized_collections",fail_loader)
    out=root/"runs/access-before-loader"
    with pytest.raises(RuntimeError,match="toy loader failure"):
        bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=out,split="test",policy_kind="full",device="cpu",root=root,budget_fn=writer._budget,process_scan=lambda:[])
    receipt=json.loads(next((root/"logs/offline-phase-wall").glob("frozen-prediction-*.json")).read_text())
    assert receipt["status"]=="failed" and receipt["test_accessed"] is True
    assert receipt["test_materialization_started"] is True and receipt["test_numeric_materialized"] is False
    assert receipt["test_origins_evaluated"] is False

def test_serialized_engine_arrays_must_match_actual_evaluator_contents(tmp_path,monkeypatch):
    root,data,ckpt,cfg,binding,policy,payload,policy_cfg=writer._fixture(tmp_path,monkeypatch,"full")
    original=engine.evaluate_run
    def mutate_export(*args,**kwargs):
        result=original(*args,**kwargs)
        path=root/"runs/array-integrity/predictions.npz"
        with __import__("numpy").load(path,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        arrays["targets"][0,0,0,0]+=3
        __import__("numpy").savez_compressed(path,**arrays)
        return result
    monkeypatch.setattr(engine,"evaluate_run",mutate_export)
    out=root/"runs/array-integrity"
    with pytest.raises(RuntimeError,match="Serialized targets differs"):
        bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=out,split="test",policy_kind="full",device="cpu",root=root,budget_fn=writer._budget,process_scan=lambda:[])
    receipt=json.loads(next((root/"logs/offline-phase-wall").glob("frozen-prediction-*.json")).read_text())
    assert receipt["status"]=="failed" and receipt["test_accessed"] is True

def test_root_acceptance_requires_false_science_and_pipeline_flags(tmp_path,monkeypatch):
    root,data,ckpt,cfg,binding,policy,payload,policy_cfg=writer._fixture(tmp_path,monkeypatch,"periodic")
    acceptance=root/"reviews/blind-reference-cost-policy-root-acceptance-v1.json"
    doc=json.loads(acceptance.read_text());doc["scientific_hypothesis_accepted"]=True
    acceptance.write_text(json.dumps(doc),encoding="utf8")
    with pytest.raises(RuntimeError,match="root engineering acceptance"):
        bridge._validate_blind_acceptance(root,root/"edsgn/staging/blind_reference_cost_policy_v1.py")

def test_budget_is_not_rechecked_after_post_guard_output_hashing(tmp_path,monkeypatch):
    root,data,ckpt,cfg,binding,policy,payload,policy_cfg=writer._fixture(tmp_path,monkeypatch,"full")
    state={"expired":False,"final_guard_seen":False}
    def guard(*args):
        receipt=args[3]
        if receipt.get("outputs"):
            state["final_guard_seen"]=True
            state["expired"]=True
        elif state["expired"]:
            raise RuntimeError("mock budget expired")
    out=root/"runs/final-budget-order"
    result=bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=out,split="test",policy_kind="full",device="cpu",root=root,budget_fn=guard,process_scan=lambda:[])
    receipt=json.loads(next((root/"logs/offline-phase-wall").glob("frozen-prediction-*.json")).read_text())
    assert result["pairs"] and state["final_guard_seen"] and state["expired"]
    assert receipt["status"]=="completed"
