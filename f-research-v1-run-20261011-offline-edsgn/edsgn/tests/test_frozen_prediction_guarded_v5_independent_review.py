import importlib.util, json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("v5_writer_tests_for_independent_review",ROOT/"edsgn/tests/test_frozen_prediction_guarded_v5.py")
writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
bridge=writer.bridge

def test_budget_expiring_after_output_rehash_is_rejected_before_completed(tmp_path,monkeypatch):
    root,data,ckpt,cfg,binding,policy,payload,policy_cfg=writer._fixture(tmp_path,monkeypatch,"full")
    state={"final_checks":0}
    def budget(root_arg,output,receipt_path,doc,started):
        if doc.get("outputs"):
            state["final_checks"]+=1
            if state["final_checks"]==2:
                raise RuntimeError("toy budget expired after final output fence")
    out=root/"runs/v5-budget-order"
    with pytest.raises(RuntimeError,match="expired after final output fence"):
        bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=out,split="test",policy_kind="full",device="cpu",root=root,budget_fn=budget,process_scan=lambda:[])
    receipt=json.loads(next((root/"logs/offline-phase-wall").glob("frozen-prediction-*.json")).read_text())
    assert state["final_checks"]==2 and receipt["status"]=="failed"
    assert receipt["test_accessed"] is True and (out/"predictions.npz").is_file()
