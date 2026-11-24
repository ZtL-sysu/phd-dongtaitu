"""Temporary CPU-only acceptance fixtures; no canonical arrays are loaded."""
from __future__ import annotations
import csv, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pytest

RUN=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(RUN/"edsgn"/"src"))
sys.path.insert(0,str(RUN/"edsgn"/"scripts"))
SPEC=importlib.util.spec_from_file_location("simple_fit_verify_v4",RUN/"edsgn"/"staging"/"simple_baseline_train_fit_verification_v4.py")
mod=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(mod)


def _toy():
    from edsgn.data import PreparedData
    T,N,F=12,2,1
    raw=np.arange(T*N*F,dtype=np.float32).reshape(T,N,F)/7+0.25
    targets=raw.copy(); mask=np.ones_like(raw,dtype=bool)
    # One node/origin target row is incomplete; it must not enter train OLS.
    mask[3,1,0]=False
    ds=PreparedData(values=raw.copy(),targets=targets,input_mask=np.ones_like(mask),target_mask=mask,
      obs_age=np.zeros_like(raw),interval_end=np.arange(T,dtype=float),node_ids=np.array([10,20],dtype=np.int64),
      splits={"train":(0,7),"validation":(7,11),"test":(11,12)},mean=np.array([0.]),std=np.array([1.]),
      scale=np.ones((N,F)),count_data=False,horizons=(1,2),history=2,hmax=2,metadata={"raw_values":raw.copy()})
    ds.calls=[]
    original=ds.target
    def spy(origin):
        ds.calls.append(int(origin)); return original(origin)
    ds.target=spy
    return ds


def _fixture(tmp_path,monkeypatch):
    from edsgn import engine
    ds=_toy(); run=tmp_path/"run";run.mkdir()
    data=run/"toy.npz";data.write_bytes(b"toy source fixture; data access is supplied by a temporary loader")
    legacy=run/"legacy";legacy.mkdir()
    def loader(path,split):
        assert split in {"train","validation","val"} # test access is an immediate fixture failure
        return [ds]
    monkeypatch.setattr(engine,"datasets_at",loader)
    engine.simple_baselines(data,legacy,split="val")
    state={"stages":[{"id":"P2","status":"passed","gate_verdict":"pass","inputs":[],"evidence":[]},
      {"id":"P3","status":"running","gate_verdict":"pending","inputs":[],"evidence":[]} ]}
    (run/"pipeline-state.json").write_text(json.dumps(state),encoding="utf-8")
    return run,data,legacy,ds


def test_train_only_reconstructs_legacy_and_complete_val_parity(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch); out=run/"audit"
    result=mod.verify_legacy_fit(data,legacy,out,root=run,official_validator=False,allow_test_fixture=True)
    assert result["status"]=="completed" and result["coefficient_exact"] is True
    doc=json.loads((out/"simple-baseline-train-fit-verification.json").read_text(encoding="utf-8"))
    assert doc["artifact_role"]=="train_only_simple_baseline_fit" and doc["split"]=="train"
    assert doc["test_access"]["test_origins_evaluated"] is False
    assert doc["reconstruction"]["exact_array_equality"] is True
    assert doc["train_node_window_samples"]>0
    assert doc["validation"]["window_count"]==3
    assert isinstance(doc["node_ids"],list) and doc["node_ids"]==[10,20]
    assert len(doc["source_hashes"])==22 and doc["source_hashes_sha256"]==mod.canonical_hash(doc["source_hashes"])
    assert doc["training_source_fingerprint_sha256"]==mod.canonical_hash(doc["training_source_fingerprint"])
    assert doc["linear_coefficients_path"]==str((legacy/"linear_coefficients.npy").resolve())
    assert doc["node_ids_sha256"]
    # Only target accesses used to make/replay val and fit train; none at test origin.
    assert all(x < 9 for x in ds.calls)


def test_rejects_wrong_pipeline_chronology_before_array_loader(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch)
    state=json.loads((run/"pipeline-state.json").read_text());state["stages"][0]["status"]="running"
    (run/"pipeline-state.json").write_text(json.dumps(state),encoding="utf-8")
    from edsgn import engine
    calls=[];monkeypatch.setattr(engine,"datasets_at",lambda *_:(calls.append(True) or [ds]))
    with pytest.raises(mod.VerificationError,match="P2=passed/pass"):
        mod.verify_legacy_fit(data,legacy,run/"blocked",root=run,official_validator=False,allow_test_fixture=True)
    assert calls==[]


def test_input_mutation_during_serialization_is_rejected(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch); real=mod._snapshot; calls={"n":0}
    def changed(*args,**kwargs):
        calls["n"]+=1
        if calls["n"]==3: data.write_bytes(b"changed after numeric pass")
        return real(*args,**kwargs)
    monkeypatch.setattr(mod,"_snapshot",changed)
    with pytest.raises(mod.VerificationError,match="changed"):
        mod.verify_legacy_fit(data,legacy,run/"audit",root=run,official_validator=False,allow_test_fixture=True)


def test_reconstruction_matches_engine_ridge_and_ignores_validation_targets(monkeypatch):
    from edsgn import engine
    ds=_toy(); train=[ds]
    coef,info=mod.reconstruct_train_coefficients(train)
    # A changed validation target must not change the train-only coefficients.
    altered=_toy();altered.targets[8:11]+=10000
    coef2,_=mod.reconstruct_train_coefficients([altered])
    assert np.array_equal(coef,coef2)
    assert info["ridge"]==1e-3*info["train_node_window_samples"]


def test_dual_lease_is_owned_and_foreign_lease_is_never_removed(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,"_owned_processes",lambda _root:[])
    monkeypatch.setattr(mod,"_budget",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":1.0})
    run=tmp_path/"run";run.mkdir();out=run/"audit"
    g=mod.Guard(run,out,"toy-job",["python"]);g.acquire()
    assert len(g.paths)==2 and all(p.exists() for p in g.paths)
    g.release();assert all(not p.exists() for p in g.paths)
    foreign=run/"model-resource.lock";foreign.write_text('{"owner":"other"}',encoding="utf-8")
    g2=mod.Guard(run,out,"toy-job2",["python"])
    with pytest.raises(mod.VerificationError,match="foreign/stale"):
        g2.acquire()
    assert foreign.exists() and not g2.paths


def test_budget_gap_or_stoploss_fails_before_data_loader(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(mod,"_owned_processes",lambda _root:[])
    monkeypatch.setattr(mod,"_budget",lambda *_a,**_k:{"coverage_complete":False,"spent_seconds":0,"coverage_gaps":["fixture"]})
    guard=mod.Guard(run,run/"audit","job",["python"])
    with pytest.raises(mod.VerificationError,match="coverage incomplete"):
        guard.acquire()
    # A complete ledger at/over the frozen 72-hour boundary also fails closed.
    monkeypatch.setattr(mod,"_budget",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":72*3600})
    with pytest.raises(mod.VerificationError,match="stop-loss"):
        mod.Guard(run,run/"audit","job2",["python"]).acquire()


def test_official_before_p3_validator_precedes_train_val_loader(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch)
    from edsgn import engine
    order=[]
    monkeypatch.setattr(mod,"_official_before_p3",lambda root:order.append("official-before-P3"))
    def loader(path,split):order.append("load:"+split);return [ds]
    monkeypatch.setattr(engine,"datasets_at",loader)
    mod.verify_legacy_fit(data,legacy,run/"audit",root=run,official_validator=False,
      validator_callback=lambda root:order.append("official-before-P3"),allow_test_fixture=True)
    assert order[0]=="official-before-P3"
    assert "load:test" not in order


def test_guarded_public_route_keeps_lease_until_outer_receipt(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(mod,"_owned_processes",lambda _root:[])
    monkeypatch.setattr(mod,"_budget",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":1.0})
    order=[];g=mod.Guard(run,run/"audit","toy-job",["python"]);g.acquire()
    result=mod.verify_legacy_fit(data,legacy,run/"audit",root=run,checkpoint=g.checkpoint,
      validator_callback=lambda _root:order.append("mock-validator"),allow_test_fixture=True)
    assert result["status"]=="completed" and order==["mock-validator"]
    assert all(p.exists() for p in g.paths),"inner verifier must not release outer CLI leases"
    g.write_receipt("completed");g.release();assert all(not p.exists() for p in g.paths)


def test_prevalidator_input_mutation_rejected_before_array_load(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch)
    from edsgn import engine
    loaded=[];monkeypatch.setattr(engine,"datasets_at",lambda *_:(loaded.append(True) or [ds]))
    def mutate(_root):data.write_bytes(b"mutated during validator")
    with pytest.raises(mod.VerificationError,match="changed"):
        mod.verify_legacy_fit(data,legacy,run/"audit",root=run,official_validator=False,
          validator_callback=mutate,allow_test_fixture=True)
    assert loaded==[]


def test_extra_output_after_serialization_rejected(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch);real=mod._snapshot;calls={"n":0}
    def attack(*args,**kwargs):
        calls["n"]+=1
        result=real(*args,**kwargs)
        if calls["n"]>=4:
            (run/"audit"/"rogue.txt").write_text("unexpected",encoding="utf-8")
        return result
    monkeypatch.setattr(mod,"_snapshot",attack)
    with pytest.raises(mod.VerificationError,match="output membership"):
        mod.verify_legacy_fit(data,legacy,run/"audit",root=run,official_validator=False,allow_test_fixture=True)


def test_canonical_package_inventory_produces_22_bound_sources(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch);out=run/"audit"
    mod.verify_legacy_fit(data,legacy,out,root=run,official_validator=False,allow_test_fixture=True)
    from edsgn.engine import source_manifest
    doc=json.loads((out/"simple-baseline-train-fit-verification.json").read_text(encoding="utf-8"))
    assert len(source_manifest())==21
    assert len(doc["source_hashes"])==22
    assert {Path(p).name for p in doc["source_hashes"] if Path(p).parent.name=="edsgn"}==set(source_manifest())


def test_shared_cross_stage_refs_are_allowed_but_within_stage_duplicates_fail(tmp_path):
    run=tmp_path/"run";run.mkdir(); shared=run/"shared.json";shared.write_text("{}",encoding="utf-8")
    digest=mod.sha256_file(shared); ref={"path":"shared.json","sha256":digest}
    state={"stages":[
      {"id":"P2","status":"passed","gate_verdict":"pass","inputs":[ref],"evidence":[]},
      {"id":"P3","status":"running","gate_verdict":"pending","inputs":[],"evidence":[]},
      {"id":"P1","status":"passed","gate_verdict":"pass","inputs":[],"evidence":[ref]}]}
    (run/"pipeline-state.json").write_text(json.dumps(state),encoding="utf-8")
    _,refs=mod._state_and_refs(run);assert refs[str(shared.resolve())]==digest
    state["stages"][2]["inputs"]=[ref]
    (run/"pipeline-state.json").write_text(json.dumps(state),encoding="utf-8")
    with pytest.raises(mod.VerificationError,match="duplicate passed-stage reference"):
        mod._state_and_refs(run)
    state["stages"][2]["inputs"]=[];state["stages"][2]["evidence"]=[{"path":"shared.json","sha256":"0"*64}]
    (run/"pipeline-state.json").write_text(json.dumps(state),encoding="utf-8")
    with pytest.raises(mod.VerificationError,match="stale/out-of-root"):
        mod._state_and_refs(run)


def test_cli_completed_receipt_is_written_before_outer_lease_release(tmp_path,monkeypatch,capsys):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(mod,"_owned_processes",lambda _root:[])
    monkeypatch.setattr(mod,"_budget",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":1.0})
    monkeypatch.setattr(mod,"_official_before_p3",lambda _root:None)
    receipt_events=[];original_write=mod.Guard.write_receipt
    def record_write(g,status,error=None):
        receipt_events.append((status,all(p.exists() for p in g.paths)))
        return original_write(g,status,error)
    monkeypatch.setattr(mod.Guard,"write_receipt",record_write)
    assert mod.main(["--dataset",str(data),"--legacy-output",str(legacy),"--output",str(run/"audit"),"--root",str(run)])==0
    rows=list((run/"logs"/"offline-phase-wall").glob("simple-fit-verify-*.json"))
    assert len(rows)==1
    receipt=json.loads(rows[0].read_text(encoding="utf-8"))
    assert receipt["status"]=="completed" and receipt["returncode"]==0
    assert ("completed",True) in receipt_events
    assert receipt["output_files"] and receipt["input_fence_before_sha256"]==receipt["input_fence_after_sha256"]
    assert not (run/"logs"/"model-resource.lock").exists()
    assert not (run/"logs"/"offline-phase-wall"/"campaign.lock").exists()


def test_cli_coefficient_mismatch_writes_failed_receipt_under_guard(tmp_path,monkeypatch,capsys):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch)
    from edsgn import engine
    coef=np.load(legacy/"linear_coefficients.npy",allow_pickle=False);coef[0,0]+=0.25
    np.save(legacy/"linear_coefficients.npy",coef,allow_pickle=False)
    replay,_=mod.validation_replay([ds],coef);row=replay["linear"]
    np.savez(legacy/"linear_val_predictions.npz",predictions=row["predictions"],targets=row["targets"],
      target_mask=row["target_mask"],origins=row["pairs"])
    from edsgn.common import csv_rows
    csv_rows(legacy/"linear_val_metrics.csv",engine.metric_rows(row,ds.horizons,"linear",0,"val"))
    monkeypatch.setattr(mod,"_owned_processes",lambda _root:[])
    monkeypatch.setattr(mod,"_budget",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":1.0})
    monkeypatch.setattr(mod,"_official_before_p3",lambda _root:None)
    receipt_events=[];original_write=mod.Guard.write_receipt
    def record_write(g,status,error=None):
        receipt_events.append((status,all(p.exists() for p in g.paths)))
        return original_write(g,status,error)
    monkeypatch.setattr(mod.Guard,"write_receipt",record_write)
    with pytest.raises(mod.VerificationError,match="refusing completed"):
        mod.main(["--dataset",str(data),"--legacy-output",str(legacy),"--output",str(run/"audit"),"--root",str(run)])
    rows=list((run/"logs"/"offline-phase-wall").glob("simple-fit-verify-*.json"));assert len(rows)==1
    receipt=json.loads(rows[0].read_text(encoding="utf-8"))
    report=json.loads((run/"audit"/"verification-report.json").read_text(encoding="utf-8"))
    assert receipt["status"]=="failed" and receipt["returncode"]==1
    assert ("failed",True) in receipt_events
    assert report["status"]!="completed" and receipt["output_files"]
    assert not (run/"logs"/"model-resource.lock").exists()
    assert not (run/"logs"/"offline-phase-wall"/"campaign.lock").exists()


def test_coefficient_mismatch_keeps_failed_report_and_never_completes_receipt(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch)
    from edsgn import engine
    coef=np.load(legacy/"linear_coefficients.npy",allow_pickle=False);coef[0,0]+=0.25
    np.save(legacy/"linear_coefficients.npy",coef,allow_pickle=False)
    replay,_=mod.validation_replay([ds],coef)
    row=replay["linear"]
    np.savez(legacy/"linear_val_predictions.npz",predictions=row["predictions"],targets=row["targets"],
      target_mask=row["target_mask"],origins=row["pairs"])
    from edsgn.common import csv_rows
    csv_rows(legacy/"linear_val_metrics.csv",engine.metric_rows(row,ds.horizons,"linear",0,"val"))
    monkeypatch.setattr(mod,"_owned_processes",lambda _root:[])
    monkeypatch.setattr(mod,"_budget",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":1.0})
    out=run/"audit";g=mod.Guard(run,out,"mismatch-job",["python"]);g.acquire()
    with pytest.raises(mod.VerificationError,match="refusing completed"):
        mod.verify_legacy_fit(data,legacy,out,root=run,checkpoint=g.checkpoint,
          validator_callback=lambda _root:None,allow_test_fixture=True)
    report=json.loads((out/"verification-report.json").read_text(encoding="utf-8"))
    assert report["status"]=="coefficient_mismatch_root_decision_required"
    g.write_receipt("failed",mod.VerificationError("mismatch"))
    receipt=json.loads((run/"logs"/"offline-phase-wall"/"mismatch-job.json").read_text(encoding="utf-8"))
    assert receipt["status"]=="failed" and receipt["returncode"]==1
    g.release()


@pytest.mark.parametrize("argv",[
    ["python.exe","-u","-m","edsgn.train","--config","x.json"],
    ["python.exe","-B","-X","dev","-m","edsgn.profile","--config","x.json"],
    ["python.exe","-u",str(RUN/"edsgn"/"staging"/"frozen_graph_state_diagnostic_v8.py"),"--config","x.json"],
])
def test_owned_scan_finds_python_flags_before_module_or_staging_script(tmp_path,monkeypatch,argv):
    class FakeProc:
        pid=998877
        info={"pid":998877,"name":"python.exe","exe":"C:/Python/python.exe","cmdline":argv,"cwd":str(RUN)}
        def name(self):return "python.exe"
    class FakePsutil:
        NoSuchProcess=type("NoSuchProcess",(Exception,),{})
        ZombieProcess=type("ZombieProcess",(Exception,),{})
        AccessDenied=type("AccessDenied",(Exception,),{})
        @staticmethod
        def process_iter(_attrs):return [FakeProc()]
    monkeypatch.setitem(sys.modules,"psutil",FakePsutil)
    assert mod._owned_processes(RUN,process_iter=[FakeProc()],current_pid=1,ancestor_pids=[])==[{"pid":998877}]


def test_public_guarded_cli_rejects_active_owned_job_before_any_dataset_load(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch)
    from edsgn import engine
    loads=[];monkeypatch.setattr(engine,"datasets_at",lambda *_:(loads.append(True) or [ds]))
    monkeypatch.setattr(mod,"_official_before_p3",lambda _root:None)
    monkeypatch.setattr(mod,"_budget",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":1.0})
    monkeypatch.setattr(mod,"_owned_processes",lambda _root:[{"pid":424242,"reason":"active owned train command"}])
    with pytest.raises(mod.VerificationError,match="another owned model job is active"):
        mod.main(["--dataset",str(data),"--legacy-output",str(legacy),"--output",str(run/"blocked"),"--root",str(run)])
    assert loads==[]
    receipts=list((run/"logs"/"offline-phase-wall").glob("simple-fit-verify-*.json"))
    assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["status"]=="failed" and receipt["returncode"]==1
    assert not (run/"logs"/"model-resource.lock").exists()
    assert not (run/"logs"/"offline-phase-wall"/"campaign.lock").exists()
    assert not (run/"blocked").exists()


def test_public_guarded_cli_rejects_job_appearing_midphase(tmp_path,monkeypatch):
    run,data,legacy,ds=_fixture(tmp_path,monkeypatch)
    from edsgn import engine
    loads=[];monkeypatch.setattr(engine,"datasets_at",lambda *_:(loads.append(True) or [ds]))
    monkeypatch.setattr(mod,"_official_before_p3",lambda _root:None)
    monkeypatch.setattr(mod,"_budget",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":1.0})
    scans={"n":0}
    def appears(_root):
        scans["n"]+=1
        return [] if scans["n"]==1 else [{"pid":31337,"reason":"new owned profile job"}]
    monkeypatch.setattr(mod,"_owned_processes",appears)
    with pytest.raises(mod.VerificationError,match="another owned model job is active"):
        mod.main(["--dataset",str(data),"--legacy-output",str(legacy),"--output",str(run/"midphase-blocked"),"--root",str(run)])
    assert scans["n"]>=2 and loads==[]
    receipts=list((run/"logs"/"offline-phase-wall").glob("simple-fit-verify-*.json"))
    assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["status"]=="failed" and receipt["returncode"]==1
    assert not (run/"logs"/"model-resource.lock").exists()
    assert not (run/"logs"/"offline-phase-wall"/"campaign.lock").exists()
    assert not (run/"midphase-blocked").exists()
