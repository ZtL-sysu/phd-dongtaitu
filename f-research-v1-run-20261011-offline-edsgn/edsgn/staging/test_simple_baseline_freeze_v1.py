from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("simple_baseline_freeze_v1.py")
spec = importlib.util.spec_from_file_location("simple_baseline_freeze_v1", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FakePrepared:
    def __init__(self):
        from edsgn.data import PreparedData
        class SpyPrepared(PreparedData):
            def target(self, origin):
                self.target_calls.append(int(origin))
                return super().target(origin)
        T, N, F = 18, 3, 2
        raw = np.arange(T*N*F, dtype=np.float32).reshape(T,N,F) / 7.0
        mask = np.ones_like(raw, dtype=bool)
        self.ds = SpyPrepared(
            values=raw.copy(), targets=raw.copy(), input_mask=mask,
            target_mask=mask.copy(), obs_age=np.zeros_like(raw),
            interval_end=np.arange(T,dtype=np.float64), node_ids=np.array([11,22,33]),
            splits={"train":(0,10),"validation":(10,14),"test":(14,18)},
            mean=np.zeros(F),std=np.ones(F),scale=np.ones((N,F)),
            count_data=False,horizons=(1,2),history=2,hmax=2,
            metadata={"raw_values":raw},
        )
        self.ds.target_calls=[]


class FakeLaunchGuard(mod._LaunchGuard):
    def __init__(self):
        self.checkpoints=0

    def checkpoint(self):
        self.checkpoints+=1
        return None


def fake_checkpoint():
    return FakeLaunchGuard().checkpoint


def test_fit_only_train_and_refuses_existing_output(tmp_path, monkeypatch):
    fake=FakePrepared(); requests=[]
    monkeypatch.setattr(mod,"_load_datasets",lambda path,split: (requests.append(split) or [fake.ds]))
    dataset=tmp_path/"dataset.npz"; dataset.write_bytes(b"fake bundle fingerprint")
    out=tmp_path/"fit"
    guard=FakeLaunchGuard()
    result=mod.fit(dataset,out,checkpoint=guard.checkpoint)
    assert guard.checkpoints>=4
    assert requests==["train"]
    assert all(o in set(fake.ds.origins("train")) for o in fake.ds.target_calls)
    doc=json.loads(Path(result["fit_manifest"]).read_text(encoding="utf-8"))
    assert doc["split"]=="train" and not doc["validation_accessed"] and not doc["test_accessed"]
    assert doc["held_out_target_accessor_calls"] is False
    assert doc["held_out_arrays_may_be_materialized"] is True
    assert doc["fit"]["train_origins_by_trajectory"]==[fake.ds.origins("train").tolist()]
    with pytest.raises(FileExistsError): mod.fit(dataset,out,checkpoint=fake_checkpoint())


def test_staged_fit_api_cannot_bypass_launch_guard(tmp_path):
    with pytest.raises(mod.FrozenBaselineError,match="guarded CLI"):
        mod.fit(tmp_path/"unused.npz",tmp_path/"fit",checkpoint=lambda:None)


def test_pending_gate_blocks_before_loading_any_split(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr(mod,"_verify_gate",lambda *a: (_ for _ in ()).throw(mod.FrozenBaselineError("pending P3A")))
    monkeypatch.setattr(mod,"_load_datasets",lambda *a: calls.append(a))
    with pytest.raises(mod.FrozenBaselineError,match="pending P3A"):
        mod.frozen_evaluate(tmp_path/"data.npz",tmp_path/"fit.json",tmp_path/"accept.json","test",tmp_path/"out",root=tmp_path,
                            checkpoint=fake_checkpoint())
    assert calls==[]


def test_real_gate_checks_p3a_and_root_relative_raw_metric_hashes(tmp_path):
    source=tmp_path/"source.txt";source.write_text("immutable\n",encoding="utf-8")
    raw=tmp_path/"raw-val.csv";raw.write_text("method,mae\nlinear,0.5\n",encoding="utf-8")
    acceptance={"simple_baseline_freeze":{
        "fit_manifest_path":"fit.json","fit_manifest_sha256":"a"*64,
        "linear_coefficients_sha256":"b"*64,"training_statistics_sha256":"c"*64,
        "node_ids_sha256":"d"*64,"dataset_fingerprint_sha256":"e"*64,
        "training_source_fingerprint_sha256":"f"*64,"source_hashes_sha256":"0"*64,
        "accepted_validation_metrics":[{"path":"raw-val.csv","sha256":mod.sha256_file(raw)}]}}
    ap=tmp_path/"acceptance.json"
    ap.write_text(json.dumps(acceptance),encoding="utf-8")
    state={"stages":[{"id":"P3A","status":"passed","gate_verdict":"pass",
           "inputs":[{"path":"source.txt","sha256":mod.sha256_file(source)}],
           "evidence":[{"path":"acceptance.json","sha256":mod.sha256_file(ap)}]}]}
    (tmp_path/"pipeline-state.json").write_text(json.dumps(state),encoding="utf-8")
    _,freeze=mod._verify_gate(tmp_path,"acceptance.json")
    assert freeze["accepted_validation_metrics"][0]["path"]=="raw-val.csv"
    raw.write_text("changed\n",encoding="utf-8")
    with pytest.raises(mod.FrozenBaselineError,match="raw validation metric hash"):
        mod._verify_gate(tmp_path,"acceptance.json")


def test_evaluation_failure_receipt_keeps_evaluation_phase(monkeypatch,tmp_path):
    captured=[]
    class FakeGuard(FakeLaunchGuard):
        token="fixture"
        def release(self): pass
    monkeypatch.setattr(mod,"_acquire_launch_guard",lambda *a: FakeGuard())
    monkeypatch.setattr(mod,"_verify_gate",lambda *a: (_ for _ in ()).throw(mod.FrozenBaselineError("closed gate")))
    monkeypatch.setattr(mod,"_phase_receipt",lambda *a,**k: captured.append((a,k)))
    with pytest.raises(mod.FrozenBaselineError,match="closed gate"):
        mod.main(["evaluate","--dataset-path",str(tmp_path/"data.npz"),"--fit-manifest",str(tmp_path/"fit.json"),
                  "--acceptance",str(tmp_path/"accept.json"),"--split","test","--output-dir",str(tmp_path/"out")])
    assert len(captured)==1
    assert captured[0][0][2]=="evaluation"
    assert captured[0][0][6]=="stopped"
    assert captured[0][1]["error"].args==("closed gate",)


def test_frozen_validation_uses_bound_coefficients_and_raw_outputs(tmp_path, monkeypatch):
    fake=FakePrepared(); requests=[]
    monkeypatch.setattr(mod,"_load_datasets",lambda path,split: (requests.append(split) or [fake.ds]))
    dataset=tmp_path/"dataset.npz"; dataset.write_bytes(b"fake bundle fingerprint")
    fit_result=mod.fit(dataset,tmp_path/"fit",checkpoint=fake_checkpoint())
    fit_path=Path(fit_result["fit_manifest"])
    fit_doc=json.loads(fit_path.read_text(encoding="utf-8"))
    metric=tmp_path/"accepted-val.csv"; metric.write_text("method,mae\nlinear,1.0\n",encoding="utf-8")
    accepted_metrics=[{"path":str(metric),"sha256":mod.sha256_file(metric)}]
    acceptance_path=tmp_path/"acceptance.json"
    acceptance_path.write_text('{"test_fixture":true}\n',encoding="utf-8")
    freeze={"fit_manifest_path":str(fit_path),"fit_manifest_sha256":mod.sha256_file(fit_path),
            "linear_coefficients_sha256":fit_doc["linear_coefficients_sha256"],
            "training_statistics_sha256":fit_doc["training_statistics_sha256"],
            "node_ids_sha256":fit_doc["node_ids_sha256"],
            "training_source_fingerprint_sha256":fit_doc["training_source_fingerprint_sha256"],
            "source_hashes_sha256":fit_doc["source_hashes_sha256"],
            "dataset_fingerprint_sha256":mod.canonical_json_hash(mod.file_fingerprint(dataset,train_only=False)),
            "accepted_validation_metrics":accepted_metrics}
    monkeypatch.setattr(mod,"_verify_gate",lambda *a: ({"accepted":True},freeze))
    before=mod.sha256_file(fit_doc["linear_coefficients_path"])
    result=mod.frozen_evaluate(dataset,fit_path,acceptance_path,"val",tmp_path/"val-out",root=tmp_path,
                               checkpoint=fake_checkpoint())
    assert requests==["train","train","validation"]
    assert result["linear_coefficients_sha256_before"]==before==result["linear_coefficients_sha256_after"]
    assert result["test_accessed"] is False
    assert result["prediction_files"]["linear"]["window_count"]==len(fake.ds.origins("validation"))
    pred=np.load(result["prediction_files"]["linear"]["path"],allow_pickle=False)
    assert pred["predictions"].shape==(len(fake.ds.origins("validation")),2,3,2)
    assert np.array_equal(pred["origins"],np.array([(0,int(o)) for o in fake.ds.origins("validation")]))


def test_fit_arrays_and_predictions_match_legacy_engine_exactly(tmp_path, monkeypatch):
    from edsgn import engine
    fake=FakePrepared()
    monkeypatch.setattr(engine,"datasets_at",lambda _path,split:[fake.ds])
    dataset=tmp_path/"bundle.npz";dataset.write_bytes(b"toy identity")
    legacy_out=tmp_path/"legacy";engine.simple_baselines(dataset,legacy_out,split="val")
    frozen=mod.fit(dataset,tmp_path/"frozen",checkpoint=fake_checkpoint())
    legacy_coef=np.load(legacy_out/"linear_coefficients.npy",allow_pickle=False)
    frozen_coef=np.load(frozen["linear_coefficients_path"],allow_pickle=False)
    assert np.array_equal(frozen_coef,legacy_coef)
    validation=[fake.ds]
    for method in ("persistence","linear"):
        actual,_=mod._evaluate_method(validation,"val",method,
                                      frozen_coef if method=="linear" else None)
        with np.load(legacy_out/f"{method}_val_predictions.npz",allow_pickle=False) as expected:
            for key in ("predictions","targets","target_mask"):
                assert np.array_equal(actual[key],expected[key])
            assert np.array_equal(np.asarray(actual["pairs"]),expected["origins"])


def test_fit_manifest_fingerprint_digest_and_source_binding_fail_closed(tmp_path, monkeypatch):
    fake=FakePrepared();monkeypatch.setattr(mod,"_load_datasets",lambda *_:[fake.ds])
    dataset=tmp_path/"bundle.npz";dataset.write_bytes(b"first")
    fitted=mod.fit(dataset,tmp_path/"fit",checkpoint=fake_checkpoint())
    manifest=Path(fitted["fit_manifest"])
    doc=json.loads(manifest.read_text(encoding="utf-8"))
    doc["training_source_fingerprint_sha256"]="wrong"
    manifest.write_text(json.dumps(doc),encoding="utf-8")
    with pytest.raises(mod.FrozenBaselineError,match="fingerprint digest"):
        mod._verify_fit_manifest(manifest)


def test_frozen_eval_rejects_changed_train_bundle_before_loading_splits(tmp_path, monkeypatch):
    fake=FakePrepared();calls=[]
    monkeypatch.setattr(mod,"_load_datasets",lambda _path,split:(calls.append(split) or [fake.ds]))
    dataset=tmp_path/"bundle.npz";dataset.write_bytes(b"original train source")
    fitted=mod.fit(dataset,tmp_path/"fit",checkpoint=fake_checkpoint())
    fit_path=Path(fitted["fit_manifest"]);doc=json.loads(fit_path.read_text(encoding="utf-8"))
    freeze={"fit_manifest_path":str(fit_path),"fit_manifest_sha256":mod.sha256_file(fit_path),
        "linear_coefficients_sha256":doc["linear_coefficients_sha256"],
        "training_statistics_sha256":doc["training_statistics_sha256"],
        "node_ids_sha256":doc["node_ids_sha256"],
        "training_source_fingerprint_sha256":doc["training_source_fingerprint_sha256"],
        "source_hashes_sha256":doc["source_hashes_sha256"],
        "dataset_fingerprint_sha256":mod.canonical_json_hash(mod.file_fingerprint(dataset,train_only=False)),
        "accepted_validation_metrics":[{"path":"unused.csv","sha256":"0"*64}]}
    monkeypatch.setattr(mod,"_verify_gate",lambda *_a:( {},freeze))
    freeze["source_hashes_sha256"]="mismatched accepted source digest"
    with pytest.raises(mod.FrozenBaselineError,match="P3A-bound fitter/guard source hash"):
        mod.frozen_evaluate(dataset,fit_path,tmp_path/"accept.json","val",tmp_path/"eval-source",
                            root=tmp_path,checkpoint=fake_checkpoint())
    assert calls==["train"]
    freeze["source_hashes_sha256"]=doc["source_hashes_sha256"]
    dataset.write_bytes(b"changed train source")
    calls.clear()
    with pytest.raises(mod.FrozenBaselineError,match="Current train-source fingerprint"):
        mod.frozen_evaluate(dataset,fit_path,tmp_path/"accept.json","val",tmp_path/"eval",
                            root=tmp_path,checkpoint=fake_checkpoint())
    assert calls==[]


def test_launch_guard_claims_both_locks_checks_budget_and_cleans_owned_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(mod,"_owned_model_workers",lambda _root:[])
    monkeypatch.setattr(mod,"_budget_report",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":10.0})
    started=mod.time.perf_counter();epoch=mod.time.time();job="test-guard"
    command=["python","simple_baseline_freeze_v1.py"]
    guard=mod._acquire_launch_guard(tmp_path,tmp_path/"runs/out","training",job,
                                   started,epoch,command)
    lock_a=tmp_path/"logs/model-resource.lock"
    lock_b=tmp_path/"logs/offline-phase-wall/campaign.lock"
    assert lock_a.is_file() and lock_b.is_file()
    receipt=tmp_path/f"logs/offline-phase-wall/{job}.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"]=="running"
    guard.checkpoint();guard.release()
    assert not lock_a.exists() and not lock_b.exists()


def test_launch_guard_refuses_existing_lock_and_exhausted_budget_without_deleting(tmp_path, monkeypatch):
    lock=tmp_path/"logs/model-resource.lock";lock.parent.mkdir(parents=True);lock.write_text("owner")
    monkeypatch.setattr(mod,"_owned_model_workers",lambda _root:[])
    with pytest.raises(mod.FrozenBaselineError,match="lock exists"):
        mod._acquire_launch_guard(tmp_path,tmp_path/"runs/out","training","occupied",
                                 mod.time.perf_counter(),mod.time.time(),["python"])
    assert lock.read_text()=="owner"
    lock.unlink()
    monkeypatch.setattr(mod,"_budget_report",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":mod.WALL_BUDGET_S})
    with pytest.raises(mod.FrozenBaselineError,match="stop-loss"):
        mod._acquire_launch_guard(tmp_path,tmp_path/"runs/out","training","budget",
                                 mod.time.perf_counter(),mod.time.time(),["python"])
    assert not (tmp_path/"logs/model-resource.lock").exists()
    assert not (tmp_path/"logs/offline-phase-wall/campaign.lock").exists()


def test_owned_process_barrier_rejects_run_train_and_ignores_foreign_quant(tmp_path, monkeypatch):
    class Proc:
        def __init__(self,pid,cmd,cwd):
            self.info={"pid":pid,"name":"python.exe","exe":"C:/Python/python.exe","cmdline":cmd,"cwd":cwd}
    fake=types.SimpleNamespace(process_iter=lambda _attrs:iter([
        Proc(444,["python","-m","edsgn.train"],str(tmp_path)),
    ]),NoSuchProcess=type("NoSuchProcess",(Exception,),{}),
       AccessDenied=type("AccessDenied",(Exception,),{}),ZombieProcess=type("ZombieProcess",(Exception,),{}))
    monkeypatch.setitem(sys.modules,"psutil",fake)
    with pytest.raises(mod.FrozenBaselineError,match="owned model/runner/resource process"):
        mod._owned_model_workers(tmp_path)
    fake.process_iter=lambda _attrs:iter([Proc(445,["python","quantize.py"],str(tmp_path))])
    assert mod._owned_model_workers(tmp_path)==[]


def test_owned_process_barrier_fails_closed_on_python_access_denied(monkeypatch):
    denied=type("AccessDenied",(Exception,),{})
    class HiddenPython:
        @property
        def info(self): raise denied("process fields hidden")
        def name(self): return "python.exe"
    fake=types.SimpleNamespace(process_iter=lambda _attrs:iter([HiddenPython()]),
        NoSuchProcess=type("NoSuchProcess",(Exception,),{}),AccessDenied=denied,
        ZombieProcess=type("ZombieProcess",(Exception,),{}))
    monkeypatch.setitem(sys.modules,"psutil",fake)
    with pytest.raises(mod.FrozenBaselineError,match="failing closed"):
        mod._owned_model_workers(Path("C:/run-root"))


def test_budget_checkpoint_stops_before_cumulative_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(mod,"_owned_model_workers",lambda _root:[])
    monkeypatch.setattr(mod,"_budget_report",lambda *_a,**_k:{"coverage_complete":True,"spent_seconds":mod.WALL_BUDGET_S-0.1})
    start=mod.time.perf_counter()-0.2
    guard=mod._LaunchGuard(tmp_path,tmp_path/"runs/out","training","budget-midrun",
                           start,mod.time.time(),["python"],[],"token")
    with pytest.raises(mod.FrozenBaselineError,match="stop-loss reached during"):
        guard.checkpoint()


def test_budget_coverage_gap_fails_closed(tmp_path, monkeypatch):
    import aggregate_budget
    monkeypatch.setattr(aggregate_budget,"aggregate",lambda *_a,**_k:{
        "coverage_complete":False,"coverage_gaps":["missing timing"]})
    with pytest.raises(mod.FrozenBaselineError,match="coverage is incomplete"):
        mod._budget_report(tmp_path,tmp_path/"runs/out")


def test_failed_phase_receipt_keeps_exception_and_command(tmp_path):
    exc=RuntimeError("mock validation gate failure")
    started=mod.time.perf_counter()-0.01;epoch=mod.time.time()-0.01
    mod._phase_receipt(tmp_path,tmp_path/"runs/out","evaluation","failure-fixture",
                       started,epoch,"failed",["python","--split","test"],error=exc)
    receipt=json.loads((tmp_path/"logs/offline-phase-wall/failure-fixture.json").read_text(encoding="utf-8"))
    assert receipt["status"]=="failed" and receipt["returncode"]==1
    assert receipt["exception_type"]=="RuntimeError"
    assert receipt["exception_message"]=="mock validation gate failure"
    assert receipt["command"]==["python","--split","test"]
