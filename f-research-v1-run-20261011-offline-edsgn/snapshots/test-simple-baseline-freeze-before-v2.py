from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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


def test_fit_only_train_and_refuses_existing_output(tmp_path, monkeypatch):
    fake=FakePrepared(); requests=[]
    monkeypatch.setattr(mod,"_load_datasets",lambda path,split: (requests.append(split) or [fake.ds]))
    dataset=tmp_path/"dataset.npz"; dataset.write_bytes(b"fake bundle fingerprint")
    out=tmp_path/"fit"
    result=mod.fit(dataset,out)
    assert requests==["train"]
    assert all(o in set(fake.ds.origins("train")) for o in fake.ds.target_calls)
    doc=json.loads(Path(result["fit_manifest"]).read_text(encoding="utf-8"))
    assert doc["split"]=="train" and not doc["validation_accessed"] and not doc["test_accessed"]
    assert doc["fit"]["train_origins_by_trajectory"]==[fake.ds.origins("train").tolist()]
    with pytest.raises(FileExistsError): mod.fit(dataset,out)


def test_pending_gate_blocks_before_loading_any_split(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr(mod,"_verify_gate",lambda *a: (_ for _ in ()).throw(mod.FrozenBaselineError("pending P3A")))
    monkeypatch.setattr(mod,"_load_datasets",lambda *a: calls.append(a))
    with pytest.raises(mod.FrozenBaselineError,match="pending P3A"):
        mod.frozen_evaluate(tmp_path/"data.npz",tmp_path/"fit.json",tmp_path/"accept.json","test",tmp_path/"out",root=tmp_path)
    assert calls==[]


def test_real_gate_checks_p3a_and_root_relative_raw_metric_hashes(tmp_path):
    source=tmp_path/"source.txt";source.write_text("immutable\n",encoding="utf-8")
    raw=tmp_path/"raw-val.csv";raw.write_text("method,mae\nlinear,0.5\n",encoding="utf-8")
    acceptance={"simple_baseline_freeze":{
        "fit_manifest_path":"fit.json","fit_manifest_sha256":"a"*64,
        "linear_coefficients_sha256":"b"*64,"training_statistics_sha256":"c"*64,
        "node_ids_sha256":"d"*64,"dataset_fingerprint_sha256":"e"*64,
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
    monkeypatch.setattr(mod,"_verify_gate",lambda *a: (_ for _ in ()).throw(mod.FrozenBaselineError("closed gate")))
    monkeypatch.setattr(mod,"_phase_receipt",lambda *a: captured.append(a))
    with pytest.raises(mod.FrozenBaselineError,match="closed gate"):
        mod.main(["evaluate","--dataset-path",str(tmp_path/"data.npz"),"--fit-manifest",str(tmp_path/"fit.json"),
                  "--acceptance",str(tmp_path/"accept.json"),"--split","test","--output-dir",str(tmp_path/"out")])
    assert len(captured)==1
    assert captured[0][2]=="evaluation"
    assert captured[0][5]=="failed"


def test_frozen_validation_uses_bound_coefficients_and_raw_outputs(tmp_path, monkeypatch):
    fake=FakePrepared(); requests=[]
    monkeypatch.setattr(mod,"_load_datasets",lambda path,split: (requests.append(split) or [fake.ds]))
    dataset=tmp_path/"dataset.npz"; dataset.write_bytes(b"fake bundle fingerprint")
    fit_result=mod.fit(dataset,tmp_path/"fit")
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
            "dataset_fingerprint_sha256":mod.canonical_json_hash(mod.file_fingerprint(dataset,train_only=False)),
            "accepted_validation_metrics":accepted_metrics}
    monkeypatch.setattr(mod,"_verify_gate",lambda *a: ({"accepted":True},freeze))
    before=mod.sha256_file(fit_doc["linear_coefficients_path"])
    result=mod.frozen_evaluate(dataset,fit_path,acceptance_path,"val",tmp_path/"val-out",root=tmp_path)
    assert requests==["train","train","validation"]
    assert result["linear_coefficients_sha256_before"]==before==result["linear_coefficients_sha256_after"]
    assert result["test_accessed"] is False
    assert result["prediction_files"]["linear"]["window_count"]==len(fake.ds.origins("validation"))
    pred=np.load(result["prediction_files"]["linear"]["path"],allow_pickle=False)
    assert pred["predictions"].shape==(len(fake.ds.origins("validation")),2,3,2)
    assert np.array_equal(pred["origins"],np.array([(0,int(o)) for o in fake.ds.origins("validation")]))
