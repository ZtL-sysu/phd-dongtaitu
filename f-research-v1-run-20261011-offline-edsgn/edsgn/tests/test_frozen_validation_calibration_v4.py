import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

RUN = Path(r"C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn")
sys.path[:0] = [str(RUN / "edsgn/staging"), str(RUN / "edsgn/src")]
import frozen_validation_calibration_v4 as v3
from edsgn import engine
from edsgn.data import load_dataset


def _make_bundle(path, test_offset=0.0):
    T, N, F = 14, 2, 1
    raw = np.arange(T*N*F, dtype=np.float32).reshape(T,N,F)
    raw[9:] += test_offset
    values = np.log1p(raw)
    input_mask = np.ones_like(raw, dtype=bool)
    target_mask = np.ones_like(raw, dtype=bool)
    age = np.zeros_like(raw, dtype=np.float32)
    metadata = {
        "partition":"chronological", "count_data":True, "event_kind":"synthetic",
        "history":2,"hmax":2,"horizons":[1,2],
        "splits":{"train":[0,5],"validation":[5,9],"test":[9,14]},
        "metadata":{"n_bins_total":T,"seed":17},
    }
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata)),
        values=values, raw_values=raw, targets=raw, input_mask=input_mask,
        target_mask=target_mask, obs_age=age, interval_end=np.arange(T,dtype=np.float64)+1,
        node_ids=np.arange(N), mean=np.zeros(F,np.float32), std=np.ones(F,np.float32),
        scale=np.ones((N,F),np.float32), events=np.empty((0,3),np.int64))
    return metadata


class _ZeroModel(torch.nn.Module):
    def forward(self, x, **kwargs):
        return x.new_zeros((x.shape[0], 2, x.shape[2], x.shape[3]))


class _Guard:
    def __init__(self): self.calls = 0
    def checkpoint(self): self.calls += 1


def test_chronological_metadata_support_reads_metadata_member_and_matches_formula(tmp_path, monkeypatch):
    path=tmp_path/"chron.npz"
    meta=_make_bundle(path)
    real_load=np.load; touched=[]
    class Proxy:
        def __init__(self, archive): self.archive=archive; self.files=archive.files
        def __enter__(self): return self
        def __exit__(self,*a): self.archive.close()
        def __getitem__(self,k):
            touched.append(k)
            if k != "metadata_json": raise AssertionError(f"numeric member accessed: {k}")
            return self.archive[k]
    monkeypatch.setattr(v3.np,"load",lambda *a,**kw: Proxy(real_load(*a,**kw)))
    assert v3._safe_val_collection(path)==[path]
    assert v3.metadata_test_support(path)=={"17":4}  # metadata support only
    assert set(touched)=={"metadata_json"}
    assert v3._legal_origin_count(meta,"validation")==3


def test_real_bundle_loader_and_two_origin_cpu_eval_ignore_test_suffix_targets(tmp_path):
    p1=tmp_path/"a.npz"; p2=tmp_path/"b.npz"
    _make_bundle(p1,0); _make_bundle(p2,100000)
    ds1=engine.datasets_at(p1,"val"); ds2=engine.datasets_at(p2,"val")
    assert len(ds1)==len(ds2)==1
    assert np.array_equal(ds1[0].origins("val"),np.array([4,5,6]))
    # Evaluate two validation origins with the production loader/evaluator on CPU.
    model=_ZeroModel()
    r1=engine.evaluate_batches(model,ds1,"val","cpu",batch_size=2,limit=2)
    r2=engine.evaluate_batches(model,ds2,"val","cpu",batch_size=2,limit=2)
    assert len(r1["pairs"])==2 and r1["pairs"]==r2["pairs"]
    for key in ("predictions","targets","target_mask"):
        assert np.array_equal(r1[key],r2[key]), key
    # Metadata support and train-only statistics are independent of changed test suffix.
    assert v3.metadata_test_support(p1)==v3.metadata_test_support(p2)=={"17":4}
    assert np.array_equal(ds1[0].raw_values[:5],ds2[0].raw_values[:5])
    import torch as _torch
    ckpt=tmp_path/"toy.pt"
    _torch.save({"state_dict":{"weight":_torch.ones(1)},"config":{"seed":0},
                 "task":{"scope":"toy","normalization":"train-only"},
                 "provenance":{"scope":"full_protocol"}},ckpt)
    loaded=_torch.load(ckpt,map_location="cpu",weights_only=False)
    v3._require_fp32_payload(loaded)
    assert loaded["task"]["normalization"]=="train-only"


def test_validation_guards_check_each_batch_and_reject_test_origins(tmp_path):
    path=tmp_path/"chron.npz"; _make_bundle(path)
    ds=engine.datasets_at(path,"val"); guard=_Guard(); restore=v3._install_validation_guards(engine,engine,ds,guard)
    try:
        b=engine.batch_at(ds,[(0,4)],"cpu")
        assert b["targets"].shape[0]==1 and guard.calls==1
        with pytest.raises(RuntimeError,match="unauthorized"):
            engine.batch_at(ds,[(0,10)],"cpu")
        with pytest.raises(RuntimeError,match="non-validation split"):
            engine.index_origins(ds,"test")
    finally: restore()
    assert engine.index_origins(ds,"val")==[(0,4),(0,5),(0,6)]


def test_hash_snapshot_and_gate_verdict_fail_closed(tmp_path):
    f=tmp_path/"evidence"; f.write_text("ok",encoding="utf8")
    inp=tmp_path/"input"; inp.write_text("input",encoding="utf8")
    import hashlib
    h=hashlib.sha256(f.read_bytes()).hexdigest(); ih=hashlib.sha256(inp.read_bytes()).hexdigest()
    (tmp_path/"pipeline-state.json").write_text(json.dumps({"stages":[
      {"id":"P2","status":"passed","gate_verdict":"pass"},
      {"id":"P3","status":"passed","gate_verdict":"pass"},
      {"id":"P3A","status":"passed","gate_verdict":"pass","inputs":[{"path":"input","sha256":ih}],"evidence":[{"path":"evidence","sha256":h}]},
      {"id":"P4","status":"running"}]}),encoding="utf8")
    assert v3._canonical_stage_gate(tmp_path)
    with pytest.raises(RuntimeError,match="pass"):
      state=json.loads((tmp_path/"pipeline-state.json").read_text()); state["stages"][2]["gate_verdict"]="fail"
      (tmp_path/"pipeline-state.json").write_text(json.dumps(state)); v3._canonical_stage_gate(tmp_path)
    with pytest.raises(RuntimeError,match="changed"):
      v3._verify_hash_snapshot({"x":"a"},{"x":"b"})



def test_strict_gate_rejects_duplicate_stage_and_json_keys(tmp_path):
    proof=tmp_path/"proof";proof.write_text("ok",encoding="utf8")
    evidence=tmp_path/"evidence";evidence.write_text("ok",encoding="utf8")
    h=__import__("hashlib").sha256(proof.read_bytes()).hexdigest()
    eh=__import__("hashlib").sha256(evidence.read_bytes()).hexdigest()
    state={"stages":[{"id":"P2","status":"passed","gate_verdict":"pass"},
      {"id":"P3","status":"passed","gate_verdict":"pass"},
      {"id":"P3A","status":"passed","gate_verdict":"pass","inputs":[{"path":"proof","sha256":h}],"evidence":[{"path":"evidence","sha256":eh}]},
      {"id":"P4","status":"running"}]}
    pipe=tmp_path/"pipeline-state.json";pipe.write_text(json.dumps(state),encoding="utf8")
    assert v3._canonical_stage_gate(tmp_path)["stages"]
    duplicate=json.dumps(state).replace('"id": "P2"','"id": "P2", "id": "P2"',1)
    pipe.write_text(duplicate,encoding="utf8")
    with pytest.raises(ValueError,match="duplicate JSON key"):
      v3._canonical_stage_gate(tmp_path)
    state["stages"].append(state["stages"][0]);pipe.write_text(json.dumps(state),encoding="utf8")
    with pytest.raises(RuntimeError,match="unique"):
      v3._canonical_stage_gate(tmp_path)


def test_strict_gate_requires_nonempty_distinct_inroot_bound_inputs_and_evidence(tmp_path):
    proof=tmp_path/"proof";proof.write_text("ok",encoding="utf8")
    evidence=tmp_path/"evidence";evidence.write_text("ok",encoding="utf8")
    h=__import__("hashlib").sha256(proof.read_bytes()).hexdigest()
    eh=__import__("hashlib").sha256(evidence.read_bytes()).hexdigest()
    state={"stages":[{"id":"P2","status":"passed","gate_verdict":"pass"},
      {"id":"P3","status":"passed","gate_verdict":"pass"},
      {"id":"P3A","status":"passed","gate_verdict":"pass","inputs":[{"path":"proof","sha256":h}],"evidence":[{"path":"evidence","sha256":eh}]},
      {"id":"P4","status":"running"}]}
    pipe=tmp_path/"pipeline-state.json";pipe.write_text(json.dumps(state),encoding="utf8")
    state["stages"][2]["evidence"]=[{"path":"proof","sha256":h}]
    pipe.write_text(json.dumps(state),encoding="utf8")
    with pytest.raises(RuntimeError,match="duplicate P3A"):
      v3._canonical_stage_gate(tmp_path)
    state["stages"][2]["evidence"]=[{"path":"..\\outside","sha256":h}]
    pipe.write_text(json.dumps(state),encoding="utf8")
    with pytest.raises(ValueError,match="escapes"):
      v3._canonical_stage_gate(tmp_path)


def test_final_gate_rechecks_bytes_and_bound_evidence(tmp_path):
    proof=tmp_path/"proof";proof.write_text("ok",encoding="utf8")
    h=__import__("hashlib").sha256(proof.read_bytes()).hexdigest()
    state={"stages":[{"id":"P2","status":"passed","gate_verdict":"pass"},
      {"id":"P3","status":"passed","gate_verdict":"pass"},
      {"id":"P3A","status":"passed","gate_verdict":"pass","inputs":[{"path":"proof","sha256":h}],"evidence":[{"path":"evidence","sha256":h}]},
      {"id":"P4","status":"running"}]}
    evidence=tmp_path/"evidence";evidence.write_text("ok",encoding="utf8")
    pipe=tmp_path/"pipeline-state.json";pipe.write_text(json.dumps(state),encoding="utf8")
    before=pipe.read_bytes();initial=v3._canonical_stage_gate(tmp_path)
    assert v3._verify_final_canonical_gate(tmp_path,before,initial)==initial
    evidence.write_text("changed",encoding="utf8")
    with pytest.raises(RuntimeError,match="stale/missing"):
      v3._verify_final_canonical_gate(tmp_path,before,initial)
