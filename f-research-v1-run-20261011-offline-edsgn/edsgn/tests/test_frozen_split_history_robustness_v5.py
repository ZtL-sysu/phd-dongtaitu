import json
import shutil
import uuid

import numpy as np
import pytest
import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn" / "staging"))
sys.path.insert(0, str(ROOT / "edsgn" / "src"))
sys.path.insert(0, str(ROOT / "edsgn" / "scripts"))

import frozen_split_history_robustness_v5 as gated
import c06_domain_manifest_preflight_v4 as domain_preflight
from edsgn import engine
from edsgn.common import sha256
from edsgn.data import PreparedData
from edsgn.scheduling import Policy


def _data(*, seed, partition, node_ids=(7, 9), mean=(0.25,), std=(0.5,)):
    raw = (np.arange(24, dtype=np.float32).reshape(12, 2, 1) + seed) / 10
    mask = np.ones_like(raw, dtype=bool)
    mask[(seed + 2) % 12, seed % 2, 0] = False
    values = np.where(mask, (raw - np.asarray(mean)) / np.asarray(std), 0).astype(np.float32)
    targets = raw + 20
    age = np.zeros_like(values)
    previous = np.zeros((2, 1), dtype=np.float32)
    for t in range(12):
        previous = np.where(mask[t], 0, previous + 1)
        age[t] = previous
    return PreparedData(values=values, targets=targets, input_mask=mask,
        target_mask=np.ones_like(mask), obs_age=age, interval_end=np.arange(12),
        node_ids=np.asarray(node_ids), splits={"train": (0, 6), "test": (3, 6),
        "validation": (3, 6)}, mean=np.asarray(mean), std=np.asarray(std),
        scale=np.ones((2, 1), dtype=np.float32), count_data=False, horizons=(1, 2),
        history=3, hmax=2, events=np.empty((0, 3), dtype=np.int64),
        event_kind="synthetic", metadata={"seed": seed, "source": "toy-synthetic",
        "raw_values": raw}, partition=partition)


def _save_data(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"event_kind": data.event_kind, "partition": data.partition,
        "splits": data.splits, "count_data": data.count_data,
        "horizons": data.horizons, "history": data.history, "hmax": data.hmax,
        "metadata": {k: v for k, v in data.metadata.items() if k != "raw_values"}}
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata)),
        values=data.values, raw_values=data.raw_values, targets=data.targets,
        input_mask=data.input_mask, target_mask=data.target_mask, obs_age=data.obs_age,
        interval_end=data.interval_end, node_ids=data.node_ids, mean=data.mean,
        std=data.std, scale=data.scale, events=data.events)


def _state(path, *, p3a_verdict="pass", alter_evidence=False, duplicate=False):
    rows = []
    for name in ("P2", "P3", "P3A"):
        source = path.parent / f"{name}-input.txt"
        evidence = path.parent / f"{name}-evidence.txt"
        source.write_text(f"{name} input", encoding="utf-8")
        evidence.write_text(f"{name} evidence", encoding="utf-8")
        rows.append({"id": name, "status": "passed",
            "gate_verdict": p3a_verdict if name == "P3A" else "pass",
            "inputs": [{"path": source.name, "sha256": sha256(source)}],
            "evidence": [{"path": evidence.name, "sha256": sha256(evidence)}]})
    if alter_evidence:
        (path.parent / "P3A-evidence.txt").write_text("tampered", encoding="utf-8")
    rows.append({"id": "P4", "status": "running", "inputs": [], "evidence": []})
    if duplicate:
        rows.append(dict(rows[0]))
    path.write_text(json.dumps({"stages": rows}), encoding="utf-8")
    return path


@pytest.mark.parametrize("verdict,tamper,duplicate", [("fail", False, False), ("pass", True, False), ("pass", False, True)])
def test_gate_rejects_bad_verdict_stale_evidence_and_duplicate_stage(tmp_path, monkeypatch,
                                                                     verdict, tamper, duplicate):
    monkeypatch.setattr(gated, "ROOT", tmp_path)
    state = _state(tmp_path / "pipeline-state.json", p3a_verdict=verdict,
                   alter_evidence=tamper, duplicate=duplicate)
    with pytest.raises(RuntimeError):
        gated.require_phase_gate("test", state)


def test_gate_requires_canonical_pipeline_path(tmp_path, monkeypatch):
    monkeypatch.setattr(gated, "ROOT", tmp_path)
    _state(tmp_path / "pipeline-state.json")
    with pytest.raises(RuntimeError, match="canonical"):
        gated.require_phase_gate("test", tmp_path / "other.json")


def test_gate_rejects_duplicate_json_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(gated, "ROOT", tmp_path)
    state = tmp_path / "pipeline-state.json"
    state.write_text('{"stages":[],"stages":[]}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="strict unambiguous"):
        gated.require_phase_gate("test", state)


def test_real_cpu_forecastmodel_two_origin_full_frozen_replay(tmp_path, monkeypatch):
    # A real PreparedData -> train-built ForecastModel -> immutable freeze ->
    # metadata-only preflight -> normal load/normalization -> engine predict path.
    scratch = gated.ROOT / "logs" / f".c06-v3-integration-{uuid.uuid4().hex}"
    train_dir = scratch / "train"
    target_dir = scratch / "target"
    checkpoint_dir = scratch / "checkpoint"
    out = scratch / "runs" / "toy"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(gated, "ROOT", scratch)
    monkeypatch.setattr(gated, "STATE_PATH", scratch / "pipeline-state.json")
    try:
        train = _data(seed=10, partition="train")
        test = _data(seed=20, partition="test")
        _save_data(train_dir / "train_seed10.npz", train)
        _save_data(target_dir / "test_seed20.npz", test)
        _state(scratch / "pipeline-state.json")
        evidence = scratch / "validation-selection.json"
        evidence.write_text('{"toy":true}', encoding="utf-8")

        config = {"variant": "V1", "dataset_path": str(train_dir), "seed": 10,
                  "model": {"d_total": 16, "encoder_width": 4, "summary_length": 2,
                            "candidate_cap": 4, "k": 2}}
        train_data = engine.datasets_at(train_dir, "train")
        stats = engine.task_statistics(train_data[0])
        model = engine.build_model(config, train_data[0], stats)
        provenance = {"source_sha256": engine.source_manifest(),
            "dependency_source_sha256": engine.dependency_manifest(config["variant"]),
            "dataset_path": str(train_dir.resolve()), "model_statistics_train_only": stats,
            "train_windows": len(engine.index_origins(train_data, "train")),
            "val_windows": 1, "scope": "full_protocol"}
        payload = {"config": config, "task": {"n_nodes": 2, "n_features": 1,
            "history": 3, "hmax": 2, "count_data": False},
            "statistics": stats, "train_activity": [1.0, 2.0],
            "state_dict": model.state_dict(), "provenance": provenance}
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / "checkpoint.pt"
        torch.save(payload, checkpoint)
        policy = Policy(kind="full")
        recipe = {"schema_version": 1, "seed": 2027,
            "missing": {"kind": "time_block", "rate": 0.3},
            "count_noise_max_abs": 0, "event_time_jitter_bins": 0}
        domain_path = scratch / "domain.json"
        domain_preflight.write_domain_manifest_exclusive(domain_path, target_dir, "test")

        fake_sources = {rel: "b" * 64 for rel in gated._SOURCE_PATHS}
        monkeypatch.setattr(gated, "_source_hashes", lambda _root=scratch: fake_sources.copy())
        extras = gated.build_freeze_extras(checkpoint=checkpoint, ckpt=payload,
            policy=policy, datasets=None, dataset_path=target_dir,
            domain_manifest_path=domain_path, recipe=recipe, split="test", root=scratch)
        engine.freeze_checkpoint(checkpoint, {"full": policy}, [evidence], 3, 0.1, extra=extras)

        budgets = []
        def budget(*_args):
            budgets.append(1)
            return {"coverage_complete": True, "spent_seconds": 0.0}
        result = gated.evaluate_frozen(checkpoint=checkpoint, dataset_path=target_dir,
            domain_manifest_path=domain_path, recipe=recipe, output_dir=out,
            split="test", policy=policy, device_name="cpu", root=scratch,
            state_path=scratch / "pipeline-state.json", process_scan=lambda: [], budget_fn=budget)
        with np.load(out / "predictions.npz", allow_pickle=False) as saved:
            assert saved["predictions"].shape == (2, 2, 2, 1)
            assert saved["targets"].shape == (2, 2, 2, 1)
            assert saved["origins"].tolist() == [[0, 2], [0, 3]]
            assert int(saved["seed_id"]) == 10
            expected_targets = np.stack([engine.get_target(test, origin)[0] for origin in (2, 3)])
            expected_masks = np.stack([engine.get_target(test, origin)[1] for origin in (2, 3)])
            assert np.array_equal(saved["targets"], expected_targets)
            assert np.array_equal(saved["target_mask"], expected_masks)
        assert result["n_origins"] == 2 and result["all_origins_complete_and_ordered"]
        assert result["weights_unchanged"] and result["optimizer_steps"] == 0
        assert all(row["mode"] == "checkpoint-frozen-synthetic-normalizer"
                   for row in result["normalization_qc"])
        assert len(budgets) >= 3
        receipts = list((scratch / "logs/offline-phase-wall").glob("c06-history-*.json"))
        receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
        assert receipt["status"] == "completed" and receipt["test_accessed"] is True
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _budget_fixture(tmp_path, monkeypatch, tag):
    scratch = gated.ROOT / "logs" / f".c06-v4-{tag}-{uuid.uuid4().hex}"
    train_dir=scratch/"train"; target_dir=scratch/"target"; checkpoint_dir=scratch/"checkpoint"; out=scratch/"runs/toy"
    scratch.mkdir(parents=True); monkeypatch.setattr(gated,"ROOT",scratch); monkeypatch.setattr(gated,"STATE_PATH",scratch/"pipeline-state.json")
    train=_data(seed=10,partition="train"); test=_data(seed=20,partition="test")
    _save_data(train_dir/"train_seed10.npz",train); _save_data(target_dir/"test_seed20.npz",test)
    _state(scratch/"pipeline-state.json")
    evidence=scratch/"validation-selection.json"; evidence.write_text('{"toy":true}',encoding="utf-8")
    config={"variant":"V1","dataset_path":str(train_dir),"seed":10,"model":{"d_total":16,"encoder_width":4,"summary_length":2,"candidate_cap":4,"k":2}}
    td=engine.datasets_at(train_dir,"train")[0]; stats=engine.task_statistics(td); model=engine.build_model(config,td,stats)
    payload={"config":config,"task":{"n_nodes":2,"n_features":1,"history":3,"hmax":2,"count_data":False},"statistics":stats,"train_activity":[1.,2.],"state_dict":model.state_dict(),"provenance":{"source_sha256":engine.source_manifest(),"dependency_source_sha256":engine.dependency_manifest("V1"),"dataset_path":str(train_dir.resolve()),"model_statistics_train_only":stats,"train_windows":len(engine.index_origins([td],"train")),"val_windows":1,"scope":"full_protocol"}}
    checkpoint_dir.mkdir(); checkpoint=checkpoint_dir/"checkpoint.pt"; torch.save(payload,checkpoint)
    policy=Policy(kind="full"); recipe={"schema_version":1,"seed":2027,"missing":{"kind":"time_block","rate":0.3},"count_noise_max_abs":0,"event_time_jitter_bins":0}
    domain=scratch/"domain.json"; domain_preflight.write_domain_manifest_exclusive(domain,target_dir,"test")
    fake={rel:"d"*64 for rel in gated._SOURCE_PATHS}; monkeypatch.setattr(gated,"_source_hashes",lambda _root=scratch:fake.copy())
    extras=gated.build_freeze_extras(checkpoint=checkpoint,ckpt=payload,policy=policy,datasets=None,dataset_path=target_dir,domain_manifest_path=domain,recipe=recipe,split="test",root=scratch)
    engine.freeze_checkpoint(checkpoint,{"full":policy},[evidence],3,.1,extra=extras)
    return scratch,out,checkpoint,target_dir,domain,recipe,policy


def test_prewrite_budget_guard_fails_before_first_artifact(tmp_path, monkeypatch):
    scratch,out,checkpoint,target_dir,domain,recipe,policy=_budget_fixture(tmp_path,monkeypatch,"prewrite")
    def budget(_root,_output,_receipt_path,receipt,_started):
        # State-based boundary: all origins finished, output tree has not been created.
        if receipt.get("origins_completed")==2 and not out.exists():
            raise RuntimeError("synthetic budget exhausted before first artifact")
        return {"coverage_complete":True,"spent_seconds":0.0}
    with pytest.raises(RuntimeError,match="before first artifact"):
        gated.evaluate_frozen(checkpoint=checkpoint,dataset_path=target_dir,domain_manifest_path=domain,recipe=recipe,output_dir=out,split="test",policy=policy,device_name="cpu",root=scratch,state_path=scratch/"pipeline-state.json",process_scan=lambda:[],budget_fn=budget)
    assert not out.exists()
    receipt=json.loads(next((scratch/"logs/offline-phase-wall").glob("c06-history-*.json")).read_text())
    assert receipt["status"]=="failed" and receipt.get("origins_completed")==2


def test_postserialization_budget_guard_fails_before_completed_receipt(tmp_path, monkeypatch):
    scratch,out,checkpoint,target_dir,domain,recipe,policy=_budget_fixture(tmp_path,monkeypatch,"postwrite")
    def budget(_root,_output,_receipt_path,_receipt,_started):
        if (out/"predictions.npz").is_file() and (out/"perturbation-manifest.json").is_file():
            raise RuntimeError("synthetic cumulative budget exhausted after serialization")
        return {"coverage_complete":True,"spent_seconds":0.0}
    with pytest.raises(RuntimeError,match="after serialization"):
        gated.evaluate_frozen(checkpoint=checkpoint,dataset_path=target_dir,domain_manifest_path=domain,recipe=recipe,output_dir=out,split="test",policy=policy,device_name="cpu",root=scratch,state_path=scratch/"pipeline-state.json",process_scan=lambda:[],budget_fn=budget)
    assert (out/"predictions.npz").is_file() and (out/"perturbation-manifest.json").is_file()
    receipt=json.loads(next((scratch/"logs/offline-phase-wall").glob("c06-history-*.json")).read_text())
    assert receipt["status"]=="failed" and receipt.get("origins_completed")==2


def test_periodic_c06_freeze_fields_bind_helper_and_acceptance_exact_bytes(tmp_path):
    import json
    import shutil
    from blind_reference_cost_policy_v1 import descriptor
    scratch=tmp_path/"run"; scratch.mkdir()
    helper=scratch/"edsgn/staging/blind_reference_cost_policy_v1.py"
    helper.parent.mkdir(parents=True); shutil.copyfile(ROOT/"edsgn/staging/blind_reference_cost_policy_v1.py",helper)
    acceptance=scratch/"reviews/blind-reference-cost-policy-root-acceptance-v1.json"
    acceptance.parent.mkdir()
    acceptance.write_text(json.dumps({"accepted_by":"root","verdict":"pass","scope":"engineering-staging-only",
        "scientific_hypothesis_accepted":False,"pipeline_advanced":False,
        "evidence_sha256":{"edsgn/staging/blind_reference_cost_policy_v1.py":gated.sha256(ROOT/"edsgn/staging/blind_reference_cost_policy_v1.py")}}),encoding="utf-8")
    fields=gated._blind_policy_fields(Policy(kind="periodic"),scratch)
    assert fields["blind_reference_cost_policy_descriptor"] == descriptor()
    assert fields["blind_reference_cost_policy_acceptance_sha256"] == gated.sha256(acceptance)
    assert fields["blind_reference_cost_policy_acceptance_scope"] == "engineering-staging-only"
    acceptance.write_text("{}",encoding="utf-8")
    with pytest.raises(RuntimeError,match="root engineering-only acceptance"):
        gated._blind_policy_fields(Policy(kind="periodic"),scratch)
