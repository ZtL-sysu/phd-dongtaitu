from __future__ import annotations

import sys
from pathlib import Path

import pytest

RUN = Path(r"C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn")
sys.path.insert(0, str(RUN / "edsgn/scripts"))
import verified_frozen_dispatch_v8 as dispatch


def _job():
    return {"driver": "h2_v4", "phase_receipt_alias": "runs/x/evidence/alias.json",
            "sources": []}


def _argv(checkpoint="runs/left/seed0/checkpoint.pt"):
    return ["--checkpoint", checkpoint, "--dataset-path", "data",
            "--output", "runs/left/seed0/output", "--device", "cpu"]


def test_h2_test_driver_without_binding_is_rejected_even_without_split_flag(tmp_path):
    with pytest.raises(dispatch.DispatchError, match="test execution has no typed endpoint"):
        dispatch._validate_job_binding(_job(), _argv(), [], tmp_path)


def test_h2_test_driver_rejects_wrong_checkpoint_against_binding(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "runs/left/seed0").mkdir(parents=True)
    domain = tmp_path / "domain.json"
    domain.write_text("{}", encoding="utf-8")
    plan = {"schema_version": 3, "outcome_type": "forecast_mae_superiority",
            "domain": {"split": "test", "path": "domain.json"},
            "dataset_collection": {"root": "data", "files": []},
            "left": {"producer_source_sha256": "a" * 64, "seeds": [
                {"seed_id": 0, "planned_paths": {"checkpoint": "runs/left/seed0/checkpoint.pt"}}]},
            "right": {"producer_source_sha256": "b" * 64, "seeds": []},
            "producer_sources": []}
    run = {"side": "left", "seed_id": 0,
           "checkpoint": {"path": "runs/left/seed0/checkpoint.pt"}}
    binding = {"_verified_plan_doc": plan, "runs": [run]}
    with pytest.raises(dispatch.DispatchError, match="test job argv/source/output do not match"):
        dispatch._validate_job_binding(_job(), _argv("runs/other/checkpoint.pt"), [binding], tmp_path)


def _graph8_child(root, report_flags, name="seed0"):
    output=root/f"runs/graph/{name}";output.mkdir(parents=True)
    for artifact_name in ("predictions.npz","counter-rows.json","trace-rows.jsonl.gz","metrics_by_horizon.csv"):
        (output/artifact_name).write_bytes(b"toy")
    report={"predictions_file":"predictions.npz","scope":"frozen_test_replay_oracle_free","split":"test",
            "full_test_origin_count":2,"oracle_accessed":False,"oracle_npz_decoded":False,
            "oracle_analysis_status":"deferred_until_global_completion",**report_flags}
    (output/"frozen-graph-state-diagnostic.json").write_text(__import__("json").dumps(report),encoding="utf-8")
    phase_dir=root/"logs/offline-phase-wall";phase_dir.mkdir(parents=True,exist_ok=True)
    phase={"scope":"offline-experiment-phase","job_id":f"child-graph-{name}","output_dir":str(output.resolve()),"phase":"evaluation","status":"completed","origin_count":2,
           "outputs":{p.name:dispatch.sha(p) for p in output.iterdir() if p.is_file()}}
    (phase_dir/(phase["job_id"]+".json")).write_text(__import__("json").dumps(phase),encoding="utf-8")
    return output,phase_dir


def test_graph8_completion_requires_oracle_free_deferred_scope(tmp_path):
    output,phase_dir=_graph8_child(tmp_path,{})
    verified=dispatch._verify_child_completion("graph_v8",output,set(),phase_dir,tmp_path)
    assert verified["job_id"]=="child-graph-seed0"
    output,phase_dir=_graph8_child(tmp_path,{"oracle_accessed":True},"seed1")
    with pytest.raises(dispatch.DispatchError,match="oracle-free/deferred"):
        dispatch._verify_child_completion("graph_v8",output,set(),phase_dir,tmp_path)
