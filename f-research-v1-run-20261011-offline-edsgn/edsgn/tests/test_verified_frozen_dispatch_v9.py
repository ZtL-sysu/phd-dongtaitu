"""CPU-only dispatcher9 ordinary-admission integration fixtures."""
from __future__ import annotations
import json, sys, uuid, importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

RUN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN / "edsgn/tests"))
sys.path.insert(0, str(RUN / "edsgn/staging"))
sys.path.insert(0, str(RUN / "edsgn/scripts"))
from test_ordinary_frozen_test_admission_v1 import complete_fixture, put, sha
import verified_frozen_dispatch_v9 as dispatcher


def ready_fixture(tmp_path, monkeypatch):
    root, binding_ref, job, job_path, matrix, binder_copy = complete_fixture(tmp_path)
    out = root / matrix["jobs"][0]["output_dir"]
    out.parent.mkdir(parents=True, exist_ok=True)
    (root / "runs/evidence").mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("temp_dispatcher_v9", root / "edsgn/scripts/verified_frozen_dispatch_v9.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = root
    module.VALIDATOR = root / "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py"
    import bind_training_inputs as binder
    monkeypatch.setattr(binder, "ROOT", root)
    monkeypatch.setattr(binder, "__file__", str(binder_copy))
    return root, binding_ref, job, job_path, matrix, out, module


def toy_runner(root, *, mutate=None, stale_receipt=False):
    calls = {"validator": 0, "child": 0}
    def runner(cmd, **kwargs):
        cmd = [str(x) for x in cmd]
        if Path(cmd[1]).name == "pipeline_state.py":
            calls["validator"] += 1
            if mutate == "source-add-before-child" and calls["validator"] == 1:
                (root / "datasets/email/new-sidecar.json").write_text('{"late":true}', encoding="utf-8")
            if mutate == "source-edit-before-child" and calls["validator"] == 1:
                (root / "datasets/email/opaque.npz").write_bytes(b"modified toy source")
            if mutate == "input-after-child" and calls["validator"] == 2:
                (root / "datasets/email/opaque.npz").write_bytes(b"mutated toy input")
            if mutate == "output-after-child" and calls["validator"] == 2:
                (root / "runs/test/email/STID/seed0/predictions.npz").unlink(missing_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        calls["child"] += 1
        args = cmd[2:]
        flags = dict(zip(args[::2], args[1::2]))
        output = root / flags["--output"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "predictions.npz").write_bytes(b"toy prediction payload; never decoded")
        (output / "counter-rows.json").write_text("{}", encoding="utf-8")
        (output / "trace-rows.jsonl.gz").write_bytes(b"toy trace bytes")
        (output / "metrics_by_horizon.csv").write_text("horizon,mae,rmse,count\n1,0,0,1\n", encoding="utf-8")
        (output / "frozen-graph-state-diagnostic.json").write_text(json.dumps({
            "scope": "frozen_test_replay_oracle_free", "split": "test", "full_test_origin_count": 2,
            "predictions_file": "predictions.npz", "oracle_accessed": False,
            "oracle_npz_decoded": False, "oracle_analysis_status": "deferred_until_global_completion",
        }), encoding="utf-8")
        phase_id = "toy-child-" + uuid.uuid4().hex
        phase = {"scope": "offline-experiment-phase", "job_id": phase_id,
                 "phase": "evaluation", "status": "failed" if stale_receipt else "completed",
                 "returncode": 0, "output_dir": str(output.resolve()), "origin_count": 2}
        p = root / "logs/offline-phase-wall" / f"{phase_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        put(p, phase)
        return SimpleNamespace(returncode=0, stdout="toy-child-ok", stderr="")
    return runner, calls


def test_dispatcher9_public_entry_ordinary_binding_global_queue_and_random_receipt_alias(tmp_path, monkeypatch):
    root, binding_ref, job, job_path, matrix, out, dispatcher = ready_fixture(tmp_path, monkeypatch)
    runner, calls = toy_runner(root)
    result = dispatcher.dispatch(job_path, root=root, runner=runner)
    assert result["status"] == "accepted-output-pending-scientific-review"
    assert calls == {"validator": 2, "child": 1}
    assert result["child_phase_receipt"]["job_id"].startswith("toy-child-")
    alias = root / job["phase_receipt_alias"]
    original = root / result["child_phase_receipt"]["path"]
    assert alias.read_bytes() == original.read_bytes()
    assert result["before"]["files"][binding_ref["path"]] == binding_ref["sha256"]
    assert "protocols/ordinary-matrix.json" in result["before"]["files"]
    assert "datasets/email/opaque.npz" in result["before"]["files"]


@pytest.mark.parametrize("mutation", ["argv", "source", "alias", "binding-mix", "config", "seed",
                                      "checkpoint", "dataset", "policy", "output", "missing-matrix-gate"])
def test_dispatcher9_rejects_ordinary_binding_mismatch_before_child(tmp_path, monkeypatch, mutation):
    root, binding_ref, job, job_path, matrix, _, dispatcher = ready_fixture(tmp_path, monkeypatch)
    if mutation == "argv":
        job["argv"][job["argv"].index("full")] = "random"
    elif mutation == "source":
        job["sources"][0]["sha256"] = "0" * 64
    elif mutation == "alias":
        job["phase_receipt_alias"] = "runs/evidence/wrong-alias.json"
    elif mutation in {"config", "seed"}:
        config_path = root / "configs/stid.json"
        cfg = json.loads(config_path.read_text())
        cfg["training"] = {"epochs": 2} if mutation == "config" else cfg["training"]
        cfg["seed"] = 1 if mutation == "seed" else cfg["seed"]
        put(config_path, cfg)
    elif mutation in {"checkpoint", "dataset", "policy", "output"}:
        flag = {"checkpoint": "--checkpoint", "dataset": "--dataset-path",
                "policy": "--policy", "output": "--output"}[mutation]
        index = job["argv"].index(flag) + 1
        job["argv"][index] = {
            "checkpoint": "runs/train/email/STID/seed0/other.pt",
            "dataset": "datasets/mooc",
            "policy": "random",
            "output": "runs/test/email/STID/other-output",
        }[mutation]
    elif mutation == "missing-matrix-gate":
        state_path = root / "pipeline-state.json"
        state = json.loads(state_path.read_text())
        p3a = next(row for row in state["stages"] if row["id"] == "P3A")
        p3a["evidence"] = []
        put(state_path, state)
    elif mutation == "binding-mix":
        other = dict(binding_ref)
        other["path"] = "runs/evidence/other-primary.json"
        doc = json.loads((root / binding_ref["path"]).read_text())
        doc["scope"] = "pretest_execution_binding"
        put(root / other["path"], doc)
        job["pretest_execution_bindings"].append({"path": other["path"], "sha256": sha(root / other["path"])})
    put(job_path, job)
    runner, calls = toy_runner(root)
    with pytest.raises(dispatcher.DispatchError):
        dispatcher.dispatch(job_path, root=root, runner=runner)
    assert calls["child"] == 0


@pytest.mark.parametrize("tamper", ["input-after-child", "output-after-child", "stale-receipt"])
def test_dispatcher9_final_fences_reject_input_output_or_stale_receipt(tmp_path, monkeypatch, tamper):
    root, _, _, job_path, _, _, dispatcher = ready_fixture(tmp_path, monkeypatch)
    runner, calls = toy_runner(root, mutate=None if tamper == "stale-receipt" else tamper,
                               stale_receipt=tamper == "stale-receipt")
    with pytest.raises(dispatcher.DispatchError):
        dispatcher.dispatch(job_path, root=root, runner=runner)
    assert calls["child"] == 1


@pytest.mark.parametrize("mutation", ["source-add-before-child", "source-edit-before-child"])
def test_dispatcher9_rechecks_source_tree_after_official_gate_before_child(tmp_path, monkeypatch, mutation):
    root, _, _, job_path, _, _, dispatcher = ready_fixture(tmp_path, monkeypatch)
    runner, calls = toy_runner(root, mutate=mutation)
    with pytest.raises(dispatcher.DispatchError):
        dispatcher.dispatch(job_path, root=root, runner=runner)
    assert calls == {"validator": 1, "child": 0}
