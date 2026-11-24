"""Mock-only tests for the RP integration script's preflight gates."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


SCRIPT = Path(__file__).with_name("integrate_rp_baselines_v1.py")
SPEC = importlib.util.spec_from_file_location("rp_integration_gates", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _state(p1b="passed", p2="pending", p3="pending", backedge=True):
    history = []
    if backedge:
        history.append({"action": "reopen", "stage": "P2",
                        "reason": "explicit root backedge to P2 before P3"})
    return {
        "stages": [{"id": "P1B", "status": p1b}, {"id": "P2", "status": p2},
                   {"id": "P3", "status": p3}],
        "history": history,
    }


def _stage_root(tmp_path, monkeypatch, state):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "protocols").mkdir()
    (tmp_path / "protocols/p2-engineering-delta-gate-v2.md").write_text(
        "No unrun matrix cell is passed here. T19/T25 remain downstream.", encoding="utf-8")
    (tmp_path / "protocols/p3-training-resource-contract-v2.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state), encoding="utf-8")


def test_p2_passed_p3_running_is_rejected(tmp_path, monkeypatch):
    _stage_root(tmp_path, monkeypatch, _state(p2="passed", p3="running"))
    with pytest.raises(module.PreflightError, match="P1B-passed/P2-pending/P3-pending"):
        module._pending_stage_check()


def test_root_backedge_p1b_passed_p2_pending_p3_pending_is_accepted(tmp_path, monkeypatch):
    _stage_root(tmp_path, monkeypatch, _state())
    result = module._pending_stage_check()
    assert result["stage_statuses"] == {"P1B": "passed", "P2": "pending", "P3": "pending"}
    assert result["explicit_root_backedge"] is True


def test_missing_or_malformed_canonical_state_fails_closed(tmp_path, monkeypatch):
    _stage_root(tmp_path, monkeypatch, _state(backedge=False))
    with pytest.raises(module.PreflightError, match="explicit root backedge"):
        module._pending_stage_check()
    (tmp_path / "pipeline-state.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(module.PreflightError, match="malformed"):
        module._pending_stage_check()


class _Proc:
    def __init__(self, pid, exe, cmdline, cwd):
        self.info = {"pid": pid, "name": exe, "exe": exe, "cmdline": cmdline, "cwd": cwd}


def _mock_psutil(monkeypatch, procs):
    fake = types.SimpleNamespace(
        process_iter=lambda _attrs: iter(procs),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}),
        ZombieProcess=type("ZombieProcess", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)


def test_python_module_train_with_run_cwd_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    _mock_psutil(monkeypatch, [_Proc(420, "python.exe", ["python", "-m", "edsgn.train"], str(tmp_path))])
    with pytest.raises(module.PreflightError, match="owned Python model/runner/resource process"):
        module._process_barrier()


def test_relative_runner_with_run_cwd_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    _mock_psutil(monkeypatch, [_Proc(421, "python.exe", ["python", "edsgn/scripts/baseline_campaign.py"], str(tmp_path))])
    with pytest.raises(module.PreflightError, match="baseline_campaign.py"):
        module._process_barrier()


def test_foreign_quant_python_with_run_cwd_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    _mock_psutil(monkeypatch, [_Proc(422, "python.exe", ["python", "quantizer.py"], str(tmp_path))])
    assert module._process_barrier() == []


def test_access_denied_python_process_fails_closed(monkeypatch):
    access_denied = type("AccessDenied", (Exception,), {})

    class InaccessiblePythonProc:
        @property
        def info(self):
            raise access_denied("cmdline unavailable")

        def name(self):
            return "python.exe"

    monkeypatch.setattr(module, "ROOT", Path("C:/run-root"))
    fake = types.SimpleNamespace(
        process_iter=lambda _attrs: iter([InaccessiblePythonProc()]),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        AccessDenied=access_denied,
        ZombieProcess=type("ZombieProcess", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    with pytest.raises(module.PreflightError, match="failing closed"):
        module._process_barrier()


def test_existing_campaign_lock_is_rejected_without_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    lock = tmp_path / "logs/offline-phase-wall/campaign.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("owner", encoding="utf-8")
    _mock_psutil(monkeypatch, [])
    with pytest.raises(module.PreflightError, match="do not remove it automatically"):
        module._process_barrier()
    assert lock.read_text(encoding="utf-8") == "owner"


def test_existing_model_resource_lock_is_rejected_without_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    lock = tmp_path / "model-resource.lock"
    lock.write_text("resource owner", encoding="utf-8")
    _mock_psutil(monkeypatch, [])
    with pytest.raises(module.PreflightError, match="do not remove it automatically"):
        module._process_barrier()
    assert lock.read_text(encoding="utf-8") == "resource owner"


def test_candidate_probe_log_model_resource_lock_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    lock = tmp_path / "logs/model-resource.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("candidate probe owner", encoding="utf-8")
    _mock_psutil(monkeypatch, [])
    with pytest.raises(module.PreflightError, match="do not remove it automatically"):
        module._process_barrier()
    assert lock.read_text(encoding="utf-8") == "candidate probe owner"
