import hashlib
import json
from pathlib import Path
import types

import pytest

import count_loss_training_controller_v1 as controller


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def phase_root(tmp_path):
    proof = write_json(tmp_path / "proof.json", {"frozen": True})
    sha = controller.sha256(proof)
    write_json(tmp_path / "pipeline-state.json", {"stages": [
        {"id": "P2", "status": "passed", "gate_verdict": "pass", "inputs": [], "evidence": []},
        {"id": "P3", "status": "passed", "gate_verdict": "pass", "inputs": [], "evidence": []},
        {"id": "P3A", "status": "passed", "gate_verdict": "pass", "inputs": [{"path": "proof.json", "sha256": sha}], "evidence": []},
        {"id": "P4", "status": "running"},
    ]})
    return proof


def success_validator(*args, **kwargs):
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def test_official_before_p4_checks_canonical_state_and_hash_pins(tmp_path):
    phase_root(tmp_path)
    snapshot = controller.official_before_p4(tmp_path, run=success_validator)
    assert str(tmp_path / "pipeline-state.json") in snapshot
    assert str(tmp_path / "proof.json") in snapshot


def test_before_p4_refuses_bad_verdict_or_missing_evidence_hash(tmp_path):
    proof = phase_root(tmp_path)
    state = json.loads((tmp_path / "pipeline-state.json").read_text())
    state["stages"][2]["gate_verdict"] = "pending"
    write_json(tmp_path / "pipeline-state.json", state)
    with pytest.raises(controller.ControllerError, match="P3A must be passed/pass"):
        controller.official_before_p4(tmp_path, run=success_validator)
    phase_root(tmp_path)
    (tmp_path / "proof.json").write_text("changed", encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="hash-mismatched"):
        controller.official_before_p4(tmp_path, run=success_validator)


def test_official_validator_cannot_mutate_canonical_state_during_check(tmp_path):
    phase_root(tmp_path)
    def mutating_validator(*args, **kwargs):
        state = json.loads((tmp_path / "pipeline-state.json").read_text())
        state["mutation"] = True
        state["stages"][2]["inputs"] = [{"path": "proof.json", "sha256": controller.sha256(tmp_path / "proof.json")}]
        write_json(tmp_path / "pipeline-state.json", state)
        return success_validator()
    with pytest.raises(controller.ControllerError, match="changed during official"):
        controller.official_before_p4(tmp_path, run=mutating_validator)


def test_private_bound_run_queue_preserves_module_and_changes_only_training_command():
    import bound_phase_campaign as bound
    original = bound.run_queue.__globals__["command_for"]
    runner, private, _ = controller._private_bound_controller()
    assert runner.__globals__ is private
    assert bound.run_queue.__globals__["command_for"] is original
    assert private["command_for"]("training", ["--config", "cfg.json"]) == [
        str(controller.PYTHON), str(controller.ADAPTER), "--config", "cfg.json"]
    with pytest.raises(controller.ControllerError, match="only accepts training"):
        private["command_for"]("evaluation", ["--output", "x"])


def test_real_prepared_count_config_is_not_launchable_without_root_selected_policy(tmp_path):
    root = controller.ROOT
    config_path = root / "configs/count-loss-v1/email-V4-Poisson-seed0.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pool_path = root / controller.EXPECTED_POOL_PATHS["email"]
    job = {
        "id": "email-count", "phase": "training", "dataset": "email", "seed": 0,
        "variant": "V4", "scope": "formal", "args": ["--config", str(config_path)],
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": controller.sha256(config_path)},
        "output_dir": config["output_dir"],
        "candidate_pool": {"path": pool_path.relative_to(root).as_posix(), "sha256": controller.sha256(pool_path)},
        "candidate_selection": {"path": "runs/formal-candidate-selection-v2/absent-policy.json", "sha256": "0" * 64},
        "static_blueprint": {"path": "protocols/count-loss-v1/email-V4-Poisson-seed0-static-input-blueprint-v3.json",
                             "sha256": controller.sha256(root / "protocols/count-loss-v1/email-V4-Poisson-seed0-static-input-blueprint-v3.json")},
        "mse_numeric_acceptance": {"path": "missing-numeric.json", "sha256": "0" * 64},
        "mse_numeric_root_acceptance": {"path": "missing-root-numeric.json", "sha256": "0" * 64},
        "inputs": [],
    }
    with pytest.raises(controller.ControllerError, match="missing or invalid path"):
        controller._validate_job(root, job, {})
    assert not Path(config["output_dir"]).joinpath("input-binding-at-launch.json").exists()


def test_controller_rejects_noncanonical_adapter_command_before_policy_lookup():
    root = controller.ROOT
    config_path = root / "configs/count-loss-v1/email-V4-Poisson-seed0.json"
    job = {"id": "email-count", "phase": "training", "dataset": "email", "seed": 0,
           "variant": "V4", "scope": "formal", "args": ["-m", "edsgn.train"],
           "config": {"path": config_path.relative_to(root).as_posix(), "sha256": controller.sha256(config_path)},
           "output_dir": "runs/no-op", "candidate_pool": {}, "candidate_selection": {},
           "static_blueprint": {}, "mse_numeric_acceptance": {},
           "mse_numeric_root_acceptance": {}, "inputs": []}
    with pytest.raises(controller.ControllerError, match="exactly adapter --config"):
        controller._validate_job(root, job, {})


def _mock_queue(tmp_path):
    queue_path = write_json(tmp_path / "queue.json", {"prepared": True})
    return queue_path


def test_incomplete_aggregate_refuses_before_any_lease_or_child(tmp_path, monkeypatch):
    queue_path = _mock_queue(tmp_path)
    monkeypatch.setattr(controller, "validate_queue", lambda root, path: ({"jobs": []}, {str(queue_path): controller.sha256(queue_path)}))
    monkeypatch.setattr(controller, "official_before_p4", lambda *a, **k: {})
    monkeypatch.setattr(controller, "_owned_processes", lambda root: [])
    monkeypatch.setattr(controller, "_private_bound_controller", lambda: pytest.fail("must not build runner"))
    with pytest.raises(controller.ControllerError, match="incomplete or exhausted"):
        controller.run_queue(queue_path, root=tmp_path, aggregate_fn=lambda root: {"coverage_complete": False, "spent_seconds": 1})
    assert not (tmp_path / "logs" / "model-resource.lock").exists()
    assert not (tmp_path / "logs" / "offline-phase-wall" / "campaign.lock").exists()


def test_existing_resource_lease_is_preserved_and_never_treated_as_permission(tmp_path, monkeypatch):
    queue_path = _mock_queue(tmp_path)
    lease = tmp_path / "logs" / "model-resource.lock"
    lease.parent.mkdir(parents=True)
    lease.write_text("foreign lease", encoding="utf-8")
    monkeypatch.setattr(controller, "validate_queue", lambda root, path: ({"jobs": []}, {str(queue_path): controller.sha256(queue_path)}))
    monkeypatch.setattr(controller, "official_before_p4", lambda *a, **k: {})
    monkeypatch.setattr(controller, "_owned_processes", lambda root: [])
    with pytest.raises(controller.ControllerError, match="lease already exists"):
        controller.run_queue(queue_path, root=tmp_path, aggregate_fn=lambda root: {"coverage_complete": True, "spent_seconds": 0})
    assert lease.read_text(encoding="utf-8") == "foreign lease"


def test_child_is_fenced_immediately_before_spawn_and_owned_lease_released(tmp_path, monkeypatch):
    queue_path = _mock_queue(tmp_path)
    phase_calls = []
    monkeypatch.setattr(controller, "validate_queue", lambda root, path: ({"jobs": []}, {str(queue_path): controller.sha256(queue_path)}))
    def official(*args, **kwargs):
        phase_calls.append("official")
        return {"pipeline-state.json": "frozen"}
    monkeypatch.setattr(controller, "official_before_p4", official)
    monkeypatch.setattr(controller, "_owned_processes", lambda root: [])
    child_calls = []
    fake_subprocess = types.SimpleNamespace(Popen=lambda command, *a, **k: child_calls.append(command) or "child")
    private = {"subprocess": fake_subprocess, "command_input_paths": lambda job: set()}
    def mock_runner(path):
        return private["subprocess"].Popen([str(controller.PYTHON), str(controller.ADAPTER), "--config", "config.json"])
    monkeypatch.setattr(controller, "_private_bound_controller", lambda: (mock_runner, private, fake_subprocess))
    result = controller.run_queue(
        queue_path, root=tmp_path,
        aggregate_fn=lambda root, **kw: {"coverage_complete": True, "spent_seconds": 5})
    assert result == "child"
    assert len(phase_calls) == 2  # initial official check and immediately pre-child
    assert child_calls == [[str(controller.PYTHON), str(controller.ADAPTER), "--config", "config.json"]]
    assert not (tmp_path / "logs" / "model-resource.lock").exists()
