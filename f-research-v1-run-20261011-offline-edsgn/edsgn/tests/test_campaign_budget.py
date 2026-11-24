import json
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import aggregate_budget
import baseline_campaign as campaign


def test_episode_campaign_config_uses_dedicated_data_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    (cfgdir / "email-GRU.json").write_text(json.dumps({"dataset_path": "old-email.npz", "variant": "GRU", "output_dir": "old"}), encoding="utf-8")
    cfg = campaign.load_config("email-episode", "GRU", {"GRU": {"email-episode": 4}})
    assert cfg["dataset_path"] == str(tmp_path / "data/processed/email-episode-v1/email-episode.npz")
    assert cfg["output_dir"] == str(tmp_path / "runs/baselines/email-episode/GRU/seed0")
    assert cfg["batch_size"] == 4


def test_resource_config_is_exclusive_and_exactly_reusable(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    p = campaign.config_path("synthetic", "GRU")
    cfg = {"batch_size": 128}
    campaign.write_config_immutable(p, cfg)
    original = p.read_bytes()
    campaign.write_config_immutable(p, cfg)
    assert p.read_bytes() == original
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        campaign.write_config_immutable(p, {"batch_size": 16})
    assert p.read_bytes() == original


def test_explicit_resource_device_and_training_cache_only_apply_to_core(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    resource = tmp_path / "resource.json"
    monkeypatch.setattr(campaign, "CONTRACT", resource)
    resource.write_text(json.dumps({"devices": {"V1": {"email": "cpu"}},
                                    "training_prior_cache": True,
                                    "training_prior_cache_bytes": 123456}), encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/email-GRU.json").write_text(json.dumps({"model": {"k": 8}}), encoding="utf-8")
    core = campaign.load_config("email", "V1", {"V1": {"email": 8}})
    control = campaign.load_config("email", "GRU", {"GRU": {"email": 8}})
    assert core["device"] == "cpu"
    assert core["model"]["training_prior_cache"] is True
    assert core["model"]["training_prior_cache_bytes"] == 123456
    assert control["device"] == "cuda"
    assert "training_prior_cache" not in control["model"]


def test_queue_failure_time_deduplicates_against_unsummarized_log(tmp_path):
    out = tmp_path / "runs/baselines/synthetic/GRU/seed0"
    out.mkdir(parents=True)
    (out / "training_log.csv").write_text("epoch,elapsed_seconds\n1,12.5\n", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "baseline-campaign-queue.json").write_text(json.dumps({"runs": [
        {"dataset": "synthetic", "method": "GRU", "seed": 0, "output_dir": str(out), "status": "failed", "training_wall_seconds": 15.0}
    ]}), encoding="utf-8")
    rows, outputs, _ = aggregate_budget._training_time(tmp_path)
    assert outputs[str(out.resolve()).casefold()] == 15.0
    assert sum(r["seconds"] for r in rows) == 15.0


def test_resource_v2_probes_and_diagnostic_timestamps_are_accounted(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "p3-resource-v2-synthetic-GRU.json").write_text(json.dumps({"wall_seconds": 4.5, "scope": "single-real-training-batch-resource-probe"}), encoding="utf-8")
    (logs / "p3-resource-v2-email-V1-B24-config.json").write_text(json.dumps({"wall_seconds": 99}), encoding="utf-8")
    (logs / "cuda-resource-diagnostic.json").write_text(json.dumps({"started_utc": "2026-10-11T05:57:41Z", "completed_utc": "2026-10-11T05:58:04Z"}), encoding="utf-8")
    (logs / "odd-name.json").write_text(json.dumps({"scope": "single-real-training-batch-resource-probe", "wall_seconds": 2}), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    probe_rows = [r for r in report["accounting_rows"] if r["kind"].startswith("resource_")]
    assert sum(r["seconds"] for r in probe_rows) == 29.5
    assert report["resource_probe_json_files_seen"] == 3
    assert report["profile_execution_status"] == "not-started"
    assert not any("runs/profile" in g for g in report["coverage_gaps"])


def test_manual_conservative_bound_covers_probe_without_double_count(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    setup = logs / "setup-resource-probe.json"
    setup.write_text(json.dumps({"scope": "random-tensor functionality only"}), encoding="utf-8")
    p3 = logs / "p3-resource-v2-synthetic-GRU.json"
    p3.write_text(json.dumps({"scope": "single-real-training-batch-resource-probe", "wall_seconds": 13}), encoding="utf-8")
    protocols = tmp_path / "protocols"
    protocols.mkdir()
    (protocols / aggregate_budget.MANUAL_LEDGER_NAME).write_text(json.dumps({"entries": [
        {"id": "T26-setup-bound", "description": "T26 and setup engineering allowance", "seconds": 25, "timing_basis": "conservative-bound", "covers_source": [str(setup), str(p3)]}
    ]}), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    assert report["spent_seconds"] == 25
    assert report["measured_seconds"] == 0
    assert report["conservative_bound_seconds"] == 25
    assert report["profile_execution_status"] == "not-started"
    assert not any(str(setup) in g or str(p3) in g for g in report["coverage_gaps"])


def test_resource_functionality_diagnostic_scope_uses_only_time_fields(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "functionality-diagnostic.json").write_text(json.dumps({
        "scope": "random tensor resource/functionality diagnostic; not a scientific benchmark",
        "started_utc": "2026-10-11T05:51:04Z",
        "completed_utc": "2026-10-11T05:53:35Z",
        "environment": {"private": "must not be copied into accounting"},
    }), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    rows = [r for r in report["accounting_rows"] if "diagnostic" in r["kind"]]
    assert len(rows) == 1 and rows[0]["seconds"] == 151
    assert "environment" not in rows[0]


def test_resource_diagnostic_prefers_explicit_whole_process_timer_fields(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "process-timer.json").write_text(json.dumps({
        "scope": "resource/functionality diagnostic",
        "process_wall_seconds": 12.5,
        "whole_command_wall_seconds": 99,
        "elapsed_seconds": 1000,
    }), encoding="utf-8")
    (logs / "whole-command-timer.json").write_text(json.dumps({
        "scope": "resource functionality diagnostic",
        "whole_command_wall_seconds": 17.25,
        "elapsed_seconds": 1000,
    }), encoding="utf-8")
    (logs / "wall-priority.json").write_text(json.dumps({
        "scope": "resource functionality diagnostic",
        "wall_seconds": 3.5,
        "process_wall_seconds": 40,
    }), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    rows = [r for r in report["accounting_rows"] if r["kind"].startswith("resource_")]
    by_source = {Path(r["source"]).name: r for r in rows}
    assert len(rows) == 3
    assert by_source["process-timer.json"]["seconds"] == 12.5
    assert by_source["process-timer.json"]["duration_basis"] == "process_wall_seconds"
    assert by_source["whole-command-timer.json"]["seconds"] == 17.25
    assert by_source["whole-command-timer.json"]["duration_basis"] == "whole_command_wall_seconds"
    assert by_source["wall-priority.json"]["seconds"] == 3.5
    assert by_source["wall-priority.json"]["duration_basis"] == "wall_seconds"
    assert report["spent_seconds"] == 33.25


def test_resource_manual_cover_precedes_explicit_timer_without_double_count(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    probe = logs / "functionality-resource-diagnostic.json"
    probe.write_text(json.dumps({
        "scope": "resource/functionality diagnostic",
        "process_wall_seconds": 40,
    }), encoding="utf-8")
    protocols = tmp_path / "protocols"
    protocols.mkdir()
    (protocols / aggregate_budget.MANUAL_LEDGER_NAME).write_text(json.dumps({"entries": [
        {"id": "T26-explicit-cover", "description": "measured diagnostic duration", "seconds": 40,
         "timing_basis": "measured", "covers_source": [str(probe)]}
    ]}), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    assert report["spent_seconds"] == 40
    assert report["measured_seconds"] == 40
    assert not any(r.get("source") == str(probe) and r["kind"].startswith("resource_") for r in report["accounting_rows"])
    assert sum(r["seconds"] for r in report["accounting_rows"] if r["source"] == str(probe)) == 0


def test_resource_component_elapsed_without_total_or_span_remains_coverage_gap(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "component-only.json").write_text(json.dumps({
        "scope": "resource/functionality diagnostic",
        "elapsed_seconds": 88.0,
        "component_elapsed_seconds": {"forward": 12.0},
    }), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    assert report["spent_seconds"] == 0
    assert not report["coverage_complete"]
    assert any("no valid wall_seconds/process_wall_seconds/whole_command_wall_seconds" in g for g in report["coverage_gaps"])


def test_profile_summary_total_seconds_and_sidecar_deduplicate_replay_samples(tmp_path):
    folder = tmp_path / "runs/profile/runA"
    folder.mkdir(parents=True)
    (folder / "runtime_summary.json").write_text(json.dumps({
        "scope": "full_selected_split",
        "profile_total_wall_seconds": 12,
        "replay_wall_total_ms": {"median": 10000, "p95": 11000, "samples": [10000] * 5},
    }), encoding="utf-8")
    (folder / "profile-budget-accounting.json").write_text(json.dumps({
        "scope": "full-profile-replay", "wall_seconds": 13,
    }), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    rows = [r for r in report["accounting_rows"] if r["kind"] == "profile_total_wall_seconds"]
    assert len(rows) == 1
    assert rows[0]["seconds"] == 13
    assert report["spent_seconds"] == 13
    assert report["profile_execution_status"] == "profile-total-recorded"


def test_profile_runtime_summary_without_total_wall_is_a_coverage_gap(tmp_path):
    folder = tmp_path / "runs/profile/runB"
    folder.mkdir(parents=True)
    (folder / "runtime_summary.json").write_text(json.dumps({
        "scope": "full_selected_split",
        "replay_wall_total_ms": {"median": 10000, "p95": 11000, "samples": [10000] * 5},
    }), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    assert report["profile_execution_status"] == "executed-record-missing-wall-time"
    assert any("profile_total_wall_seconds" in gap for gap in report["coverage_gaps"])
    assert report["spent_seconds"] == 0


def test_queue_process_elapsed_above_training_phase_counts_only_increment(tmp_path):
    out = tmp_path / "runs/baselines/synthetic/GRU/seed0"
    out.mkdir(parents=True)
    (out / "training_summary.json").write_text(json.dumps({"training_wall_seconds": 100}), encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "baseline-campaign-queue.json").write_text(json.dumps({"runs": [{
        "dataset": "synthetic", "method": "GRU", "seed": 0,
        "output_dir": str(out), "status": "completed", "training_wall_seconds": 100,
        "process_wall_seconds": 150,
        "started_utc": "2026-10-11T06:16:20+00:00", "finished_utc": "2026-10-11T06:18:50+00:00",
    }]}), encoding="utf-8")
    rows, outputs, _ = aggregate_budget._training_time(tmp_path)
    assert outputs[str(out.resolve()).casefold()] == 150
    assert sum(row["seconds"] for row in rows) == 150
    adjustments = [row for row in rows if row["kind"] == "queue_process_wall_adjustment"]
    assert len(adjustments) == 1 and adjustments[0]["seconds"] == 50


def test_process_barrier_ignores_powershell_parent_but_catches_python_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    root_text = str(tmp_path).lower()
    class Proc:
        def __init__(self, pid, exe, cmd):
            self.pid = pid
            self.info = {"pid": pid, "create_time": 1000.0, "cmdline": cmd, "cwd": root_text, "exe": exe}
    processes = [
        Proc(101, r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe", ["powershell.exe", "python", "edsgn/scripts/baseline_campaign.py"]),
        Proc(102, r"C:\Python\python.exe", [r"C:\Python\python.exe", "-c", "print('edsgn/scripts/baseline_campaign.py')"]),
        Proc(103, r"C:\Python\python.exe", [r"C:\Python\python.exe", "edsgn/scripts/baseline_campaign.py"]),
        Proc(104, r"C:\Python\python.exe", [r"C:\Python\python.exe", "-m", "edsgn.train"]),
    ]
    fake_psutil = types.SimpleNamespace(
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}),
        process_iter=lambda attrs: iter(processes),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(campaign.os, "getpid", lambda: 100)
    active = campaign._is_training_process()
    assert [r["pid"] for r in active] == [103, 104]


def test_simple_baseline_named_summaries_cover_episode_and_deduplicate_outputs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    same = str(tmp_path / "runs/baselines/email/simple")
    episode = str(tmp_path / "runs/baselines/email-episode/simple")
    (logs / "p3-simple-baseline-summary.json").write_text(json.dumps([
        {"dataset": "email", "output": same, "wall_seconds": 5},
    ]), encoding="utf-8")
    (logs / "p3-simple-baseline-email-episode-summary.json").write_text(json.dumps([
        {"dataset": "email", "output": same, "wall_seconds": 9},
        {"dataset": "email-episode", "output": episode, "wall_seconds": 3},
    ]), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    rows = [r for r in report["accounting_rows"] if r["kind"] == "simple_baseline_wall_seconds"]
    assert len(rows) == 2
    assert sum(r["seconds"] for r in rows) == 12
    assert {r["output_dir"] for r in rows} == {same, episode}


def test_failed_child_verifies_binding_records_elapsed_and_stops_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "PYTHON", tmp_path / "python.exe")
    campaign.PYTHON.write_bytes(b"mock")
    monkeypatch.setattr(campaign, "QUEUE", tmp_path / "logs/baseline-campaign-queue.json")
    monkeypatch.setattr(campaign, "CONTRACT", tmp_path / "resource.json")
    (tmp_path / "pipeline-state.json").write_text(json.dumps({"stages": [{"id": "P2", "status": "passed"}, {"id": "P3", "status": "running"}]}), encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/synthetic-GRU.json").write_text(json.dumps({"dataset_path": "synthetic.npz", "variant": "GRU", "output_dir": "unused"}), encoding="utf-8")
    monkeypatch.setattr(campaign, "read_contract", lambda: {"GRU": {"synthetic": 128}})
    monkeypatch.setattr(campaign, "_is_training_process", lambda: [])
    monkeypatch.setattr(aggregate_budget, "aggregate", lambda *a, **k: {"coverage_complete": True, "coverage_gaps": [], "spent_seconds": 0})
    binding_path = tmp_path / "runs/baselines/synthetic/GRU/seed0/input-binding-at-launch.json"
    binding_calls = []

    def prepare(*args, **kwargs):
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        binding_path.write_text("{}", encoding="utf-8")
        binding_calls.append("prepare")
        return binding_path

    def verify(path):
        binding_calls.append("verify")
        return {"status": "verified", "binding_path": str(path)}

    monkeypatch.setitem(sys.modules, "bind_training_inputs", type("BindingModule", (), {"prepare_binding": staticmethod(prepare), "verify_binding": staticmethod(verify)}))

    class FailedProcess:
        returncode = 7
        def __init__(self, *args, **kwargs): pass
        def poll(self): return 7

    monkeypatch.setattr(campaign.subprocess, "Popen", FailedProcess)
    campaign.run(type("Args", (), {"datasets": ["synthetic"], "methods": ["GRU"]})())
    queue = json.loads(campaign.QUEUE.read_text(encoding="utf-8"))
    row = queue["runs"][0]
    assert binding_calls == ["prepare", "verify"]
    assert row["status"] == "failed"
    assert row["training_wall_seconds"] >= 0
    assert "binding_verification" in row


def _phase_receipt(out, job, phase, seconds, status="completed"):
    return {"scope": "offline-experiment-phase", "id": job, "job_id": job, "output_dir": str(out.resolve()),
            "phase": phase, "wall_seconds": seconds, "status": status,
            "started_utc": "2026-10-11T00:00:00Z", "finished_utc": "2026-10-11T00:00:10Z"}


def test_phase_receipts_deduplicate_training_and_profile_by_output_max(tmp_path):
    out = tmp_path / "runs/baselines/email/V1/seed0"
    out.mkdir(parents=True)
    (out / "training_summary.json").write_text(json.dumps({"training_wall_seconds": 100}), encoding="utf-8")
    profile = tmp_path / "runs/profile/email-V1"
    profile.mkdir(parents=True)
    (profile / "profile-budget-accounting.json").write_text(json.dumps({
        "scope": "full-profile-replay", "wall_seconds": 20, "output_dir": str(out.resolve())
    }), encoding="utf-8")
    phase_dir = tmp_path / "logs/offline-phase-wall"
    phase_dir.mkdir(parents=True)
    (phase_dir / "train.json").write_text(json.dumps(_phase_receipt(out, "train-job", "training", 100)), encoding="utf-8")
    (phase_dir / "profile.json").write_text(json.dumps(_phase_receipt(out, "profile-job", "profiling", 20)), encoding="utf-8")
    (phase_dir / "all.json").write_text(json.dumps(_phase_receipt(out, "aggregate-job", "evaluation", 150)), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    rows = [r for r in report["accounting_rows"] if r["kind"] == "offline_experiment_phase_wall_seconds"]
    assert len(rows) == 1 and rows[0]["already_accounted_seconds"] == 120
    assert rows[0]["phase_total_wall_seconds"] == 270
    assert rows[0]["seconds"] == 150
    assert report["spent_seconds"] == 270


def test_phase_driver_receipt_id_and_job_id_are_accepted(tmp_path):
    out = tmp_path / "runs/baselines/synthetic/V1/seed0"
    out.mkdir(parents=True)
    phase_dir = tmp_path / "logs/offline-phase-wall"
    phase_dir.mkdir(parents=True)
    (phase_dir / "driver-job.json").write_text(json.dumps(_phase_receipt(out, "driver-123", "training", 7)), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    row = next(r for r in report["accounting_rows"] if r["kind"] == "offline_experiment_phase_wall_seconds")
    assert row["job_ids"] == ["driver-123"]
    assert row["seconds"] == 7


def test_phase_receipt_exclude_active_output_drops_its_budget_row(tmp_path):
    out = tmp_path / "runs/baselines/mooc/V1/seed0"
    out.mkdir(parents=True)
    phase_dir = tmp_path / "logs/offline-phase-wall"
    phase_dir.mkdir(parents=True)
    (phase_dir / "live.json").write_text(json.dumps(_phase_receipt(out, "live-job", "profiling", 77, "running")), encoding="utf-8")
    profile = tmp_path / "runs/profile/active-profile"
    profile.mkdir(parents=True)
    (profile / "profile-budget-accounting.json").write_text(json.dumps({
        "scope": "full-profile-replay", "wall_seconds": 20, "output_dir": str(out.resolve())
    }), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path, exclude_active_output=out)
    assert report["offline_phase_receipts_seen"] == 1
    assert not any(r["kind"] == "offline_experiment_phase_wall_seconds" for r in report["accounting_rows"])
    assert not any(r["kind"] == "profile_total_wall_seconds" for r in report["accounting_rows"])
    assert report["profile_execution_status"] == "not-started"
    assert report["spent_seconds"] == 0


def test_failed_and_stopped_phase_receipts_count_actual_wall_time(tmp_path):
    out = tmp_path / "runs/baselines/synthetic/DGCRN/seed0"
    out.mkdir(parents=True)
    phase_dir = tmp_path / "logs/offline-phase-wall"
    phase_dir.mkdir(parents=True)
    (phase_dir / "failed.json").write_text(json.dumps(_phase_receipt(out, "failed-job", "calibration", 31, "failed")), encoding="utf-8")
    (phase_dir / "stopped.json").write_text(json.dumps(_phase_receipt(out, "stopped-job", "robustness", 9, "stopped")), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    row = next(r for r in report["accounting_rows"] if r["kind"] == "offline_experiment_phase_wall_seconds")
    assert row["seconds"] == 40
    assert row["statuses"] == ["failed", "stopped"]
    assert report["spent_seconds"] == 40


def test_malformed_phase_wall_blocks_coverage_and_candidate_probe_wall_seconds_counts(tmp_path):
    out = tmp_path / "runs/baselines/email/SeqSNN/seed0"
    out.mkdir(parents=True)
    phase_dir = tmp_path / "logs/offline-phase-wall"
    phase_dir.mkdir(parents=True)
    (phase_dir / "bad.json").write_text(json.dumps(_phase_receipt(out, "bad-job", "evaluation", "NaN")), encoding="utf-8")
    logs = tmp_path / "logs"
    (logs / "candidate.json").write_text(json.dumps({
        "diagnostic_scope": "candidate function equivalence/resourceprobe; CPU only",
        "wall_seconds": 2.514,
        "candidate_edges_scored": 17,
    }), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    assert not report["coverage_complete"]
    assert any("offline phase receipt" in g and "wall_seconds" in g for g in report["coverage_gaps"])
    candidate = [r for r in report["accounting_rows"] if r["kind"] == "standalone_candidate_probe_duration_seconds"]
    assert len(candidate) == 1 and candidate[0]["seconds"] == 2.514
    assert candidate[0]["duration_basis"] == "wall_seconds"
    assert report["spent_seconds"] == 2.514


def test_candidate_probe_is_not_double_counted_when_generic_resource_scan_also_matches(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    path = logs / "p3-resource-probe-candidate.json"
    path.write_text(json.dumps({
        "scope": "candidate resource probe",
        "diagnostic_scope": "candidate resource probe",
        "wall_seconds": 4.0,
    }), encoding="utf-8")
    report = aggregate_budget.aggregate(tmp_path)
    rows = [r for r in report["accounting_rows"] if r["source"] == str(path)]
    assert len(rows) == 1
    assert report["spent_seconds"] == 4.0
