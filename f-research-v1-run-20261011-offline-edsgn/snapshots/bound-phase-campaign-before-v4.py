"""Serial, prebound local model phases with one cumulative budget.

This driver does not select a model or read prediction arrays. A reviewed queue
must supply all commands and input paths. Test commands require P3A acceptance.
Interrupted/failed output directories are preserved and are never auto-resumed.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from aggregate_budget import aggregate
from baseline_campaign import PYTHON, ROOT, WALL_BUDGET_S, sha256
from bind_training_inputs import prepare_binding, verify_binding

MODULES = {
    "training": "edsgn.train", "calibration": "edsgn.calibrate",
    "evaluation": "edsgn.evaluate", "robustness": None,
    "profiling": "edsgn.profile",
}
INPUT_FLAGS = {"--config", "--checkpoint", "--dataset-path", "--queue", "--recipe", "--policy-config"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _is_training_process():
    import psutil
    active = []
    for proc in psutil.process_iter(["pid", "cmdline", "cwd", "exe"]):
        try:
            argv = [str(a).replace("\\", "/").lower() for a in (proc.info.get("cmdline") or [])]
            exe = str(proc.info.get("exe") or (argv[0] if argv else "")).replace("\\", "/")
            if proc.pid == os.getpid() or not Path(exe).name.lower().startswith("python"):
                continue
            owned = str(ROOT).replace("\\", "/").lower() in " ".join(argv) or str(proc.info.get("cwd", "")).casefold() == str(ROOT).casefold()
            modules = {x for x in MODULES.values() if x}
            model_job = any(argv[i] == "-m" and i+1 < len(argv) and argv[i+1] in modules for i in range(len(argv)))
            scripts = {"baseline_campaign.py", "bound_phase_campaign.py", "frozen_robustness_phase.py", "profile_training_forward_components.py"}
            model_job |= any(Path(x).name in scripts or "resource_probe" in Path(x).name for x in argv)
            if owned and model_job:
                active.append({"pid":proc.pid})
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, TypeError):
            continue
    return active


def command_for(kind, args):
    if kind == "robustness":
        return [str(PYTHON), str(ROOT/"edsgn"/"scripts"/"frozen_robustness_phase.py"), *args]
    return [str(PYTHON), "-m", MODULES[kind], *args]


def contained(value):
    p = Path(value)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"Phase artifact is outside the run root: {p}")
    return p


def write_state(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def verify_entries(entries):
    for item in entries:
        p = Path(item["path"])
        if not p.is_file() or sha256(p) != item["sha256"]:
            raise RuntimeError(f"Phase input changed: {p}")


def validate_outputs(job, output, kind):
    required = {
        "training": ("training_summary.json", "checkpoint.pt", "manifest.json", "training_log.csv", "validation_predictions.npz", "metrics_by_horizon_val.csv"),
        "calibration": ("calibration.json", "cost-table.json", "grouped_action_costs.csv"),
        "evaluation": ("evaluation_manifest.json", "predictions.npz", "metrics_by_horizon.csv", "trace.jsonl"),
        "robustness": ("perturbation-manifest.json", "predictions.npz", "accuracy-rows.json", "counter-rows.json"),
        "profiling": ("runtime_summary.json", "profile-budget-accounting.json", "window_timings.csv"),
    }[kind]
    for name in required:
        if not (output/name).is_file():
            raise FileNotFoundError(f"Completed phase lacks artifact: {output/name}")
    filename = {"training":"training_summary.json", "calibration":"calibration.json", "evaluation":"evaluation_manifest.json", "robustness":"perturbation-manifest.json", "profiling":"runtime_summary.json"}[kind]
    data = json.loads((output/filename).read_text(encoding="utf-8"))
    scope = {"training":"full_protocol", "calibration":"full_validation", "evaluation":"full_protocol", "profiling":"full_selected_split"}.get(kind)
    if scope and data.get("scope") != scope:
        raise RuntimeError("Phase does not contain full-protocol output")
    if kind == "training":
        if data.get("test_accessed") is not False or data.get("stop_loss_reached") is not False or data.get("checkpoint_sha256") != sha256(output/"checkpoint.pt"):
            raise RuntimeError("Training completion/checkpoint integrity failed")
        manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        with (output/"training_log.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or len(rows) != data.get("epochs_executed") or any(int(row["train_windows"]) != manifest["train_windows"] for row in rows):
            raise RuntimeError("Training epoch/window coverage mismatch")
    else:
        before, after = {
            "calibration":("parameters_before", "parameters_after"),
            "evaluation":("parameters_before", "parameters_after"),
            "robustness":("model_weights_before_sha256", "model_weights_after_sha256"),
            "profiling":("parameters_sha256_before", "parameters_sha256_after"),
        }[kind]
        if not data.get(before) or data[before] != data.get(after):
            raise RuntimeError("Frozen phase weight hash mismatch")
        if kind != "calibration":
            expected = job.get("expected_n_windows")
            if not isinstance(expected, int) or expected < 1 or data.get("n_windows") != expected:
                raise RuntimeError("Phase window coverage differs from frozen queue")
        if kind == "profiling" and (data.get("batch_size") != 1 or data.get("dtype") != "float32" or data.get("repeats", 0) < 5 or data.get("warmup_windows", 0) < 10):
            raise RuntimeError("Incomplete profiling protocol")
    return [{"path":str(output/name), "sha256":sha256(output/name)} for name in required]


def validate_job(job):
    kind = job["phase"]
    if kind not in MODULES:
        raise ValueError(f"Unsupported phase: {kind}")
    identifier = job["id"]
    if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in identifier):
        raise ValueError("Phase id must be a simple filename component")
    args = job["args"]
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError("Phase args must be a list of strings")
    output = contained(job["output_dir"])
    option = "--config" if kind == "training" else "--output"
    if args.count(option) != 1 or args.index(option)+1 >= len(args):
        raise ValueError(f"Phase requires exactly one {option}")
    if kind == "training":
        config_path = contained(args[args.index(option)+1])
        cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if contained(cfg["output_dir"]) != output:
            raise ValueError("Training config/output mismatch")
        if any(k in cfg for k in ("resume", "smoke_train_windows", "smoke_val_windows", "max_run_seconds")):
            raise ValueError("This campaign requires new full-protocol training")
    elif contained(args[args.index(option)+1]) != output:
        raise ValueError("Command/output mismatch")
    if "--smoke-limit" in args:
        raise ValueError("This driver requires complete selected-split phases")
    return kind, output


def command_input_paths(job):
    """Bind input flags independently of a queue author's optional inputs list."""
    args = job["args"]
    result = {contained(p) for p in job.get("inputs", [])}
    for flag in INPUT_FLAGS:
        if flag in args:
            if args.count(flag) != 1 or args.index(flag)+1 >= len(args):
                raise ValueError(f"Ambiguous input flag: {flag}")
            result.add(contained(args[args.index(flag)+1]))
    return result


def gate(job, kind):
    state = json.loads((ROOT/"pipeline-state.json").read_text(encoding="utf-8"))
    stages = {row["id"]: row["status"] for row in state["stages"]}
    if stages.get("P2") != "passed":
        raise RuntimeError("P2 must be accepted before model phases")
    if kind == "training":
        expected = "P4" if job.get("scope") == "formal" else "P3"
        if stages.get(expected) != "running":
            raise RuntimeError(f"Training requires {expected}=running")
    else:
        if stages.get("P3A") != "passed" or stages.get("P4") != "running":
            raise RuntimeError("Frozen calibration/evaluation/profile requires P3A=passed/P4=running")


def run_queue(queue_path):
    queue_path = contained(queue_path)
    queue_hash = sha256(queue_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8-sig"))
    jobs = queue["jobs"]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job id")
    folder = ROOT/"logs"/"offline-phase-wall"
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder/"campaign.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(json.dumps({"pid":os.getpid(), "started_utc":utc(), "queue":str(queue_path), "queue_sha256":queue_hash}))
    try:
        for job in jobs:
            kind, output = validate_job(job)
            gate(job, kind)
            report_path = folder/f"{job['id']}.json"
            if report_path.exists():
                old = json.loads(report_path.read_text(encoding="utf-8"))
                if old.get("status") != "completed" or old.get("queue_sha256") != queue_hash:
                    raise RuntimeError(f"Preserved phase requires explicit recovery: {report_path}")
                verify_entries(old["inputs"])
                if not old.get("outputs"):
                    raise RuntimeError("Completed receipt lacks output bindings")
                verify_entries(old["outputs"])
                validate_outputs(job, output, kind)
                continue
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Preserve existing phase output: {output}")
            if _is_training_process():
                raise RuntimeError("An owned training/resource process is active")
            budget = aggregate(ROOT)
            if not budget["coverage_complete"] or budget["spent_seconds"] >= WALL_BUDGET_S:
                raise RuntimeError(f"Budget gate refused phase: {budget['coverage_gaps']}")
            if sha256(queue_path) != queue_hash:
                raise RuntimeError("Queue changed during execution")
            paths = {queue_path, Path(__file__).resolve()}
            if kind == "robustness":
                paths.add(ROOT/"edsgn"/"scripts"/"frozen_robustness_phase.py")
            paths.update((ROOT/"edsgn"/"src"/"edsgn").glob("*.py"))
            for p in command_input_paths(job):
                if not p.exists():
                    raise FileNotFoundError(f"Phase input does not exist: {p}")
                paths.update(x for x in p.rglob("*") if x.is_file() and x.suffix in {".py", ".json", ".npz"}) if p.is_dir() else paths.add(p)
            inputs = [{"path":str(p), "sha256":sha256(p)} for p in sorted(paths)]
            output.mkdir(parents=True, exist_ok=True)
            binding = None
            if kind == "training":
                binding = prepare_binding(contained(job["args"][job["args"].index("--config")+1]), output_dir=output)
            command = command_for(kind, job["args"])
            record = {"schema_version":1, "scope":"offline-experiment-phase", "id":job["id"], "job_id":job["id"], "phase":kind,
                      "queue_sha256":queue_hash, "output_dir":str(output), "command":command,
                      "inputs":inputs, "status":"running", "wall_seconds":0., "started_utc":utc(),
                      "binding_path":str(binding) if binding else None, "budget_policy":"single cumulative72-hour budget"}
            write_state(report_path, record)
            env = os.environ.copy()
            env.update(PYTHONPATH=str(ROOT/"edsgn"/"src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
            started = time.monotonic()
            reason = None
            proc = None
            try:
                with (folder/f"{job['id']}.log").open("x", encoding="utf-8") as log:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    while proc.poll() is None:
                        elapsed = time.monotonic()-started
                        record["wall_seconds"] = elapsed
                        write_state(report_path, record)
                        budget = aggregate(ROOT, exclude_active_output=str(output))
                        if not budget["coverage_complete"] or budget["spent_seconds"]+elapsed >= WALL_BUDGET_S:
                            reason = "cumulative budget/coverage stop"
                            proc.terminate()
                            break
                        time.sleep(5)
                    proc.wait(timeout=30)
            except BaseException as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=30)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            record.update(wall_seconds=time.monotonic()-started, finished_utc=utc(), returncode=proc.returncode if proc else None)
            try:
                verify_entries(inputs)
                if binding: verify_binding(binding)
                if reason is not None or record["returncode"] != 0:
                    raise RuntimeError(reason or "Child phase failed")
                if sha256(queue_path) != queue_hash:
                    raise RuntimeError("Queue changed during phase")
                record["outputs"] = validate_outputs(job, output, kind)
                record["status"] = "completed"
            except Exception as exc:
                record.update(status="stopped" if reason else "failed", reason=str(exc))
            write_state(report_path, record)
            print(json.dumps({k:record.get(k) for k in ("id", "status", "wall_seconds", "reason")}), flush=True)
            if record["status"] != "completed":
                raise RuntimeError(f"Preserved unsuccessful phase: {job['id']}")
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    run_queue(parser.parse_args().queue)
