"""Continue authorized GRU replications after the existing P3 coordinator.

This local serial continuation performs no test selection, new stage, or
scheduled automation. Actual training is owned by the unchanged bound runner.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe")
QUEUE = ROOT / "protocols/p3-gru-replication-queue-v1.json"
ACCEPTANCE = ROOT / "protocols/p3-gru-replication-root-acceptance-v1.json"
PREDECESSOR = ROOT / "logs/p3-serial-continuation-v1.json"
PREDECESSOR_SCRIPT = ROOT / "edsgn/scripts/continue_p3_baselines_serial_v1.py"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(raw):
    path = Path(raw)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError("continuation artifact escapes RUN")
    return path


def preflight(expected_queue, expected_acceptance):
    acceptance = read(ACCEPTANCE)
    if sha(QUEUE) != expected_queue or sha(ACCEPTANCE) != expected_acceptance:
        raise RuntimeError("accepted continuation bytes changed")
    if acceptance.get("accepted") is not True or acceptance.get("accepted_by") != "root":
        raise RuntimeError("GRU continuation lacks root acceptance")
    if acceptance.get("queue_sha256") != expected_queue:
        raise RuntimeError("accepted queue hash differs")
    for rel, value in acceptance["file_inputs_sha256"].items():
        if sha(inside(rel)) != value:
            raise RuntimeError(f"accepted preparation changed: {rel}")
    for job in read(QUEUE)["jobs"]:
        if job["phase"] != "training" or job["scope"] != "baseline":
            raise RuntimeError("GRU continuation must remain P3 baseline training only")
        config = read(job["config"])
        if config["variant"] != "GRU" or config["seed"] not in [1, 2, 3, 4]:
            raise RuntimeError("unexpected model/seed in GRU replication")
        if inside(job["output_dir"]).exists():
            raise RuntimeError("GRU replication output already exists")


def verify_queue_completed(path, expected_sha):
    path = inside(path)
    if sha(path) != expected_sha:
        raise RuntimeError("completed predecessor queue bytes changed")
    for job in read(path)["jobs"]:
        receipt = read(ROOT / "logs/offline-phase-wall" / f"{job['id']}.json")
        if (receipt.get("status") != "completed" or receipt.get("returncode") != 0
                or receipt.get("queue_sha256") != expected_sha):
            raise RuntimeError(f"predecessor/completed GRU receipt failed: {job['id']}")
        for row in [*receipt.get("inputs", []), *receipt.get("outputs", [])]:
            if sha(inside(row["path"])) != row["sha256"]:
                raise RuntimeError(f"completed receipt artifact changed: {row['path']}")


def require_p3():
    rows = {r["id"]: r for r in read(ROOT / "pipeline-state.json")["stages"]}
    if rows["P2"]["status"] != "passed" or rows["P2"]["gate_verdict"] != "pass" or rows["P3"]["status"] != "running":
        raise RuntimeError("GRU continuation requires P2 passed/pass and P3 running")
    checked = subprocess.run([str(PYTHON), str(ROOT / "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py"),
                              str(ROOT / "pipeline-state.json"), "--before", "P3"],
                             cwd=ROOT, capture_output=True, text=True, shell=False)
    if checked.returncode:
        raise RuntimeError(f"official P3 predecessor validator rejected: {checked.stdout[-1000:]} {checked.stderr[-1000:]}")


def main():
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("use the selected local Conda interpreter")
    target = ROOT / "logs/p3-gru-serial-continuation-v1.json"
    if target.exists():
        raise FileExistsError(target)
    queue_sha, acceptance_sha = sha(QUEUE), sha(ACCEPTANCE)
    predecessor_source_sha = sha(PREDECESSOR_SCRIPT)
    own_source_sha = sha(Path(__file__))
    preflight(queue_sha, acceptance_sha)
    previous = read(PREDECESSOR)
    awaited = [(previous["awaited_queue"], previous["awaited_queue_sha256"])] + [
        (r["queue"], r["queue_sha256"]) for r in previous["queues"]]
    record = {"status": "awaiting_seed0_coordinator", "pid": os.getpid(),
              "scope": "root-authorized P3 baseline GRU seeds1-4 serial continuation",
              "queue_path": str(QUEUE), "queue_sha256": queue_sha,
              "acceptance_sha256": acceptance_sha, "awaited_queues": awaited,
              "predecessor_source_sha256": predecessor_source_sha,
              "source_sha256": own_source_sha, "job_count": 16,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "test_access": False, "training_started": False,
              "clock": "unchanged per-output bound-controller 72h accounting; wait not charged"}

    def save():
        record["last_updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)

    save()
    try:
        while True:
            previous = read(PREDECESSOR)
            if previous["status"] == "failed":
                raise RuntimeError("seed0 predecessor failed; no GRU replication launched")
            if previous["status"] == "completed_all_baseline_queues":
                break
            if previous["status"] not in {"awaiting_current_campaign", "running_baseline_queue"}:
                raise RuntimeError("unrecognized predecessor status")
            if sha(PREDECESSOR_SCRIPT) != predecessor_source_sha or sha(Path(__file__)) != own_source_sha:
                raise RuntimeError("serial continuation source changed while waiting")
            time.sleep(5)
        require_p3()
        for path, value in awaited:
            verify_queue_completed(path, value)
        preflight(queue_sha, acceptance_sha)
        if any((ROOT / rel).exists() for rel in ["logs/model-resource.lock", "logs/offline-phase-wall/campaign.lock"]):
            raise RuntimeError("resource lease still active after predecessor completion")
        record.update(status="running_gru_replications", training_started=True)
        save()
        child = subprocess.run([str(PYTHON), str(ROOT / "edsgn/scripts/bound_phase_campaign.py"),
                                "--queue", str(QUEUE)], cwd=ROOT, shell=False)
        record["returncode"] = child.returncode
        if child.returncode:
            raise RuntimeError("GRU continuation failed; preserve all receipts")
        verify_queue_completed(QUEUE, queue_sha)
        record["status"] = "completed_gru_replications"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
