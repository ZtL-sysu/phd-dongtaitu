"""Continue root-accepted baseline queues after the existing campaign completes.

This is a local turn's serial continuation, not an automation or new stage.
Actual training remains in the unchanged bound controller and its 72h guard.
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
CURRENT = ROOT / "protocols/p3-bound-strong-baseline-queue-v3.json"
CONTINUATIONS = (
    ("protocols/p3-internal-bound-queue-v4.json", "protocols/p3-internal-bound-queue-root-acceptance-v4.json"),
    ("protocols/p3-external-bound-queue-v2.json", "protocols/p3-external-bound-queue-root-acceptance-v2.json"),
    ("protocols/p3-fresh-remaining-bound-queue-v1.json", "protocols/p3-fresh-remaining-bound-root-acceptance-v1.json"),
)

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def inside(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError("Continuation input escapes RUN")
    return path

def preflight(queue_path, acceptance_path):
    acceptance = read(acceptance_path)
    if acceptance.get("accepted") is not True and acceptance.get("verdict") != "pass":
        raise RuntimeError("Queue lacks root acceptance")
    if sha(queue_path) != acceptance.get("queue_sha256"):
        raise RuntimeError("Accepted queue bytes changed")
    for rel, digest in acceptance.get("file_inputs_sha256", {}).items():
        if sha(inside(rel)) != digest:
            raise RuntimeError("Accepted preparation input changed")
    if acceptance.get("resource_contract"):
        if sha(inside(acceptance["resource_contract"])) != acceptance.get("resource_contract_sha256"):
            raise RuntimeError("Accepted resource contract changed")
    if isinstance(acceptance.get("jobs"), list):
        for row in acceptance["jobs"]:
            cp = inside(row.get("config_path", row.get("config")))
            if sha(cp) != row["config_sha256"]:
                raise RuntimeError("Accepted config bytes changed")
            for entry in row.get("inputs", []):
                if isinstance(entry, dict) and sha(inside(entry["path"])) != entry["sha256"]:
                    raise RuntimeError("Accepted config input changed")
    queue = read(queue_path)
    for job in queue["jobs"]:
        if job["phase"] != "training" or job["scope"] not in {"development", "baseline"}:
            raise RuntimeError("Continuation is baseline-only")
        for path in job.get("inputs", []):
            if not inside(path).exists():
                raise FileNotFoundError(path)
    return {"queue":str(queue_path), "queue_sha256":sha(queue_path),
            "acceptance":str(acceptance_path), "acceptance_sha256":sha(acceptance_path),
            "jobs":len(queue["jobs"])}

def verify_current_complete(expected_sha):
    if sha(CURRENT) != expected_sha:
        raise RuntimeError("Awaited campaign queue changed")
    for job in read(CURRENT)["jobs"]:
        receipt_path = ROOT / "logs/offline-phase-wall" / (job["id"] + ".json")
        receipt = read(receipt_path)
        if receipt.get("status") != "completed" or receipt.get("returncode") != 0 or receipt.get("queue_sha256") != expected_sha:
            raise RuntimeError("Existing campaign did not complete successfully; preserve it for root recovery")
        for row in [*receipt.get("inputs", []), *receipt.get("outputs", [])]:
            if sha(inside(row["path"])) != row["sha256"]:
                raise RuntimeError("Completed awaited campaign binding changed")

def main():
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("Use the selected local Conda interpreter")
    report_path = ROOT / "logs/p3-serial-continuation-v1.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    records = [preflight(inside(q), inside(a)) for q, a in CONTINUATIONS]
    current_sha = sha(CURRENT)
    report = {"scope":"root-authorized serial P3 continuation", "status":"awaiting_current_campaign",
              "pid":os.getpid(), "awaited_queue":str(CURRENT), "awaited_queue_sha256":current_sha,
              "started_utc":dt.datetime.now(dt.timezone.utc).isoformat(), "queues":records,
              "training_clock_accounting":"Per-output unchanged bound-controller receipts, deduplicated with training timers; wait time is not charged twice.",
              "test_access":False, "proposed_training":False}
    def save():
        temp = report_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, report_path)
    save()
    lock = ROOT / "logs/offline-phase-wall/campaign.lock"
    try:
        while lock.exists():
            owner = read(lock)
            if inside(owner.get("queue", "")) != CURRENT or owner.get("queue_sha256") != current_sha:
                raise RuntimeError("Unexpected campaign owns the awaited resource lease")
            time.sleep(5)
        verify_current_complete(current_sha)
        for record in records:
            preflight(Path(record["queue"]), Path(record["acceptance"]))
            if sha(record["acceptance"]) != record["acceptance_sha256"]:
                raise RuntimeError("Root acceptance changed during continuation")
            report.update(status="running_baseline_queue", current_queue=record["queue"])
            save()
            print(json.dumps({"starting_queue":record["queue"], "jobs":record["jobs"]}), flush=True)
            child = subprocess.run([str(PYTHON), str(ROOT / "edsgn/scripts/bound_phase_campaign.py"),
                                    "--queue", record["queue"]], cwd=ROOT)
            record["returncode"] = child.returncode
            save()
            if child.returncode != 0:
                raise RuntimeError("Preserved failed continuation; no later queue launched")
        report.update(status="completed_all_baseline_queues")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["last_updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()

if __name__ == "__main__":
    main()
