"""Wait for accepted seed0/GRU coordinators, then dispatch strong seeds 1-4 serially.

Waiting is not charged to the shared phase ledger. Actual work is delegated only
to the unchanged bound_phase_campaign controller after all gates and bindings pass.
This continuation never opens prediction arrays or test values.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe")
QUEUE = ROOT / "protocols/p3-strong-replication-queue-v1.json"
ROOT_ACCEPTANCE = ROOT / "reviews/p3-strong-replication-root-acceptance-v1.json"
ROOT_METADATA = ROOT / "logs/p3-strong-replication-root-metadata-v1.json"
SEED0_NUMERIC_ACCEPTANCE = ROOT / "reviews/p3-strong-seed0-numeric-root-acceptance-v1.json"
SEED0_COORDINATOR = ROOT / "logs/p3-serial-continuation-v1.json"
GRU_COORDINATOR = ROOT / "logs/p3-gru-serial-continuation-v1.json"
SEED0_QUEUE = ROOT / "protocols/p3-bound-strong-baseline-queue-v3.json"
SEED0_QUEUE_SHA256 = "d93689af64b9f9744f9efc6f8b7177c97e236dfab904a117f2757a98114a0d9f"
SEED0_QUEUE_SPECS = (
    ("protocols/p3-bound-strong-baseline-queue-v3.json", "d93689af64b9f9744f9efc6f8b7177c97e236dfab904a117f2757a98114a0d9f", 8, "protocols/p3-bound-strong-baseline-root-acceptance-v3.json"),
    ("protocols/p3-internal-bound-queue-v4.json", "cef3d05099c5a910f4c69aa3b5b10cf804779ba4d2c76ea15c21ab9f30f85f79", 10, "protocols/p3-internal-bound-queue-root-acceptance-v4.json"),
    ("protocols/p3-external-bound-queue-v2.json", "41c116fc812817ed1152c5030a626aa96922e94d468e84512335b4efada8865d", 16, "protocols/p3-external-bound-queue-root-acceptance-v2.json"),
    ("protocols/p3-fresh-remaining-bound-queue-v1.json", "a7d0a2d23425724c6bdcb920f2bff89a49b4570b445f3d571d09ff442db94e0e", 6, "protocols/p3-fresh-remaining-bound-root-acceptance-v1.json"),
)
GRU_QUEUE_SPEC = ("protocols/p3-gru-replication-queue-v1.json", "9712f88677bb459a96a89d62b4804c2c4b3201b7938d80f29a8f3915671152ab", 16)
COORDINATOR_SOURCE_PINS = {
    "edsgn/scripts/continue_p3_baselines_serial_v1.py": "fabfa9b6c84cfb5625522fbed9ed5de5cdb632bfd9aa785fbecd055b71385c26",
    "edsgn/scripts/continue_p3_gru_replication_serial_v1.py": "4bb2aa86c22a4e83de46933fb9f3005b14b9a3d0024a5c386b40e0b3fe921e40",
}
TRAINING_OUTPUTS = {
    "training_summary.json", "checkpoint.pt", "manifest.json", "config_resolved.json",
    "training_log.csv", "validation_predictions.npz", "metrics_by_horizon_val.csv",
}
DATASETS = ("synthetic", "email", "mooc")
METHODS = ("DGCRN", "SeqSNN")


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_atomic_status(target: str | Path, record: dict, *, replace=os.replace,
                       monotonic=time.monotonic, sleep=time.sleep) -> Path:
    """Durably write status then atomically replace, retrying transient Windows locks only.

    On retry exhaustion the unique same-directory temporary file is deliberately
    preserved for diagnosis. Scientific subprocesses are never retried here.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    deadline = monotonic() + 15.0
    delay = 0.05
    while True:
        try:
            replace(temp, target)
            return target
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33}:
                raise
            now = monotonic()
            if now >= deadline:
                raise TimeoutError(f"status replace remained locked; preserved temporary file: {temp}") from exc
            sleep(min(delay, 0.5, deadline - now))
            delay = min(delay * 2.0, 0.5)


def read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def inside(raw: str | Path) -> Path:
    path = Path(raw)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError(f"continuation input escapes RUN: {path}")
    return path


def require_hash_map(mapping: dict, *, exact_count: int | None = None, exclude: set[str] = frozenset()):
    if not isinstance(mapping, dict) or (exact_count is not None and len(mapping) != exact_count):
        raise RuntimeError("invalid or incomplete SHA-256 pin map")
    for rel, expected in mapping.items():
        path = inside(rel)
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"pinned input changed or missing: {rel}")


def verify_replication_acceptance(expected_queue_sha: str):
    acceptance = read(ROOT_ACCEPTANCE)
    metadata = read(ROOT_METADATA)
    if (acceptance.get("accepted_by") != "root" or acceptance.get("verdict") != "pass"
            or acceptance.get("scope") != "P3 primary strong baseline replication preparation"
            or acceptance.get("pipeline_advanced") is not False
            or acceptance.get("scientific_hypothesis_accepted") is not False
            or acceptance.get("actual_training_started") is not False
            or acceptance.get("test_access_authorized") is not False
            or acceptance.get("queue_path") != "protocols/p3-strong-replication-queue-v1.json"
            or acceptance.get("queue_sha256") != expected_queue_sha
            or acceptance.get("job_count") != 24):
        raise RuntimeError("strong replication root acceptance does not match its prepared-only scope")
    pins = acceptance.get("evidence_sha256")
    require_hash_map(pins, exact_count=112)
    if pins.get("logs/p3-strong-replication-root-metadata-v1.json") != sha(ROOT_METADATA):
        raise RuntimeError("root metadata review is not the accepted pinned record")
    if (metadata.get("status") != "pass-prepared-only" or metadata.get("reviewer") != "root"
            or metadata.get("job_count") != 24 or metadata.get("metadata_controller_validation") is not True
            or metadata.get("actual_training_started") is not False or metadata.get("arrays_decoded") is not False):
        raise RuntimeError("root metadata/controller review no longer proves preparation only")
    metadata_pins = metadata.get("file_inputs_sha256")
    if not isinstance(metadata_pins, dict) or len(metadata_pins) != 111:
        raise RuntimeError("unexpected root metadata pin schema")
    expected_without_metadata = {k: v for k, v in pins.items() if k != "logs/p3-strong-replication-root-metadata-v1.json"}
    if metadata_pins != expected_without_metadata:
        raise RuntimeError("root acceptance evidence pins differ from the metadata review pin set")
    return {"acceptance_sha256": sha(ROOT_ACCEPTANCE), "metadata_sha256": sha(ROOT_METADATA), "pins_verified": len(pins)}


def verify_queue_and_configs(expected_queue_sha: str, expected_acceptance_pins: dict):
    if sha(QUEUE) != expected_queue_sha:
        raise RuntimeError("prepared strong queue bytes changed")
    queue = read(QUEUE)
    if (queue.get("schema_version") != 3 or queue.get("status") != "prepared-not-launched"
            or queue.get("required_gates") != {"P2": "passed", "P3": "running"}
            or queue.get("test_access_authorized") is not False
            or queue.get("execution_authorized") is not False):
        raise RuntimeError("strong queue no longer has the accepted schema/gates/scope")
    if len(queue.get("jobs", [])) != 24 or len(queue.get("config_bindings", [])) != 24:
        raise RuntimeError("strong queue job/config binding count changed")
    require_hash_map({Path(item["path"]).relative_to(ROOT).as_posix(): item["sha256"] for item in queue["source_bindings"]})
    seen = set()
    for job in queue["jobs"]:
        dataset, method = job.get("dataset"), job.get("variant")
        if dataset not in DATASETS or method not in METHODS:
            raise RuntimeError("strong queue contains an out-of-scope dataset/method")
        args = job.get("args", [])
        if len(args) != 2 or args[0] != "--config" or str(inside(args[1])) != str(inside(job["config_path"])):
            raise RuntimeError("strong queue config argv is not exact")
        config_path = inside(job["config_path"])
        config = read(config_path)
        source0 = inside(job["seed0_config_path"])
        config0 = read(source0)
        seed = config.get("seed")
        identity = (dataset, method, seed)
        if identity in seen or seed not in (1, 2, 3, 4):
            raise RuntimeError("duplicate or out-of-range strong model seed")
        seen.add(identity)
        rel = config_path.relative_to(ROOT).as_posix()
        seed0_rel = source0.relative_to(ROOT).as_posix()
        if (sha(config_path) != job.get("config_sha256") or sha(source0) != job.get("seed0_config_sha256")
                or expected_acceptance_pins.get(rel) != sha(config_path)
                or expected_acceptance_pins.get(seed0_rel) != sha(source0)):
            raise RuntimeError("strong config or seed0 template hash is not root-pinned")
        if config.get("variant") != method or config.get("output_dir") != job.get("output_dir"):
            raise RuntimeError("strong config/job method or output path mismatch")
        if config.get("epochs") != 100 or config.get("patience") != 10:
            raise RuntimeError("strong config training cap changed")
        if {k: v for k, v in config.items() if k not in {"seed", "output_dir"}} != {
            k: v for k, v in config0.items() if k not in {"seed", "output_dir"}
        }:
            raise RuntimeError("replication config changed fields beyond seed/output_dir")
        output = inside(job["output_dir"])
        if output.exists():
            raise RuntimeError(f"strong replication output already exists: {output}")
        if job.get("phase") != "training" or job.get("scope") != "baseline":
            raise RuntimeError("strong jobs must remain P3 baseline training")
        for row in job.get("input_binding", {}).get("static_source_files", []):
            path = inside(row["path"])
            if not path.is_file() or sha(path) != row.get("sha256"):
                raise RuntimeError(f"strong static source binding changed: {path}")
    if len(seen) != 24:
        raise RuntimeError("expected all 3x2x4 distinct replication configurations")
    return queue


def verify_bound_root_acceptance(path: Path, queue_path: Path, expected_sha: str):
    accepted = read(path)
    if accepted.get("accepted") is not True and accepted.get("verdict") != "pass":
        raise RuntimeError(f"predecessor queue is not root accepted: {path}")
    if accepted.get("queue_sha256") != expected_sha or sha(queue_path) != expected_sha:
        raise RuntimeError(f"predecessor queue/acceptance hash mismatch: {path}")
    for rel, expected in accepted.get("file_inputs_sha256", {}).items():
        target = inside(rel)
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"predecessor accepted input changed: {rel}")


def verify_receipt(queue_path: Path, queue_sha: str, job: dict):
    receipt_path = ROOT / "logs/offline-phase-wall" / f"{job['id']}.json"
    receipt = read(receipt_path)
    if (receipt.get("status") != "completed" or receipt.get("returncode") != 0
            or receipt.get("queue_sha256") != queue_sha or receipt.get("phase") != "training"
            or inside(receipt.get("output_dir", "")) != inside(job["output_dir"])):
        raise RuntimeError(f"predecessor phase receipt incomplete or mismatched: {job['id']}")
    input_rows, output_rows = receipt.get("inputs", []), receipt.get("outputs", [])
    if not input_rows or not output_rows:
        raise RuntimeError(f"predecessor receipt lacks input/output bindings: {job['id']}")
    for row in [*input_rows, *output_rows]:
        path = inside(row["path"])
        if not path.is_file() or sha(path) != row.get("sha256"):
            raise RuntimeError(f"predecessor receipt artifact changed: {path}")
    names = {Path(row["path"]).name for row in output_rows}
    if not TRAINING_OUTPUTS.issubset(names):
        raise RuntimeError(f"predecessor training receipt lacks full-protocol outputs: {job['id']}")
    config_index = job.get("config")
    if not config_index:
        args = job.get("args", [])
        if args.count("--config") != 1:
            raise RuntimeError(f"predecessor queue job lacks exact --config: {job['id']}")
        config_index = args[args.index("--config") + 1]
    config_path = inside(config_index)
    expected_cfg = sha(config_path)
    if not any(inside(row["path"]) == config_path and row["sha256"] == expected_cfg for row in input_rows):
        raise RuntimeError(f"phase inputs do not bind the exact config: {job['id']}")
    return receipt_path



def require_coordinator_source_pins():
    for rel, expected in COORDINATOR_SOURCE_PINS.items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"pinned predecessor coordinator source changed: {rel}")


def validate_exact_predecessor_registry(seed0_rows, gru_row, gru_awaited, queue_jobs):
    expected = [(str(inside(path)), digest, count, str(inside(accept))) for path, digest, count, accept in SEED0_QUEUE_SPECS]
    normalized = [(str(inside(path)), digest, count, str(inside(accept))) for path, digest, count, accept in seed0_rows]
    if len(normalized) != 4 or len({row[0] for row in normalized}) != 4 or normalized != expected:
        raise RuntimeError("seed0 queue descriptors differ from exact four-queue registry")
    expected_gru = (str(inside(GRU_QUEUE_SPEC[0])), GRU_QUEUE_SPEC[1], GRU_QUEUE_SPEC[2])
    normalized_gru = (str(inside(gru_row[0])), gru_row[1], gru_row[2])
    if normalized_gru != expected_gru:
        raise RuntimeError("GRU queue differs from exact accepted source queue")
    if len(gru_awaited) != 4:
        raise RuntimeError("GRU continuation must await exactly four seed0 queues")
    awaited = [(str(inside(path)), digest) for path, digest in gru_awaited]
    wanted = [(row[0], row[1]) for row in expected]
    if len({path for path, _ in awaited}) != 4 or awaited != wanted:
        raise RuntimeError("GRU awaited identities differ from exact unique seed0 queues")
    ids = []
    for path, _digest, count, _accept in expected:
        jobs = queue_jobs.get(path)
        if not isinstance(jobs, list) or len(jobs) != count:
            raise RuntimeError(f"wrong fixed job count for {path}")
        local = [job.get("id") for job in jobs]
        if any(not isinstance(x, str) or not x for x in local) or len(set(local)) != count:
            raise RuntimeError(f"invalid or duplicate job IDs in {path}")
        ids.extend(local)
    gru_jobs = queue_jobs.get(expected_gru[0])
    if not isinstance(gru_jobs, list) or len(gru_jobs) != 16:
        raise RuntimeError("GRU source queue must have exactly 16 jobs")
    gru_ids = [job.get("id") for job in gru_jobs]
    if any(not isinstance(x, str) or not x for x in gru_ids) or len(set(gru_ids)) != 16:
        raise RuntimeError("invalid or duplicate GRU job IDs")
    ids.extend(gru_ids)
    if len(ids) != 56 or len(set(ids)) != 56:
        raise RuntimeError("predecessor queues must define exactly 56 unique job IDs")
    return ids


def verify_coordinator_queues():
    require_coordinator_source_pins()
    seed0 = read(SEED0_COORDINATOR)
    gru = read(GRU_COORDINATOR)
    if seed0.get("status") != "completed_all_baseline_queues":
        raise RuntimeError("seed0 coordinator has not completed all four source-bound queues")
    if gru.get("status") != "completed_gru_replications":
        raise RuntimeError("GRU coordinator has not completed its 16 replications")
    current_path = inside(seed0["awaited_queue"])
    if current_path != SEED0_QUEUE or seed0.get("awaited_queue_sha256") != SEED0_QUEUE_SHA256:
        raise RuntimeError("seed0 coordinator's original strong-baseline queue binding changed")
    if len(seed0.get("queues", [])) != 3:
        raise RuntimeError("seed0 coordinator must declare exactly three continuations")
    seed0_queues = [(inside(rel), digest) for rel, digest, _count, _accept in SEED0_QUEUE_SPECS]
    if seed0_queues[0] != (current_path, SEED0_QUEUE_SHA256):
        raise RuntimeError("seed0 current queue differs from exact registry")
    for row, (rel, digest, count, acceptance_rel) in zip(seed0["queues"], SEED0_QUEUE_SPECS[1:]):
        if (str(inside(row.get("queue", ""))) != str(inside(rel)) or row.get("queue_sha256") != digest
                or row.get("jobs") != count or str(inside(row.get("acceptance", ""))) != str(inside(acceptance_rel))):
            raise RuntimeError("seed0 continuation descriptor differs from exact accepted registry")
    gru_queue_path = inside(gru["queue_path"])
    if str(gru_queue_path) != str(inside(GRU_QUEUE_SPEC[0])):
        raise RuntimeError("GRU coordinator queue path changed")
    gru_queue_sha = gru.get("queue_sha256")
    if gru_queue_sha != GRU_QUEUE_SPEC[1] or sha(gru_queue_path) != gru_queue_sha:
        raise RuntimeError("GRU replication queue hash changed")
    queue_jobs = {}
    for rel, digest, _count, _accept in SEED0_QUEUE_SPECS:
        path = inside(rel)
        if sha(path) != digest:
            raise RuntimeError(f"seed0 queue changed: {path}")
        queue_jobs[str(path)] = read(path).get("jobs", [])
    queue_jobs[str(gru_queue_path)] = read(gru_queue_path).get("jobs", [])
    expected_ids = validate_exact_predecessor_registry(
        [(str(inside(rel)), digest, count, str(inside(accept))) for rel, digest, count, accept in SEED0_QUEUE_SPECS],
        (str(gru_queue_path), gru_queue_sha, 16), gru.get("awaited_queues", []), queue_jobs)
    total_jobs = 0
    receipts = []
    for (queue_path, queue_sha), (_rel, _digest, _count, acceptance_rel) in zip(seed0_queues, SEED0_QUEUE_SPECS):
        if sha(queue_path) != queue_sha:
            raise RuntimeError(f"seed0 queue changed: {queue_path}")
        queue = read(queue_path)
        jobs = queue.get("jobs", [])
        if not jobs:
            raise RuntimeError(f"seed0 queue has no jobs: {queue_path}")
        if len(jobs) != _count:
            raise RuntimeError(f"seed0 queue count differs from fixed registry: {queue_path}")
        if queue_path != inside(SEED0_QUEUE_SPECS[0][0]):
            acceptance_path = inside(acceptance_rel)
            descriptor = next(row for row in seed0["queues"] if inside(row["queue"]) == queue_path)
            if sha(acceptance_path) != descriptor.get("acceptance_sha256"):
                raise RuntimeError(f"seed0 queue acceptance changed: {acceptance_path}")
            verify_bound_root_acceptance(acceptance_path, queue_path, queue_sha)
        for job in jobs:
            verify_receipt(queue_path, queue_sha, job)
            total_jobs += 1
            receipts.append(f"{job['id']}.json")
    gru_acceptance_path = ROOT / "protocols/p3-gru-replication-root-acceptance-v1.json"
    verify_bound_root_acceptance(gru_acceptance_path, gru_queue_path, gru_queue_sha)
    if sha(gru_acceptance_path) != gru.get("acceptance_sha256"):
        raise RuntimeError("GRU coordinator acceptance SHA is not bound")
    gru_queue = read(gru_queue_path)
    if len(gru_queue.get("jobs", [])) != 16:
        raise RuntimeError("expected exactly 16 completed GRU model-seed replications")
    for job in gru_queue["jobs"]:
        verify_receipt(gru_queue_path, gru_queue_sha, job)
        total_jobs += 1
        receipts.append(f"{job['id']}.json")
    if total_jobs != 56 or len(receipts) != 56 or set(receipts) != {f"{job_id}.json" for job_id in expected_ids}:
        raise RuntimeError("actual receipts do not equal the exact 56 accepted queue job IDs")
    return {"seed0_queues": [{"path": str(p), "sha256": h, "jobs": len(read(p)["jobs"])} for p, h in seed0_queues],
            "gru_queue": {"path": str(gru_queue_path), "sha256": gru_queue_sha, "jobs": 16},
            "receipt_count": total_jobs,
            "expected_receipt_ids": sorted(f"{job_id}.json" for job_id in expected_ids)}


def verify_seed0_numeric_acceptance(path: Path, seed0_summary: dict):
    """Require a root-authored six-row independent numeric acceptance; absence means wait."""
    acceptance = read(path)
    if (acceptance.get("accepted_by") != "root" or acceptance.get("verdict") != "pass"
            or acceptance.get("scope") != "P3 strong seed0 full train/validation independent numeric acceptance"
            or acceptance.get("pipeline_advanced") is not False
            or acceptance.get("scientific_hypothesis_accepted") is not False
            or acceptance.get("test_accessed") is not False):
        raise RuntimeError("seed0 numerical root acceptance is not pass/full train-validation/non-test")
    rows = acceptance.get("seed0_records")
    if not isinstance(rows, list) or len(rows) != 6:
        raise RuntimeError("seed0 numerical root acceptance must bind six dataset/method rows")
    expected = {(d, m) for d in DATASETS for m in METHODS}
    observed = set()
    for row in rows:
        key = (row.get("dataset"), row.get("method"))
        if key not in expected or key in observed or row.get("seed") != 0 or row.get("accepted") is not True:
            raise RuntimeError("seed0 acceptance row is duplicate, out of scope, or not explicitly accepted")
        observed.add(key)
        for path_key, hash_key in (("independent_review_path", "independent_review_sha256"),
                                   ("phase_receipt_path", "phase_receipt_sha256"),
                                   ("input_binding_path", "input_binding_sha256"),
                                   ("config_path", "config_sha256")):
            evidence_path = inside(row[path_key])
            if not evidence_path.is_file() or sha(evidence_path) != row[hash_key]:
                raise RuntimeError(f"seed0 accepted evidence changed: {evidence_path}")
            rel = evidence_path.relative_to(ROOT).as_posix()
            if acceptance.get("evidence_sha256", {}).get(rel) != row[hash_key]:
                raise RuntimeError(f"seed0 acceptance does not pin evidence: {rel}")
        # Root must attest the scientific review identity and scope on each row;
        # the referenced saved artifact is then checked against a supported schema.
        if (row.get("independent_review_verdict") != "pass"
                or row.get("independent_review_scope") != "full_train_validation_numeric"
                or row.get("independent_identity_verified_by") != "root"):
            raise RuntimeError("root row lacks typed independent-review pass/scope/identity attestation")
        review = inside(row["independent_review_path"])
        if review.suffix.lower() == ".json":
            report = read(review)
            exact_identity = (report.get("dataset") == row["dataset"]
                              and report.get("method") == row["method"]
                              and report.get("seed") == 0)
            actual_saved_review = (report.get("status") == "passed"
                                   and report.get("scope") == "independent_saved_validation_numeric_and_binding_review"
                                   and isinstance(report.get("provenance"), dict)
                                   and report["provenance"].get("test_accessed") is False)
            verifier_review = (report.get("reviewer_role") == "independent"
                               and report.get("verdict") == "pass"
                               and report.get("scope") == "full_train_validation_numeric"
                               and report.get("test_accessed") is False)
            if not exact_identity or not (actual_saved_review or verifier_review):
                raise RuntimeError(f"seed0 independent JSON review has unsupported identity/scope/no-test schema: {review}")
        else:
            if not review.read_text(encoding="utf-8-sig").strip():
                raise RuntimeError(f"seed0 Markdown independent review is empty: {review}")
    if observed != expected:
        raise RuntimeError("seed0 root acceptance omits a primary dataset/method pair")
    require_hash_map(acceptance.get("evidence_sha256"), exact_count=len(acceptance["evidence_sha256"]))
    if len(seed0_summary) != 4:
        raise RuntimeError("expected source receipts from all four completed seed0 queues")
    return {"acceptance_path": str(path), "acceptance_sha256": sha(path), "accepted_seed0_pairs": 6}


def require_p3():
    state_path = ROOT / "pipeline-state.json"
    rows = {row["id"]: row for row in read(state_path)["stages"]}
    if (rows["P2"].get("status") != "passed" or rows["P2"].get("gate_verdict") != "pass"
            or rows["P3"].get("status") != "running"):
        raise RuntimeError("strong replication requires canonical P2 passed/pass and P3 running")
    validator = ROOT / "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py"
    result = subprocess.run([str(PYTHON), str(validator), str(state_path), "--before", "P3"],
                            cwd=ROOT, capture_output=True, text=True, shell=False)
    if result.returncode:
        raise RuntimeError(f"official before-P3 validator rejected: {result.stdout[-1000:]} {result.stderr[-1000:]}")
    return sha(state_path)


def owned_model_processes():
    helper_path = ROOT / "edsgn/scripts/candidate_gpu_fullpath_probe_v2.py"
    spec = importlib.util.spec_from_file_location("_strong_continuation_process_scan", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load accepted owned-process scanner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._owned_model_processes(ROOT)


def require_idle_and_budget():
    lock_paths = [ROOT / "logs/model-resource.lock", ROOT / "logs/offline-phase-wall/campaign.lock"]
    present = [str(p) for p in lock_paths if p.exists()]
    if present:
        raise RuntimeError(f"resource locks are still held: {present}")
    active = owned_model_processes()
    if active:
        raise RuntimeError(f"owned model/resource process remains active: {active}")
    scripts = ROOT / "edsgn/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from aggregate_budget import aggregate
    from baseline_campaign import WALL_BUDGET_S
    budget = aggregate(ROOT)
    if not budget.get("coverage_complete") or budget.get("spent_seconds", WALL_BUDGET_S) >= WALL_BUDGET_S:
        raise RuntimeError(f"aggregate 72-hour budget/coverage refuses continuation: {budget.get('coverage_gaps')}")
    return {"budget_spent_seconds": budget["spent_seconds"], "budget_remaining_seconds": budget["remaining_seconds"],
            "coverage_complete": True, "locks_clear": True, "owned_model_processes": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-acceptance", default=str(SEED0_NUMERIC_ACCEPTANCE),
                        help="future root-authored six-pair independent numeric acceptance; absent file means waiting")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("use the selected local Conda interpreter")
    target = ROOT / "logs/p3-strong-serial-continuation-v4.json"
    if target.exists():
        raise FileExistsError(target)
    queue_sha = sha(QUEUE)
    if queue_sha != "97d4bd4350b52d8f76595e806f65d656d15ff3fd18825f0da896fc4cbb9b10d9":
        raise RuntimeError("strong queue SHA differs from root-accepted preparation")
    acceptance_info = verify_replication_acceptance(queue_sha)
    queue = verify_queue_and_configs(queue_sha, read(ROOT_ACCEPTANCE)["evidence_sha256"])
    require_coordinator_source_pins()
    own_sha = sha(Path(__file__))
    record = {
        "schema_version": 1, "scope": "root-authorized P3 primary strong-baseline seeds1-4 serial continuation",
        "status": "awaiting_seed0_and_gru_coordinators", "pid": os.getpid(),
        "queue_path": str(QUEUE), "queue_sha256": queue_sha,
        "root_acceptance_path": str(ROOT_ACCEPTANCE), "root_acceptance_sha256": acceptance_info["acceptance_sha256"],
        "root_pin_count": acceptance_info["pins_verified"],
        "seed0_numeric_acceptance_path": str(inside(args.seed0_acceptance)),
        "seed0_coordinator_path": str(SEED0_COORDINATOR), "gru_coordinator_path": str(GRU_COORDINATOR),
        "pinned_predecessor_coordinator_sources_sha256": COORDINATOR_SOURCE_PINS,
        "source_sha256": own_sha, "bound_controller_sha256": sha(ROOT / "edsgn/scripts/bound_phase_campaign.py"),
        "job_count": len(queue["jobs"]), "test_access": False, "training_started": False,
        "clock": "wait time is not charged; only the unchanged bound controller writes phase receipts against the shared cumulative 72-hour ledger",
        "required_seed0_phase_receipts": 40, "required_gru_phase_receipts": 16,
        "seed0_numeric_acceptance_schema": "root six-row acceptance; each row pins independent review, original phase receipt, at-launch binding, config, and typed independent_review_verdict/scope/identity attestation. JSON review schemas are strict saved-validation or standardized verifier formats; Markdown must be nonempty and is accepted only under the typed root row attestation.",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    def save():
        record["last_updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save_atomic_status(target, record)

    save()
    try:
        while True:
            require_coordinator_source_pins()
            if sha(Path(__file__)) != own_sha or sha(QUEUE) != queue_sha:
                raise RuntimeError("continuation source or prepared queue changed while awaiting predecessors")
            current_acceptance = verify_replication_acceptance(queue_sha)
            if (current_acceptance["acceptance_sha256"] != acceptance_info["acceptance_sha256"]
                    or current_acceptance["metadata_sha256"] != acceptance_info["metadata_sha256"]):
                raise RuntimeError("prepared replication root acceptance changed while awaiting predecessors")
            seed0 = read(SEED0_COORDINATOR)
            gru = read(GRU_COORDINATOR)
            if seed0.get("status") == "failed" or gru.get("status") == "failed":
                raise RuntimeError("a predecessor coordinator failed; preserve outputs and do not launch")
            if seed0.get("status") == "completed_all_baseline_queues" and gru.get("status") == "completed_gru_replications":
                break
            record.update(status="awaiting_seed0_and_gru_coordinators",
                          predecessor_status={"seed0": seed0.get("status"), "gru": gru.get("status")})
            save()
            time.sleep(5)

        predecessor = verify_coordinator_queues()
        record.update(status="awaiting_six_seed0_numeric_acceptances", predecessor_receipts=predecessor)
        save()
        while not inside(args.seed0_acceptance).is_file():
            require_coordinator_source_pins()
            if sha(Path(__file__)) != own_sha or sha(QUEUE) != queue_sha:
                raise RuntimeError("continuation source or prepared queue changed while awaiting numeric acceptance")
            time.sleep(5)
        numeric_acceptance = verify_seed0_numeric_acceptance(inside(args.seed0_acceptance), predecessor["seed0_queues"])
        p3_sha = require_p3()
        final_root_acceptance = verify_replication_acceptance(queue_sha)
        if (final_root_acceptance["acceptance_sha256"] != acceptance_info["acceptance_sha256"]
                or final_root_acceptance["metadata_sha256"] != acceptance_info["metadata_sha256"]):
            raise RuntimeError("prepared replication root acceptance changed before launch")
        queue = verify_queue_and_configs(queue_sha, read(ROOT_ACCEPTANCE)["evidence_sha256"])
        for job in queue["jobs"]:
            output = inside(job["output_dir"])
            if output.exists():
                raise RuntimeError(f"strong output path is no longer empty: {output}")
        guard = require_idle_and_budget()
        # Final prelaunch barriers: accepted pins/source/queue/state, receipts, and numeric review bytes.
        if sha(Path(__file__)) != own_sha or sha(QUEUE) != queue_sha:
            raise RuntimeError("strong continuation source/queue changed at launch barrier")
        if sha(inside(args.seed0_acceptance)) != numeric_acceptance["acceptance_sha256"]:
            raise RuntimeError("seed0 acceptance changed at launch barrier")
        if sha(ROOT / "pipeline-state.json") != p3_sha:
            raise RuntimeError("pipeline state changed after official before-P3 validation")
        verify_coordinator_queues()
        coordinator_snapshot = {str(path): path.read_bytes() for path in (SEED0_COORDINATOR, GRU_COORDINATOR)}
        coordinator_snapshot_sha = {key: hashlib.sha256(value).hexdigest() for key, value in coordinator_snapshot.items()}
        verify_replication_acceptance(queue_sha)
        require_coordinator_source_pins()
        require_idle_and_budget()
        for raw_path, raw_bytes in coordinator_snapshot.items():
            path = Path(raw_path)
            if path.read_bytes() != raw_bytes or sha(path) != coordinator_snapshot_sha[raw_path]:
                raise RuntimeError(f"validated coordinator log changed at final launch barrier: {path}")
        record.update(status="running_strong_replications", training_started=True,
                      seed0_numeric_acceptance=numeric_acceptance, pipeline_state_sha256=p3_sha,
                      final_root_acceptance_sha256=final_root_acceptance["acceptance_sha256"],
                      validated_coordinator_logs_sha256=coordinator_snapshot_sha,
                      prelaunch_guard=guard)
        save()
        child = subprocess.run([str(PYTHON), str(ROOT / "edsgn/scripts/bound_phase_campaign.py"),
                                "--queue", str(QUEUE)], cwd=ROOT, shell=False)
        record["returncode"] = child.returncode
        if child.returncode != 0:
            raise RuntimeError("strong replication campaign failed; preserve all phase receipts")
        verify_coordinator_queues()
        queue_obj = read(QUEUE)
        for job in queue_obj["jobs"]:
            verify_receipt(QUEUE, queue_sha, job)
        record.update(status="completed_strong_replications", training_started=True,
                      completed_jobs=24, test_access=False)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

