"""Audit the 80 logical P3 rows through the accepted host-recovery execution map.

This wrapper never launches a training job and never loads a model/checkpoint.
The unchanged numeric-v3 auditor scores only declared validation origins. A
combined chronological NPZ may be materialized as an array source, so reports
explicitly distinguish that fact from scoring/evaluating test origins.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
NUMERIC_AUDITOR = ROOT / "edsgn" / "scripts" / "audit_bound_baseline_numeric_v3.py"
REGISTRY = (
    ("strong8", "protocols/p3-bound-strong-baseline-queue-v3.json", "reviews/p3-strong-replication-root-acceptance-v1.json", 8),
    ("internal10", "protocols/p3-internal-bound-queue-v4.json", "protocols/p3-internal-bound-queue-root-acceptance-v4.json", 10),
    ("external16", "protocols/p3-external-bound-queue-v2.json", "protocols/p3-external-bound-queue-root-acceptance-v2.json", 16),
    ("fresh6", "protocols/p3-fresh-remaining-bound-queue-v1.json", "protocols/p3-fresh-remaining-bound-root-acceptance-v1.json", 6),
    ("gru16", "protocols/p3-gru-replication-queue-v1.json", "protocols/p3-gru-replication-root-acceptance-v1.json", 16),
    ("strong24", "protocols/p3-strong-replication-queue-v1.json", "reviews/p3-strong-replication-root-acceptance-v1.json", 24),
)
RECOVERY_QUEUE_REL = "protocols/p3-host-recovery-queue-v2.json"
RECOVERY_ACCEPTANCE_REL = "reviews/p3-host-recovery-root-acceptance-v2.json"
PREVIOUS_RECOVERY_ACCEPTANCE_REL = "reviews/p3-coverage-recovery-root-acceptance-v1.json"
TRAINING_PRINCIPAL = (
    "training_summary.json", "checkpoint.pt", "manifest.json", "config_resolved.json",
    "training_log.csv", "validation_predictions.npz", "metrics_by_horizon_val.csv",
)
TRAINING_BINDING = "input-binding-at-launch.json"
REQUIRED_HORIZONS_BY_DATASET = {
    "synthetic": [1, 3, 7],
    "email": [1, 3, 7],
    "email-episode": [1, 3, 7],
    "mooc": [1, 6, 24],
}


def _dataset_identity(row: dict[str, Any], config: dict[str, Any]) -> str:
    declared = row.get("dataset") or config.get("dataset") or config.get("dataset_id")
    source_name = Path(str(config.get("dataset_path", ""))).name.casefold()
    inferred = {
        "email.npz": "email",
        "email-episode.npz": "email-episode",
        "mooc.npz": "mooc",
    }.get(source_name)
    if source_name.startswith("synthetic-n64-"):
        inferred = "synthetic"
    if declared is not None and inferred is not None and declared != inferred:
        raise CampaignAuditError(f"dataset identity conflicts with dataset_path: {declared}/{inferred}")
    identity = declared or inferred
    if identity not in REQUIRED_HORIZONS_BY_DATASET:
        raise CampaignAuditError(f"unsupported/unknown dataset identity: {identity!r}")
    return str(identity)


class CampaignAuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pairs_no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise CampaignAuditError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs_no_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignAuditError(f"expected JSON object: {path}")
    return value


def contained(root: Path, value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not p.is_relative_to(root.resolve()):
        raise CampaignAuditError(f"path escapes run root: {p}")
    return p


def file_pin_map(value: Any) -> dict[str, str]:
    """Normalize actual acceptance maps, without inventing a new schema."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for path, digest in value.items():
        if isinstance(digest, str):
            result[str(path)] = digest.lower()
    return result


def _verify_acceptance(root: Path, queue_path: Path, acceptance_path: Path, queue_digest: str,
                       expected_n: int) -> dict[str, Any]:
    a = read_json(acceptance_path)
    if a.get("accepted") is False or a.get("verdict") not in (None, "pass"):
        raise CampaignAuditError(f"root acceptance is not accepted/pass: {acceptance_path}")
    if a.get("accepted_by") not in (None, "root"):
        raise CampaignAuditError(f"acceptance owner is not root: {acceptance_path}")
    pins = {}
    pins.update(file_pin_map(a.get("evidence_sha256")))
    pins.update(file_pin_map(a.get("file_inputs_sha256")))
    queue_rel = str(queue_path.relative_to(root)).replace("\\", "/")
    # Accepted schemas use either direct queue_path/queue or a queue pin in a file map.
    direct_path = a.get("queue_path", a.get("queue"))
    direct_hash = a.get("queue_sha256")
    queue_direct_ok = False
    if direct_path is not None:
        direct = contained(root, direct_path)
        queue_direct_ok = direct == queue_path.resolve() and direct_hash == queue_digest
    queue_pin_ok = any(str(k).replace("\\", "/").endswith(queue_rel) and v == queue_digest
                       for k, v in pins.items())
    if not queue_direct_ok and not queue_pin_ok:
        raise CampaignAuditError(f"acceptance has no matching queue pin: {acceptance_path}")
    for raw, digest in pins.items():
        p = contained(root, raw)
        if not p.is_file() or sha256(p) != digest:
            raise CampaignAuditError(f"acceptance input changed: {p}")
    # Some root records use a jobs array; when present it must agree in count.
    if queue_direct_ok:
        if isinstance(a.get("jobs"), list) and len(a["jobs"]) != expected_n:
            raise CampaignAuditError(f"acceptance job count mismatch: {acceptance_path}")
        if isinstance(a.get("jobs"), int) and a["jobs"] != expected_n:
            raise CampaignAuditError(f"acceptance scalar job count mismatch: {acceptance_path}")
        if a.get("job_count") is not None and a.get("job_count") != expected_n:
            raise CampaignAuditError(f"acceptance job_count mismatch: {acceptance_path}")
    return {"path": str(acceptance_path), "sha256": sha256(acceptance_path), "scope": a.get("scope")}


def _job_config(root: Path, row: dict[str, Any]) -> Path:
    config = row.get("config_path", row.get("config"))
    if not config:
        argv = row.get("args", [])
        if isinstance(argv, list) and "--config" in argv:
            i = argv.index("--config")
            config = argv[i + 1] if i + 1 < len(argv) else None
    if not config:
        raise CampaignAuditError(f"queue row has no config: {row.get('id')}")
    return contained(root, config)


def preflight_campaign(root: Path = ROOT) -> list[dict[str, Any]]:
    """Verify six accepted queue snapshots and return their normalized job rows."""
    root = root.resolve()
    all_ids: set[str] = set()
    queues = []
    total = 0
    for key, queue_rel, acceptance_rel, expected_n in REGISTRY:
        qp, ap = contained(root, queue_rel), contained(root, acceptance_rel)
        if not qp.is_file() or not ap.is_file():
            raise CampaignAuditError(f"missing queue/acceptance for {key}")
        qsha = sha256(qp)
        acceptance = _verify_acceptance(root, qp, ap, qsha, expected_n)
        q = read_json(qp)
        rows = q.get("jobs")
        if not isinstance(rows, list) or len(rows) != expected_n:
            raise CampaignAuditError(f"queue row count mismatch for {key}")
        jobs = []
        for row in rows:
            jid = row.get("id")
            if not isinstance(jid, str) or jid in all_ids:
                raise CampaignAuditError(f"missing/duplicate job id: {jid}")
            all_ids.add(jid)
            cfg = _job_config(root, row)
            if not cfg.is_file():
                raise CampaignAuditError(f"missing config: {cfg}")
            cfg_sha = sha256(cfg)
            declared = row.get("config_sha256")
            if declared and declared != cfg_sha:
                raise CampaignAuditError(f"queue config hash mismatch: {cfg}")
            config = read_json(cfg)
            if declared_dataset := row.get("dataset"):
                if config.get("dataset", config.get("dataset_id")) not in (None, declared_dataset):
                    raise CampaignAuditError(f"dataset identity mismatch: {cfg}")
            if row.get("variant") and config.get("variant") != row["variant"]:
                raise CampaignAuditError(f"variant identity mismatch: {cfg}")
            if row.get("seed") is not None and int(config.get("seed", -1)) != int(row["seed"]):
                raise CampaignAuditError(f"seed identity mismatch: {cfg}")
            if row.get("phase", "training") != "training":
                raise CampaignAuditError(f"non-training row in baseline queue: {jid}")
            out = contained(root, row["output_dir"])
            dataset_identity = _dataset_identity(row, config)
            jobs.append({"id": jid, "queue_key": key, "queue_path": str(qp), "queue_sha256": qsha,
                         "acceptance_path": acceptance["path"], "acceptance_sha256": acceptance["sha256"],
                         "config_path": str(cfg), "config_sha256": cfg_sha, "config": config,
                         "output_dir": str(out), "dataset": dataset_identity,
                         "phase": row.get("phase", "training"), "scope": row.get("scope"),
                         "variant": row.get("variant", config.get("variant")), "seed": int(config.get("seed", -1)),
                         "queue_inputs": row.get("inputs", [])})
        queues.append({"key": key, "path": str(qp), "sha256": qsha, "acceptance": acceptance, "jobs": jobs})
        total += len(jobs)
    if total != 80 or len(all_ids) != 80:
        raise CampaignAuditError(f"expected 80 unique P3 jobs; got {total}/{len(all_ids)}")
    return queues


def _verified_historical_ledger_authority(root: Path, queue_path: Path,
                                         acceptance_path: Path, queue_digest: str) -> dict[str, str]:
    """Verify the host-recovery record that preserves one superseded ledger.

    This is intentionally exact and narrow: the v1 coverage-recovery root
    record is the sole historical record whose old live-ledger path may be
    satisfied by the v2-preserved byte-identical snapshot. The host acceptance
    itself is fully checked first, including its current-ledger pin.
    """
    acceptance = _verify_acceptance(root, queue_path, acceptance_path, queue_digest, 68)
    doc = read_json(acceptance_path)
    if (doc.get("accepted_by") != "root" or doc.get("verdict") != "pass"
            or doc.get("scope") != "P3 baseline host recovery admission"
            or doc.get("queue_sha256") != queue_digest or doc.get("job_count") != 68
            or doc.get("test_access_authorized") is not False):
        raise CampaignAuditError("host recovery acceptance cannot authorize historical ledger verification")
    evidence = file_pin_map(doc.get("evidence_sha256"))
    previous_path = PREVIOUS_RECOVERY_ACCEPTANCE_REL
    previous = contained(root, previous_path)
    if evidence.get(previous_path) != sha256(previous):
        raise CampaignAuditError("host recovery acceptance does not pin the exact prior coverage acceptance")
    previous_doc = read_json(previous)
    previous_evidence = file_pin_map(previous_doc.get("evidence_sha256"))
    live_ledger_rel = "protocols/training-budget-manual-ledger.json"
    frozen_ledger_rel = "protocols/training-budget-manual-ledger-before-host-repair-v2.json"
    stale_digest = previous_evidence.get(live_ledger_rel)
    if (not isinstance(stale_digest, str) or len(stale_digest) != 64
            or evidence.get(frozen_ledger_rel) != stale_digest):
        raise CampaignAuditError("host recovery record does not map the exact old ledger pin to its preserved snapshot")
    frozen_ledger = contained(root, frozen_ledger_rel)
    if not frozen_ledger.is_file() or sha256(frozen_ledger) != stale_digest:
        raise CampaignAuditError("preserved pre-host-repair ledger bytes do not match the prior acceptance")
    # _verify_acceptance above has already checked the host record's live ledger
    # evidence pin against current bytes. Keep both the old and current ledgers
    # constrained; only the historical v1 record receives this one substitution.
    if live_ledger_rel not in evidence:
        raise CampaignAuditError("host recovery record omits the current live-ledger pin")
    return {"acceptance_path": str(acceptance["path"]), "live_ledger_path": live_ledger_rel,
            "historical_ledger_path": frozen_ledger_rel, "historical_ledger_sha256": stale_digest}


def _verify_root_evidence_record(root: Path, rel: str, *, expected_scope: str | None = None,
                                 historical_ledger_authority: dict[str, str] | None = None) -> dict[str, Any]:
    path = contained(root, rel)
    doc = read_json(path)
    if doc.get("accepted_by") != "root" or doc.get("verdict") != "pass":
        raise CampaignAuditError(f"root acceptance is not pass: {rel}")
    if expected_scope is not None and doc.get("scope") != expected_scope:
        raise CampaignAuditError(f"root acceptance scope mismatch: {rel}")
    if doc.get("pipeline_advanced") is not False or doc.get("scientific_hypothesis_accepted") is not False:
        raise CampaignAuditError(f"development root record must not advance pipeline or accept the hypothesis: {rel}")
    if doc.get("test_accessed") is True or doc.get("test_access_authorized") is True:
        raise CampaignAuditError(f"prior root record unexpectedly accessed/authorized test: {rel}")
    pins = file_pin_map(doc.get("evidence_sha256"))
    if not pins:
        raise CampaignAuditError(f"root acceptance lacks evidence hashes: {rel}")
    for raw, digest in pins.items():
        if (historical_ledger_authority is not None
                and rel == PREVIOUS_RECOVERY_ACCEPTANCE_REL
                and raw == historical_ledger_authority["live_ledger_path"]):
            if digest != historical_ledger_authority["historical_ledger_sha256"]:
                raise CampaignAuditError("prior acceptance old-ledger digest differs from the explicitly preserved snapshot")
            snapshot = contained(root, historical_ledger_authority["historical_ledger_path"])
            if not snapshot.is_file() or sha256(snapshot) != digest:
                raise CampaignAuditError("preserved historical ledger snapshot changed")
            continue
        target = contained(root, raw)
        if not target.is_file() or sha256(target) != digest:
            raise CampaignAuditError(f"root acceptance evidence changed: {target}")
    return {"path": str(path), "sha256": sha256(path), "document": doc}


def _logical_recovery_id(physical_id: str) -> str:
    if physical_id.startswith("host-recovery-v2-"):
        return physical_id.removeprefix("host-recovery-v2-")
    if physical_id.startswith("coverage-recovery-v1-"):
        return physical_id.removeprefix("coverage-recovery-v1-")
    raise CampaignAuditError(f"unexpected physical recovery ID: {physical_id}")


def project_recovery_jobs(original_jobs: list[dict[str, Any]], recovery_rows: list[dict[str, Any]],
                          accepted_logical_ids: set[str], root: Path = ROOT, *,
                          require_full_campaign: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map physical recovery rows to canonical logical queue rows without aliases."""
    root = root.resolve()
    originals = {str(row["id"]): row for row in original_jobs}
    if len(originals) != len(original_jobs):
        raise CampaignAuditError("duplicate canonical logical job ID")
    if not accepted_logical_ids.issubset(originals):
        raise CampaignAuditError("prior accepted row is outside canonical campaign")
    expected_pending = set(originals) - accepted_logical_ids
    mapped_ids: set[str] = set()
    physical_ids: set[str] = set()
    mapped: list[dict[str, Any]] = []
    mapping: list[dict[str, str]] = []
    for row in recovery_rows:
        physical = row.get("id")
        if not isinstance(physical, str) or physical in physical_ids:
            raise CampaignAuditError(f"missing/duplicate physical recovery ID: {physical}")
        physical_ids.add(physical)
        logical = _logical_recovery_id(physical)
        if logical not in expected_pending or logical in mapped_ids:
            raise CampaignAuditError(f"unexpected/duplicate logical recovery mapping: {physical}->{logical}")
        if require_full_campaign:
            expected_physical = ("host-recovery-v2-internal-bound-v4-email-V0-seed0"
                if logical == "internal-bound-v4-email-V0-seed0" else "coverage-recovery-v1-" + logical)
            if physical != expected_physical:
                raise CampaignAuditError(f"physical row does not match the frozen recovery mapping for {logical}")
        original = originals[logical]
        if row.get("phase") != original.get("phase", "training") or row.get("scope") != original.get("scope"):
            raise CampaignAuditError(f"recovery phase/scope differs from canonical row: {logical}")
        cfg = _job_config(root, row)
        old_cfg = Path(original["config_path"]).resolve()
        current = read_json(cfg)
        canonical = read_json(old_cfg)
        if {k: v for k, v in current.items() if k != "output_dir"} != {k: v for k, v in canonical.items() if k != "output_dir"}:
            raise CampaignAuditError(f"recovery changes model/training config beyond output_dir: {logical}")
        physical_output = contained(root, row["output_dir"])
        if contained(root, current.get("output_dir", "")) != physical_output:
            raise CampaignAuditError(f"recovery config output_dir mismatch: {logical}")
        mapped_job = dict(original)
        mapped_job.update({"id": physical, "logical_job_id": logical,
            "queue_key": "host_recovery_v2", "queue_path": str(contained(root, RECOVERY_QUEUE_REL)),
            "queue_sha256": row["_queue_sha256"], "acceptance_path": str(contained(root, RECOVERY_ACCEPTANCE_REL)),
            "acceptance_sha256": row["_acceptance_sha256"], "config_path": str(cfg),
            "config_sha256": sha256(cfg), "config": current, "output_dir": str(physical_output),
            "queue_inputs": row.get("inputs", [])})
        mapped.append(mapped_job)
        mapped_ids.add(logical)
        mapping.append({"logical_job_id": logical, "physical_job_id": physical,
                       "canonical_config_path": str(old_cfg), "recovery_config_path": str(cfg),
                       "physical_output_dir": str(physical_output)})
    if mapped_ids != expected_pending:
        raise CampaignAuditError(f"recovery mapping does not cover exact pending logical set: missing={sorted(expected_pending-mapped_ids)[:5]} extra={sorted(mapped_ids-expected_pending)[:5]}")
    if require_full_campaign and (len(original_jobs) != 80 or len(accepted_logical_ids) != 12 or len(mapped) != 68):
        raise CampaignAuditError(f"expected 80 logical rows = 12 root-accepted + 68 recovery rows; got {len(original_jobs)}/{len(accepted_logical_ids)}/{len(mapped)}")
    return mapped, mapping


def order_logical_rows(original_jobs: list[dict[str, Any]], rows_by_logical: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    expected = [str(job["id"]) for job in original_jobs]
    if len(expected) != len(set(expected)) or set(rows_by_logical) != set(expected):
        raise CampaignAuditError("logical results do not exactly cover the canonical job registry")
    return [rows_by_logical[jid] for jid in expected]


def preflight_recovery_campaign(root: Path = ROOT) -> dict[str, Any]:
    """Freeze the canonical 80-row registry and bind its 12 prior accepted rows."""
    root = root.resolve()
    original_queues = preflight_campaign(root)
    original_jobs = [j for q in original_queues for j in q["jobs"]]
    canonical_ids = {j["id"] for j in original_jobs}
    recovery_q_path = contained(root, RECOVERY_QUEUE_REL)
    recovery_a_path = contained(root, RECOVERY_ACCEPTANCE_REL)
    if not recovery_q_path.is_file() or not recovery_a_path.is_file():
        raise CampaignAuditError("host recovery queue or root acceptance is not yet available")
    qsha = sha256(recovery_q_path)
    historical_ledger_authority = _verified_historical_ledger_authority(
        root, recovery_q_path, recovery_a_path, qsha)
    old_acceptance = _verify_root_evidence_record(root, PREVIOUS_RECOVERY_ACCEPTANCE_REL,
        expected_scope="P3 baseline coverage recovery admission",
        historical_ledger_authority=historical_ledger_authority)
    previous = old_acceptance["document"]
    old_queue_path = contained(root, "protocols/p3-coverage-recovery-queue-v1.json")
    if previous.get("queue_path") != "protocols/p3-coverage-recovery-queue-v1.json" or previous.get("queue_sha256") != sha256(old_queue_path):
        raise CampaignAuditError("prior recovery acceptance is not bound to immutable v1 queue")
    accepted_rows = previous.get("reused_completed_rows")
    accepted_ids = {row.get("original_job_id") for row in accepted_rows or []}
    if len(accepted_ids) != 12 or None in accepted_ids or not accepted_ids.issubset(canonical_ids):
        raise CampaignAuditError("prior root acceptance does not identify exactly 12 canonical completed rows")
    prior_paths = previous.get("prior_numeric_root_acceptances")
    if not isinstance(prior_paths, list) or len(prior_paths) != 7:
        raise CampaignAuditError("prior root acceptance must bind the seven numeric acceptance records")
    prior_acceptances = [_verify_root_evidence_record(root, raw) for raw in prior_paths]
    for row in accepted_rows:
        receipt = contained(root, row["phase_receipt"])
        if not receipt.is_file() or sha256(receipt) != row.get("sha256"):
            raise CampaignAuditError(f"prior completed receipt changed: {receipt}")
        receipt_doc = read_json(receipt)
        if receipt_doc.get("status") != "completed" or receipt_doc.get("returncode") != 0:
            raise CampaignAuditError(f"prior completed receipt is not completed: {receipt}")
        canonical = next((j for j in original_jobs if j["id"] == row["original_job_id"]), None)
        if canonical is None or receipt_doc.get("id", receipt_doc.get("job_id")) != canonical["id"]:
            raise CampaignAuditError(f"prior accepted receipt does not match its canonical job: {receipt}")
        if (receipt_doc.get("queue_sha256") != canonical["queue_sha256"]
                or contained(root, receipt_doc.get("output_dir", "")) != Path(canonical["output_dir"]).resolve()):
            raise CampaignAuditError(f"prior accepted receipt queue/output identity mismatch: {receipt}")
    if len({row["phase_receipt"] for row in accepted_rows}) != 12:
        raise CampaignAuditError("duplicate prior completed receipt")

    acceptance = _verify_acceptance(root, recovery_q_path, recovery_a_path, qsha, 68)
    acceptance_doc = read_json(recovery_a_path)
    if (acceptance_doc.get("accepted_by") != "root" or acceptance_doc.get("verdict") != "pass"
            or acceptance_doc.get("scope") != "P3 baseline host recovery admission"
            or acceptance_doc.get("queue_sha256") != qsha
            or acceptance_doc.get("job_count") != 68
            or acceptance_doc.get("method_config_changes") is not False
            or acceptance_doc.get("test_access_authorized") is not False):
        raise CampaignAuditError("host recovery root acceptance fields do not authorize exact 68-row serial recovery")
    launcher_path = contained(root, "edsgn/scripts/launch_p3_host_recovery_root_v2.py")
    if acceptance_doc.get("source_sha256") != sha256(launcher_path):
        raise CampaignAuditError("host recovery acceptance is not bound to the exact launcher source")
    recovery_queue = read_json(recovery_q_path)
    if len(recovery_queue.get("jobs", [])) != 68:
        raise CampaignAuditError("host recovery queue row count mismatch")
    pin_rows = recovery_queue.get("preparation_source_bindings", [])
    if not isinstance(pin_rows, list) or len(pin_rows) != 301:
        raise CampaignAuditError("host recovery queue must retain all 301 preparation source pins")
    for entry in pin_rows:
        p = contained(root, entry["path"])
        if not p.is_file() or sha256(p) != entry.get("sha256"):
            raise CampaignAuditError(f"host recovery preparation source changed: {p}")
    rows = []
    for row in recovery_queue["jobs"]:
        rows.append({**row, "_queue_sha256": qsha, "_acceptance_sha256": acceptance["sha256"]})
    mapped, mapping = project_recovery_jobs(original_jobs, rows, accepted_ids, root)
    # There were two distinct stopped attempts. The first was a measured
    # budget-coverage stop on the original queue (known returncode=1); the
    # second was interrupted during host-session recovery and was reconciled
    # only as process-absent, with no inferred return code or finish time.
    first_stop = contained(root, "logs/offline-phase-wall/internal-bound-v4-email-V0-seed0.json")
    second_stop = contained(root, "logs/offline-phase-wall/coverage-recovery-v1-internal-bound-v4-email-V0-seed0.json")
    second_raw = contained(root, "logs/host-interruption-evidence-v2/raw-running-phase.json")
    interruption = contained(root, "logs/p3-host-interruption-root-v2.json")
    if not all(p.is_file() for p in (first_stop, second_stop, second_raw, interruption)):
        raise CampaignAuditError("one or more of the two stopped-attempt records are missing")
    first_doc, second_doc = read_json(first_stop), read_json(second_stop)
    raw_doc, interruption_doc = read_json(second_raw), read_json(interruption)
    evidence_pins = file_pin_map(acceptance_doc.get("evidence_sha256"))
    expected_pins = {
        "logs/offline-phase-wall/internal-bound-v4-email-V0-seed0.json": sha256(first_stop),
        "logs/offline-phase-wall/coverage-recovery-v1-internal-bound-v4-email-V0-seed0.json": sha256(second_stop),
        "logs/host-interruption-evidence-v2/raw-running-phase.json": sha256(second_raw),
        "logs/p3-host-interruption-root-v2.json": sha256(interruption),
    }
    if any(evidence_pins.get(path) != digest for path, digest in expected_pins.items()):
        raise CampaignAuditError("host recovery acceptance does not pin both exact stopped attempts and reconciliation evidence")
    canonical_first = next((j for j in original_jobs if j["id"] == "internal-bound-v4-email-V0-seed0"), None)
    if canonical_first is None:
        raise CampaignAuditError("canonical first budget-stop job is missing")
    if (first_doc.get("status") != "stopped" or first_doc.get("returncode") != 1
            or first_doc.get("finished_utc") is None
            or first_doc.get("queue_sha256") != canonical_first["queue_sha256"]
            or contained(root, first_doc.get("output_dir", "")) != Path(canonical_first["output_dir"]).resolve()
            or not isinstance(first_doc.get("wall_seconds"), (int, float))
            or first_doc["wall_seconds"] <= 0):
        raise CampaignAuditError("first measured budget-stop receipt does not match its original logical job")
    raw_rel = "logs/host-interruption-evidence-v2/raw-running-phase.json"
    interruption_rel = "logs/p3-host-interruption-root-v2.json"
    expected_second_output = contained(root, "runs/p3-coverage-recovery-v1/email-V0-seed0")
    if (second_doc.get("status") != "stopped" or second_doc.get("returncode") is not None
            or second_doc.get("finished_utc") is not None or raw_doc.get("status") != "running"
            or raw_doc.get("returncode") is not None or raw_doc.get("finished_utc") is not None
            or second_doc.get("wall_seconds") != raw_doc.get("wall_seconds")
            or second_doc.get("id", second_doc.get("job_id")) != raw_doc.get("id", raw_doc.get("job_id"))
            or second_doc.get("job_id") != "coverage-recovery-v1-internal-bound-v4-email-V0-seed0"
            or raw_doc.get("job_id") != second_doc.get("job_id")
            or second_doc.get("queue_sha256") != raw_doc.get("queue_sha256")
            or second_doc.get("queue_sha256") != sha256(contained(root, "protocols/p3-coverage-recovery-queue-v1.json"))
            or contained(root, second_doc.get("output_dir", "")) != expected_second_output
            or contained(root, raw_doc.get("output_dir", "")) != expected_second_output):
        raise CampaignAuditError("second interrupted attempt is not an exact stopped alias of its raw running receipt")
    reconciled = second_doc.get("root_reconciliation")
    if (not isinstance(reconciled, dict)
            or reconciled.get("raw_receipt_path") != str(second_raw.resolve())
            or reconciled.get("raw_receipt_sha256") != sha256(second_raw)
            or reconciled.get("no_training_completion_inferred") is not True
            or interruption_doc.get("raw_phase_sha256") != sha256(second_raw)
            or interruption_doc.get("raw_phase_receipt") != str(second_raw.resolve())
            or interruption_doc.get("measured_controller_seconds") != raw_doc.get("wall_seconds")
            or interruption_doc.get("total_budget_allowance_seconds") != 158.0):
        raise CampaignAuditError("second interrupted attempt root reconciliation does not bind its exact observed evidence")
    return {"original_queues": original_queues, "original_jobs": original_jobs,
        "accepted_ids": accepted_ids, "accepted_rows": accepted_rows,
        "prior_acceptance": old_acceptance, "prior_numeric_acceptances": prior_acceptances,
        "recovery_queue_path": str(recovery_q_path), "recovery_queue_sha256": qsha,
        "recovery_acceptance_path": str(recovery_a_path), "recovery_acceptance_sha256": acceptance["sha256"],
        "recovery_jobs": mapped, "mapping": mapping,
        "superseded_stopped_attempts":[
          {"attempt":"original_internal_email_v0_budget_stop","receipt_path":str(first_stop),
           "receipt_sha256":sha256(first_stop),"status":"stopped","returncode":1,
           "wall_seconds":first_doc["wall_seconds"],"budget_charge_is_in_ledger":True,
           "logical_completion":False},
          {"attempt":"coverage_recovery_v1_host_interruption","receipt_path":str(second_stop),
           "receipt_sha256":sha256(second_stop),"raw_receipt_path":str(second_raw),
           "raw_receipt_sha256":sha256(second_raw),"reconciliation_path":str(interruption),
           "reconciliation_sha256":sha256(interruption),"status":"stopped",
           "returncode":None,"finished_utc":None,"measured_wall_seconds":raw_doc["wall_seconds"],
           "conservative_supplement_seconds":interruption_doc["conservative_supplement_seconds"],
           "total_budget_allowance_seconds":interruption_doc["total_budget_allowance_seconds"],
           "no_training_completion_inferred":True,"logical_completion":False}]}


def _tree_hashes(path: Path) -> dict[str, str]:
    if not path.is_dir():
        return {}
    return {str(p.resolve()): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}


def _receipt_state(root: Path, job: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    receipt_path = contained(root, Path("logs/offline-phase-wall") / f"{job['id']}.json")
    if not receipt_path.is_file():
        return "pending", None, "phase receipt absent"
    try:
        receipt = read_json(receipt_path)
    except CampaignAuditError as exc:
        return "failed", None, str(exc)
    status = receipt.get("status")
    if status in ("running", "queued", "prepared", "waiting"):
        return "pending", receipt, f"receipt status={status}"
    if status != "completed" or receipt.get("returncode") != 0:
        return "failed", receipt, f"receipt status/returncode={status}/{receipt.get('returncode')}"
    return "completed", receipt, None


def _verify_completed(root: Path, job: dict[str, Any], receipt: dict[str, Any],
                      binding_verifier: Callable | None = None) -> dict[str, Any]:
    out = Path(job["output_dir"]).resolve()
    if receipt.get("scope") != "offline-experiment-phase" or receipt.get("phase") != "training":
        raise CampaignAuditError("receipt scope/phase mismatch")
    if receipt.get("id", receipt.get("job_id")) != job["id"]:
        raise CampaignAuditError("receipt job identity mismatch")
    if receipt.get("queue_sha256") != job["queue_sha256"]:
        raise CampaignAuditError("receipt queue SHA mismatch")
    if contained(root, receipt.get("output_dir", "")) != out:
        raise CampaignAuditError("receipt output_dir mismatch")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise CampaignAuditError("receipt lacks input hash list")
    input_hashes = {}
    for item in inputs:
        p = contained(root, item["path"])
        if not p.is_file() or sha256(p) != item.get("sha256"):
            raise CampaignAuditError(f"phase input changed: {p}")
        input_hashes[str(p)] = item["sha256"]
    requested_config = Path(job["config_path"]).resolve()
    if str(requested_config) not in input_hashes or input_hashes[str(requested_config)] != job["config_sha256"]:
        raise CampaignAuditError("phase receipt does not directly bind the exact requested config path and SHA")
    for declared in job.get("queue_inputs", []):
        source = contained(root, declared)
        if source.is_dir():
            required_files = [p.resolve() for p in source.rglob("*") if p.is_file()]
        elif source.is_file():
            required_files = [source]
        else:
            raise CampaignAuditError(f"queue-declared input is missing: {source}")
        for required in required_files:
            if str(required) not in input_hashes or sha256(required) != input_hashes[str(required)]:
                raise CampaignAuditError(f"queue-declared input is absent from phase binding: {required}")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list):
        raise CampaignAuditError("receipt lacks output hash list")
    output_hashes = {}
    for item in outputs:
        p = contained(root, item["path"])
        if not p.is_file() or sha256(p) != item.get("sha256"):
            raise CampaignAuditError(f"phase output changed: {p}")
        if not p.is_relative_to(out):
            raise CampaignAuditError(f"phase output outside job output_dir: {p}")
        output_hashes[str(p)] = item["sha256"]
    for name in TRAINING_PRINCIPAL:
        p = out / name
        if not p.is_file() or str(p.resolve()) not in output_hashes:
            raise CampaignAuditError(f"required phase output is unbound: {name}")
    snapshot = _tree_hashes(out)
    if not snapshot:
        raise CampaignAuditError("job output directory is empty")
    summary = read_json(out / "training_summary.json")
    manifest = read_json(out / "manifest.json")
    resolved = read_json(out / "config_resolved.json")
    if resolved != job["config"]:
        raise CampaignAuditError("config_resolved.json differs from the complete requested config")
    if contained(root, manifest.get("dataset_path", "")) != contained(root, job["config"]["dataset_path"]):
        raise CampaignAuditError("manifest dataset path differs from complete requested config")
    if summary.get("scope") != "full_protocol" or manifest.get("scope") != "full_protocol":
        raise CampaignAuditError("training summary/manifest not full_protocol")
    if summary.get("test_accessed") is not False or manifest.get("test_accessed") not in (None, False):
        raise CampaignAuditError("training provenance does not explicitly prohibit test access")
    if summary.get("stop_loss_reached") is not False:
        raise CampaignAuditError("stop_loss_reached must explicitly be false")
    for key in ("dataset_path", "variant", "seed", "history", "hmax"):
        if key in job["config"] and resolved.get(key) != job["config"].get(key):
            raise CampaignAuditError(f"resolved config mismatch at {key}")
    cp_sha = sha256(out / "checkpoint.pt")
    if summary.get("checkpoint_sha256") != cp_sha:
        raise CampaignAuditError("summary checkpoint hash mismatch")
    train_windows = manifest.get("train_windows")
    val_windows = manifest.get("val_windows")
    epochs = summary.get("epochs_executed")
    if not all(isinstance(v, int) and v > 0 for v in (train_windows, val_windows, epochs)):
        raise CampaignAuditError("missing/nonpositive train/val windows or epochs")
    if epochs > int(job["config"].get("epochs", epochs)):
        raise CampaignAuditError("executed epochs exceed config")
    binding_path = out / TRAINING_BINDING
    if not binding_path.is_file() or binding_path.resolve().parent != out:
        raise CampaignAuditError("at-launch binding is missing or outside the phase output directory")
    binding = read_json(binding_path)
    config_source = binding.get("config_source")
    if (not isinstance(config_source, dict) or not config_source.get("path")
            or contained(root, config_source["path"]) != Path(job["config_path"]).resolve()
            or config_source.get("sha256") != job["config_sha256"]
            or config_source.get("config") != job["config"]):
        raise CampaignAuditError("at-launch binding is not pinned to the exact requested config")
    if contained(root, binding.get("dataset_root", "")) != contained(root, job["config"]["dataset_path"]):
        raise CampaignAuditError("at-launch binding dataset root differs from requested config")
    if binding_verifier is None:
        bind_path = ROOT / "edsgn" / "scripts" / "bind_training_inputs.py"
        spec = importlib.util.spec_from_file_location("batch_audit_v2_binding_verifier", bind_path)
        if spec is None or spec.loader is None:
            raise CampaignAuditError("cannot load at-launch binding verifier")
        bind_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bind_module)
        binding_verifier = bind_module.verify_binding
    binding_result = binding_verifier(binding_path)
    if not isinstance(binding_result, dict) or binding_result.get("status") != "verified":
        raise CampaignAuditError("at-launch binding verification failed")
    binding_sha = sha256(binding_path)
    return {"receipt_path": str(contained(root, Path("logs/offline-phase-wall") / f"{job['id']}.json")),
            "receipt_sha256": sha256(contained(root, Path("logs/offline-phase-wall") / f"{job['id']}.json")),
            "input_hashes": input_hashes, "phase_output_hashes": output_hashes,
            "output_snapshot": snapshot, "summary": summary, "manifest": manifest,
            "config_resolved": resolved, "checkpoint_sha256": cp_sha,
            "training_summary_sha256": sha256(out / "training_summary.json"),
            "manifest_sha256": sha256(out / "manifest.json"),
            "config_resolved_sha256": sha256(out / "config_resolved.json"),
            "binding_path": str(binding_path.resolve()), "binding_sha256": binding_sha,
            "binding_verification": binding_result,
            "train_windows": train_windows, "val_windows": val_windows, "epochs": epochs}


def _safe_for_new_numeric(config: dict[str, Any]) -> bool:
    """Validate only bundle metadata; return whether source NPZ is combined."""
    dataset = Path(config["dataset_path"]).resolve()
    if dataset.is_file():
        if dataset.suffix.lower() != ".npz":
            raise CampaignAuditError("new numeric audit blocked: unsupported single-file dataset")
        import numpy as np
        with np.load(dataset, allow_pickle=False) as z:
            if "metadata_json" not in z.files:
                raise CampaignAuditError("combined dataset metadata_json missing")
            meta = json.loads(str(np.asarray(z["metadata_json"]).item()))
        split = meta.get("splits", {})
        if (meta.get("partition") != "chronological" or not isinstance(split, dict)
                or not all(k in split for k in ("train", "validation"))):
            raise CampaignAuditError("single NPZ is not a declared chronological train/validation source")
        return True
    if not dataset.is_dir():
        raise CampaignAuditError("new numeric audit blocked: dataset path missing")
    train = sorted(dataset.glob("train*.npz"))
    val = sorted(dataset.glob("validation*.npz"))
    if not train or not val:
        raise CampaignAuditError("new numeric audit blocked: split-separated train/validation archives absent")
    if any("test" in p.name.casefold() for p in [*train, *val]):
        raise CampaignAuditError("new numeric audit blocked: test archive appeared in train/validation selection")
    # Metadata only: reject archives whose partition is incompatible, without touching target arrays.
    import numpy as np
    for expected, files in (("train", train), ("validation", val)):
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                if "metadata_json" not in z.files:
                    raise CampaignAuditError(f"new numeric audit blocked: metadata absent in {path}")
                meta = json.loads(str(np.asarray(z["metadata_json"]).item()))
            part = meta.get("partition")
            if part not in (expected, "chronological"):
                raise CampaignAuditError(f"new numeric audit blocked: {path.name} declares partition={part}")
    return False


def _numeric_valid(report: dict[str, Any], job: dict[str, Any], completed: dict[str, Any],
                   binding_verifier: Callable | None = None) -> bool:
    required_horizons = REQUIRED_HORIZONS_BY_DATASET.get(str(job.get("dataset", "")))
    if required_horizons is None:
        return False
    if not (
        report.get("status") == "passed" and report.get("config_path") == job["config_path"]
        and report.get("run_dir") == job["output_dir"] and report.get("variant") == job["variant"]
        and report.get("seed") == job["seed"] and report.get("checkpoint_sha256") == completed["checkpoint_sha256"]
        and report.get("origins_complete_and_ordered") is True
        and report.get("saved_targets_and_masks_match_validation_source") is True
        and report.get("test_arrays_or_test_windows_read") is False
        and report.get("split") == "validation"
        and Path(report.get("dataset_path", "")).resolve() == Path(job["config"].get("dataset_path", "")).resolve()
        and report.get("n_validation_origins") == completed["val_windows"]
        and report.get("train_origins_recomputed_from_metadata") == completed["train_windows"]
        and report.get("required_horizons") == required_horizons
        and bool(report.get("validation_source_files")) and bool(report.get("training_metadata_source_files"))
        and all(Path(row["path"]).is_file() and sha256(Path(row["path"])) == row["sha256"]
                for row in report.get("validation_source_files", []))
        and all(Path(row["path"]).is_file() and sha256(Path(row["path"])) == row["sha256"]
                for row in report.get("training_metadata_source_files", []))
    ):
        return False
    training = report.get("training", {})
    if (training.get("epochs_executed") != completed["epochs"]
            or training.get("train_windows_per_epoch") != completed["train_windows"]
            or training.get("val_windows") != completed["val_windows"]
            or training.get("summary_best_val_mae_matches_log_minimum_and_saved_predictions") is not True
            or not report.get("required_horizon_metric_csv_checks")
            or any(row.get("matches_recomputed") is not True
                   for row in report.get("required_horizon_metric_csv_checks", []))
            or not report.get("required_horizon_feature_rows")):
        return False
    binding_info = report.get("binding", {})
    binding_path = Path(binding_info.get("binding_path", ""))
    expected_binding = Path(job["output_dir"]) / "input-binding-at-launch.json"
    if (binding_info.get("status") != "verified" or not binding_path.is_file() or binding_path.resolve() != expected_binding.resolve()
            or sha256(binding_path) != completed["binding_sha256"]):
        return False
    # Verify current source/data/config binding with the authoritative verifier.
    try:
        if binding_verifier:
            verification = binding_verifier(binding_path)
        else:
            bind_path = ROOT / "edsgn" / "scripts" / "bind_training_inputs.py"
            spec = importlib.util.spec_from_file_location("batch_audit_binding_verifier", bind_path)
            if spec is None or spec.loader is None:
                return False
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            verification = module.verify_binding(binding_path)
        if not isinstance(verification, dict) or verification.get("status") != "verified":
            return False
    except Exception:
        return False
    return True


def _source_array_may_materialize_other_partitions(config: dict[str, Any]) -> bool:
    dataset = Path(config.get("dataset_path", ""))
    return dataset.is_file() and dataset.suffix.lower() == ".npz"


def _load_numeric_auditor(path: Path):
    spec = importlib.util.spec_from_file_location("bound_numeric_auditor_v3", path)
    if spec is None or spec.loader is None:
        raise CampaignAuditError("cannot import v3 numeric audit source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary_numeric(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("required_horizon_feature_rows", [])
    metrics = report.get("required_horizon_pooled_sum_count", [])
    return {"required_horizon_feature_rows": rows,
            "required_horizon_mean_selection_mae": report.get("required_horizon_mean_selection_mae"),
            "required_horizon_pooled_sum_count": metrics}


def audit_job(root: Path, job: dict[str, Any], numeric_fn: Callable | None = None,
              reuse_path: Path | None = None, binding_verifier: Callable | None = None) -> dict[str, Any]:
    state, receipt, why = _receipt_state(root, job)
    receipt_path = contained(root, Path("logs/offline-phase-wall") / f"{job['id']}.json")
    base = {k: job[k] for k in ("id", "queue_key", "queue_path", "queue_sha256", "acceptance_path",
                                  "acceptance_sha256", "config_path", "config_sha256", "output_dir",
                                  "dataset", "variant", "seed")}
    base["scored_test_origins"] = False
    base["source_array_may_materialize_other_partitions"] = _source_array_may_materialize_other_partitions(job["config"])
    if state != "completed":
        return {**base, "status": state, "reason": why,
                "phase_receipt_path": str(receipt_path),
                "phase_receipt_sha256": sha256(receipt_path) if receipt_path.is_file() else None,
                "observed_receipt_status": receipt.get("status") if receipt else None,
                "observed_returncode": receipt.get("returncode") if receipt else None}
    try:
        completed = _verify_completed(root, job, receipt, binding_verifier=binding_verifier)
        existing = None
        existing_path = None
        candidates = [reuse_path] if isinstance(reuse_path, Path) else list(reuse_path or [])
        for candidate_path in candidates:
            if candidate_path is not None and candidate_path.is_file():
                candidate = read_json(candidate_path)
                if _numeric_valid(candidate, job, completed, binding_verifier=binding_verifier):
                    existing, existing_path = candidate, candidate_path
                    break
        if existing is not None:
            numeric = existing
            numeric_path = existing_path.resolve()
            numeric_status = "reused"
        else:
            _safe_for_new_numeric(job["config"])
            run_snapshot_before = completed["output_snapshot"]
            if numeric_fn:
                numeric = numeric_fn(job["config_path"], job["output_dir"])
            else:
                module = _load_numeric_auditor(NUMERIC_AUDITOR)
                numeric = module.audit(job["config_path"], job["output_dir"])
            if numeric.get("status") != "passed":
                raise CampaignAuditError("validation-only numeric audit did not pass")
            if not _numeric_valid(numeric, job, completed, binding_verifier=binding_verifier):
                raise CampaignAuditError("new numeric audit failed exact config/binding/window/source validation")
            if _tree_hashes(Path(job["output_dir"])) != run_snapshot_before:
                raise CampaignAuditError("numeric audit changed job output files")
            dest_dir = root / "logs" / "completed-p3-numeric-audits" / job["queue_key"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            numeric_path = dest_dir / f"{job['id']}.json"
            if numeric_path.exists():
                raise CampaignAuditError(f"numeric audit output already exists: {numeric_path}")
            numeric_path.write_text(json.dumps(numeric, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            persisted_numeric = read_json(numeric_path)
            if not _numeric_valid(persisted_numeric, job, completed, binding_verifier=binding_verifier):
                raise CampaignAuditError("saved numeric report failed the same reuse validation")
            numeric = persisted_numeric
            numeric_status = "audited"
        # Recheck all relevant immutable artifacts after numeric work/reuse.
        if _tree_hashes(Path(job["output_dir"])) != completed["output_snapshot"]:
            raise CampaignAuditError("job output changed during numeric audit")
        if sha256(Path(job["config_path"])) != job["config_sha256"]:
            raise CampaignAuditError("config changed during numeric audit")
        if sha256(Path(completed["receipt_path"])) != completed["receipt_sha256"]:
            raise CampaignAuditError("phase receipt changed during numeric audit")
        for path, digest in completed["input_hashes"].items():
            if not Path(path).is_file() or sha256(Path(path)) != digest:
                raise CampaignAuditError(f"phase input changed during numeric audit: {path}")
        return {**base, "status": numeric_status, "phase_receipt_path": completed["receipt_path"],
                "phase_receipt_sha256": completed["receipt_sha256"], "phase_input_hashes": completed["input_hashes"],
                "phase_output_hashes": completed["phase_output_hashes"], "output_directory_file_hashes": completed["output_snapshot"],
                "config_resolved_sha256": completed["config_resolved_sha256"],
                "training_summary_sha256": completed["training_summary_sha256"], "manifest_sha256": completed["manifest_sha256"],
                "checkpoint_sha256": completed["checkpoint_sha256"], "numeric_report_path": str(numeric_path),
                "numeric_report_sha256": sha256(numeric_path), "train_windows": completed["train_windows"],
                "val_windows": completed["val_windows"], "epochs": completed["epochs"],
                **_summary_numeric(numeric),
                "training_run_declared_test_accessed": completed["summary"].get("test_accessed"),
                "scored_test_origins": False,
                "source_array_may_materialize_other_partitions": _source_array_may_materialize_other_partitions(job["config"]),
                "numeric_source_mode": "combined_chronological" if _source_array_may_materialize_other_partitions(job["config"]) else "split_separated",
                "combined_source_array_may_have_materialized_nonvalidation_partitions": _source_array_may_materialize_other_partitions(job["config"]),
                "interpretation": "development validation audit only; no root acceptance, phase pass, five-seed, test, or scientific claim"}
    except Exception as exc:
        return {**base, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def run_campaign(root: Path = ROOT, output: Path | None = None, numeric_fn: Callable | None = None,
                 reuse_resolver: Callable[[dict[str, Any]], Path | None] | None = None) -> dict[str, Any]:
    root = root.resolve()
    source_sha_before = sha256(Path(__file__).resolve())
    auditor_sha_before = sha256(NUMERIC_AUDITOR)
    queues = preflight_campaign(root)
    rows = []
    for queue in queues:
        for job in queue["jobs"]:
            reuse = reuse_resolver(job) if reuse_resolver else sorted((root / "logs").glob("p3-*-numeric-v3.json"))
            rows.append(audit_job(root, job, numeric_fn=numeric_fn, reuse_path=reuse))
    for queue in queues:
        if sha256(Path(queue["path"])) != queue["sha256"]:
            raise CampaignAuditError(f"queue changed during campaign audit: {queue['path']}")
        acceptance = queue.get("acceptance")
        if not isinstance(acceptance, dict) or not isinstance(acceptance.get("path"), str) or not isinstance(acceptance.get("sha256"), str):
            raise CampaignAuditError(f"normalized queue lacks nested acceptance ref: {queue.get('key')}")
        acceptance_path = Path(acceptance["path"])
        if sha256(acceptance_path) != acceptance["sha256"]:
            raise CampaignAuditError(f"root acceptance changed during campaign audit: {acceptance_path}")
        for job in queue["jobs"]:
            if sha256(Path(job["config_path"])) != job["config_sha256"]:
                raise CampaignAuditError(f"job config changed during campaign audit: {job['config_path']}")
    if sha256(Path(__file__).resolve()) != source_sha_before or sha256(NUMERIC_AUDITOR) != auditor_sha_before:
        raise CampaignAuditError("audit source changed during campaign")
    payload = {"schema_version": 1, "scope": "completed_p3_development_validation_audit",
               "status": "completed_with_pending_or_failed_rows" if any(r["status"] in ("pending", "failed") for r in rows) else "all_rows_audited_or_reused",
               "total_jobs": len(rows), "counts": {s: sum(r["status"] == s for r in rows) for s in ("reused", "audited", "pending", "failed")},
               "source_sha256": source_sha_before, "numeric_auditor_sha256": auditor_sha_before,
               "scored_test_origins": False,
               "source_array_may_materialize_other_partitions": any(r.get("source_array_may_materialize_other_partitions") for r in rows),
               "combined_archive_disclosure": "Validation origins only are scored. The unchanged numeric-v3 auditor may materialize complete NPZ array members for a chronological combined source; this wrapper does not claim test bytes were untouched.",
               "rows": rows,
               "interpretation": "metadata and saved-validation numeric audit only; no root independent acceptance, no phase pass, no five-seed conclusion, no test evaluation, no scientific claim"}
    if output is not None:
        dest = contained(root, output)
        if dest.exists():
            raise CampaignAuditError(f"refusing to overwrite {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        with dest.open("xb") as f:
            f.write(data)
        if sha256(dest) != hashlib.sha256(data).hexdigest():
            raise CampaignAuditError("aggregate output failed post-write hash check")
    return payload


def run_recovered_campaign(root: Path = ROOT, output: Path | None = None,
                           numeric_fn: Callable | None = None,
                           reuse_resolver: Callable[[dict[str, Any]], Path | None] | None = None) -> dict[str, Any]:
    """Run the v8 logical-80 audit after the exact host-recovery map is accepted."""
    root = root.resolve()
    source_sha_before = sha256(Path(__file__).resolve())
    auditor_sha_before = sha256(NUMERIC_AUDITOR)
    plan = preflight_recovery_campaign(root)
    original_by_id = {j["id"]: j for j in plan["original_jobs"]}
    acceptance_for = {}
    for accepted in plan["prior_numeric_acceptances"]:
        doc = accepted["document"]
        rows = doc.get("seed0_records")
        if isinstance(rows, list):
            identities = [(r.get("dataset"), r.get("method", r.get("variant")), r.get("seed")) for r in rows if r.get("accepted") is True]
        else:
            identities = [(doc.get("dataset"), doc.get("variant"), doc.get("model_seed", doc.get("seed")))]
        for dataset, variant, seed in identities:
            matches = [j for j in plan["original_jobs"] if j["dataset"] == dataset and j["variant"] == variant and j["seed"] == seed]
            if len(matches) == 1:
                jid = matches[0]["id"]
                if jid in acceptance_for:
                    raise CampaignAuditError(f"multiple root acceptance records map to one logical job: {jid}")
                acceptance_for[jid] = {"path": accepted["path"], "sha256": accepted["sha256"]}
    if set(acceptance_for) != plan["accepted_ids"]:
        raise CampaignAuditError("seven prior root acceptance records do not map exactly to the accepted 12 logical jobs")
    rows_by_logical = {}
    for logical_id in sorted(plan["accepted_ids"]):
        job = original_by_id[logical_id]
        receipt_row = next(r for r in plan["accepted_rows"] if r["original_job_id"] == logical_id)
        rows_by_logical[logical_id] = {k: job[k] for k in ("id", "queue_key", "queue_path", "queue_sha256", "acceptance_path",
            "acceptance_sha256", "config_path", "config_sha256", "output_dir", "dataset", "variant", "seed")} | {
            "logical_job_id": logical_id, "status": "root_accepted_reused",
            "phase_receipt_path": str(contained(root, receipt_row["phase_receipt"])),
            "phase_receipt_sha256": receipt_row["sha256"],
            "numeric_root_acceptance": acceptance_for[logical_id],
            "scored_test_origins": False,
            "interpretation": "prior root-accepted development validation row reused by exact receipt/acceptance reference; no numeric values re-read here"}
    for job in plan["recovery_jobs"]:
        candidates = reuse_resolver(job) if reuse_resolver else sorted((root / "logs").glob("p3-*-numeric-v3.json"))
        audited = audit_job(root, job, numeric_fn=numeric_fn, reuse_path=candidates)
        audited["logical_job_id"] = job["logical_job_id"]
        audited["physical_recovery_job_id"] = job["id"]
        if job["logical_job_id"] in rows_by_logical:
            raise CampaignAuditError(f"duplicate logical result row: {job['logical_job_id']}")
        rows_by_logical[job["logical_job_id"]] = audited
    # Final whole-input barriers: all source queues/acceptances, both recovery records,
    # the seven prior root acceptance documents, and both interruption receipts.
    for queue in plan["original_queues"]:
        if sha256(Path(queue["path"])) != queue["sha256"] or sha256(Path(queue["acceptance"]["path"])) != queue["acceptance"]["sha256"]:
            raise CampaignAuditError(f"canonical source queue/acceptance changed: {queue['key']}")
    if sha256(Path(plan["recovery_queue_path"])) != plan["recovery_queue_sha256"]:
        raise CampaignAuditError("host recovery queue changed during audit")
    if sha256(Path(plan["recovery_acceptance_path"])) != plan["recovery_acceptance_sha256"]:
        raise CampaignAuditError("host recovery root acceptance changed during audit")
    if sha256(Path(plan["prior_acceptance"]["path"])) != plan["prior_acceptance"]["sha256"]:
        raise CampaignAuditError("prior recovery root acceptance changed during audit")
    for record in plan["prior_numeric_acceptances"]:
        if sha256(Path(record["path"])) != record["sha256"]:
            raise CampaignAuditError(f"prior numeric root acceptance changed: {record['path']}")
    for attempt in plan["superseded_stopped_attempts"]:
        for path_key, sha_key in (("receipt_path", "receipt_sha256"),
                                  ("raw_receipt_path", "raw_receipt_sha256"),
                                  ("reconciliation_path", "reconciliation_sha256")):
            if path_key in attempt and sha256(Path(attempt[path_key])) != attempt[sha_key]:
                raise CampaignAuditError(f"stopped attempt evidence changed: {attempt[path_key]}")
    # Revalidate every nested root-admission dependency, including the exact
    # historical ledger exception, rather than only the enclosing JSON bytes.
    final_plan = preflight_recovery_campaign(root)
    if final_plan != plan:
        raise CampaignAuditError("whole recovery metadata admission changed during audit")
    if sha256(Path(__file__).resolve()) != source_sha_before or sha256(NUMERIC_AUDITOR) != auditor_sha_before:
        raise CampaignAuditError("audit source changed during campaign")
    rows = order_logical_rows(plan["original_jobs"], rows_by_logical)
    counts = {s: sum(r["status"] == s for r in rows) for s in
              ("root_accepted_reused", "reused", "audited", "pending", "failed")}
    if len(rows) != 80 or counts["root_accepted_reused"] != 12:
        raise CampaignAuditError(f"logical campaign projection mismatch: rows={len(rows)} counts={counts}")
    payload = {"schema_version": 1, "scope": "completed_p3_development_validation_audit_via_host_recovery_v2",
        "status": "completed_with_pending_or_failed_rows" if any(r["status"] in ("pending", "failed") for r in rows) else "all_logical_rows_root_accepted_audited_or_reused",
        "total_logical_jobs": 80, "counts": counts, "source_sha256": source_sha_before,
        "numeric_auditor_sha256": auditor_sha_before, "recovery_queue_path": plan["recovery_queue_path"],
        "recovery_queue_sha256": plan["recovery_queue_sha256"],
        "recovery_acceptance_path": plan["recovery_acceptance_path"],
        "recovery_acceptance_sha256": plan["recovery_acceptance_sha256"],
        "prior_recovery_acceptance": {k: plan["prior_acceptance"][k] for k in ("path", "sha256")},
        "superseded_stopped_attempts": plan["superseded_stopped_attempts"],
        "scored_test_origins": False,
        "source_array_may_materialize_other_partitions": any(r.get("source_array_may_materialize_other_partitions") for r in rows),
        "combined_archive_disclosure": "Validation origins only are scored. Chronological combined NPZ array members may be materialized by the unchanged numeric-v3 auditor; this is not a claim that test bytes were untouched.",
        "rows": rows,
        "interpretation": "80 logical rows: 12 previously root-accepted rows referenced once; 68 pending rows resolved through the host-recovery v2 mapping. Both superseded stopped attempts remain separate and are not counted as additional logical rows. No root acceptance of this aggregate, no five-seed result, no test evaluation, and no scientific claim."}
    if output is not None:
        dest = contained(root, output)
        if dest.exists(): raise CampaignAuditError(f"refusing to overwrite {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        with dest.open("xb") as f: f.write(data)
        if sha256(dest) != hashlib.sha256(data).hexdigest(): raise CampaignAuditError("aggregate output failed post-write hash check")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="exclusive RUN-relative JSON result path")
    args = parser.parse_args()
    report = run_recovered_campaign(ROOT, Path(args.output))
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output": str(contained(ROOT, args.output))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
