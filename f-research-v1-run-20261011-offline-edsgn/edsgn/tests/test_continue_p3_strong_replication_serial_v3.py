import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continue_p3_strong_replication_serial_v3.py"
SPEC = importlib.util.spec_from_file_location("continue_p3_strong_v3_test", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def fixture_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path.resolve())
    counts = [8, 10, 16, 6]
    seed_specs = []
    queue_jobs = {}
    seed_rows = []
    for idx, count in enumerate(counts):
        rel = f"queues/seed0-{idx}.json"
        digest = f"seedsha{idx}"
        acceptance = f"accept/seed0-{idx}.json"
        seed_specs.append((rel, digest, count, acceptance))
        seed_rows.append((rel, digest, count, acceptance))
        queue_jobs[str((tmp_path / rel).resolve())] = [{"id": f"seed0-{idx}-{j}"} for j in range(count)]
    gru_spec = ("queues/gru.json", "grusha", 16)
    queue_jobs[str((tmp_path / gru_spec[0]).resolve())] = [{"id": f"gru-{j}"} for j in range(16)]
    monkeypatch.setattr(module, "SEED0_QUEUE_SPECS", tuple(seed_specs))
    monkeypatch.setattr(module, "GRU_QUEUE_SPEC", gru_spec)
    awaited = [(rel, digest) for rel, digest, _count, _accept in seed_specs]
    return seed_rows, (gru_spec[0], gru_spec[1], 16), awaited, queue_jobs


def test_exact_four_queue_registry_and_56_unique_jobs(tmp_path, monkeypatch):
    seed_rows, gru_row, awaited, queue_jobs = fixture_registry(tmp_path, monkeypatch)
    ids = module.validate_exact_predecessor_registry(seed_rows, gru_row, awaited, queue_jobs)
    assert len(ids) == 56
    assert len(set(ids)) == 56


def test_predecessor_source_pins_match_existing_accepted_coordinators():
    assert all(len(digest) == 64 for digest in module.COORDINATOR_SOURCE_PINS.values())
    module.require_coordinator_source_pins()


def test_duplicate_queue_cannot_substitute_for_another_with_same_total(tmp_path, monkeypatch):
    seed_rows, gru_row, awaited, queue_jobs = fixture_registry(tmp_path, monkeypatch)
    # Retain four descriptors and the same aggregate job count, but repeat queue 2.
    seed_rows[3] = seed_rows[2]
    with pytest.raises(RuntimeError, match="exact four-queue registry"):
        module.validate_exact_predecessor_registry(seed_rows, gru_row, awaited, queue_jobs)


def test_duplicate_job_id_across_distinct_queues_fails_closed(tmp_path, monkeypatch):
    seed_rows, gru_row, awaited, queue_jobs = fixture_registry(tmp_path, monkeypatch)
    path = str((tmp_path / seed_rows[1][0]).resolve())
    queue_jobs[path][0]["id"] = "seed0-0-0"
    with pytest.raises(RuntimeError, match="56 unique job IDs"):
        module.validate_exact_predecessor_registry(seed_rows, gru_row, awaited, queue_jobs)


def test_duplicate_gru_job_id_fails_closed(tmp_path, monkeypatch):
    seed_rows, gru_row, awaited, queue_jobs = fixture_registry(tmp_path, monkeypatch)
    path = str((tmp_path / gru_row[0]).resolve())
    queue_jobs[path][1]["id"] = queue_jobs[path][0]["id"]
    with pytest.raises(RuntimeError, match="invalid or duplicate GRU job IDs"):
        module.validate_exact_predecessor_registry(seed_rows, gru_row, awaited, queue_jobs)
