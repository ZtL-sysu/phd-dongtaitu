"""Gated successor evaluator for frozen C06 split-history robustness replay.

Gated C06 replay v4 with a final post-serialization budget guard. The pre-test freeze can be built from metadata/mask-only
NPZ reads; runtime evaluation rechecks the same bytes and uses the accepted
split-history wrapper and engine prediction path. No training or optimizer
path is present.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import torch

from edsgn import engine, robustness
from edsgn.prepare import apply_normalizer
from edsgn.common import csv_rows, device_for, parameter_hash, serialize_counters, sha256
from edsgn.scheduling import Policy
import split_history_robustness_v2 as split_history
import c06_domain_manifest_preflight_v3 as domain_preflight
import frozen_synthetic_stress as synthetic_stress

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "pipeline-state.json"
DOMAIN_SCHEMA = split_history.DOMAIN_SCHEMA
FREEZE_EXTRA_KEY = "frozen_c06_split_history_robustness_v4"
_RESOURCE_SCRIPTS = (
    "edsgn/scripts/frozen_synthetic_stress.py",
    "edsgn/scripts/candidate_gpu_fullpath_probe_v2.py",
    "edsgn/scripts/aggregate_budget.py",
    "edsgn/scripts/baseline_campaign.py",
)
_SOURCE_PATHS = (
    "edsgn/src/edsgn/robustness.py",
    "edsgn/src/edsgn/engine.py",
    "edsgn/src/edsgn/data.py",
    "edsgn/src/edsgn/prepare.py",
    "edsgn/staging/split_history_robustness_v2.py",
    "edsgn/staging/c06_domain_manifest_preflight_v3.py",
    "edsgn/staging/frozen_split_history_robustness_v4.py",
    "edsgn/scripts/frozen_synthetic_stress.py",
    *_RESOURCE_SCRIPTS,
)


def contained(path, root=ROOT) -> Path:
    p = Path(path)
    p = (Path(root) / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError("C06 artifacts and inputs must remain within the experiment root")
    return p


def _canonical_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         default=lambda x: np.asarray(x).tolist(), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_hashes(root=ROOT) -> dict[str, str]:
    result = {}
    for rel in _SOURCE_PATHS:
        p = contained(Path(root) / rel, root)
        if not p.is_file():
            raise FileNotFoundError(f"Required frozen evaluator dependency missing: {p}")
        result[rel] = sha256(p)
    return result


def _pipeline_statuses(state_path=STATE_PATH) -> dict[str, str]:
    state = _strict_pipeline_json(Path(state_path).read_bytes())
    stages = state.get("stages")
    if not isinstance(stages, list):
        raise RuntimeError("Pipeline state lacks the declared stages list")
    ids = [str(row["id"]) for row in stages if isinstance(row, dict) and "id" in row]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Pipeline state contains duplicate stage identifiers")
    return {str(row["id"]): row for row in stages if isinstance(row, dict) and "id" in row}


def _strict_pipeline_json(raw_bytes):
    def reject_constant(value):
        raise ValueError(f"Nonstandard pipeline JSON constant: {value}")
    def reject_duplicates(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate pipeline JSON key: {key}")
            output[key] = value
        return output
    try:
        return json.loads(raw_bytes.decode("utf-8-sig"), parse_constant=reject_constant,
                          object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("Canonical pipeline-state.json is not strict unambiguous UTF-8 JSON") from exc


def require_phase_gate(split: str, state_path=STATE_PATH) -> dict:
    key = str(split).lower()
    if key == "validation":
        key = "val"
    if key not in {"val", "test"}:
        raise ValueError("Only complete validation or test split replay is supported")
    state_path = Path(state_path).resolve()
    canonical = (ROOT / "pipeline-state.json").resolve()
    if state_path != canonical:
        raise RuntimeError("C06 requires the canonical run-root pipeline-state.json")
    state_bytes = state_path.read_bytes()
    raw = _strict_pipeline_json(state_bytes)
    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list):
        raise RuntimeError("Pipeline state lacks the declared stages list")
    ids = [str(row["id"]) for row in stages_raw if isinstance(row, dict) and "id" in row]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Pipeline state contains duplicate stage identifiers")
    rows = {str(row["id"]): row for row in stages_raw if isinstance(row, dict) and "id" in row}
    for name in ("P2", "P3", "P3A", "P4"):
        if name not in rows:
            raise RuntimeError(f"Pipeline state lacks required stage {name}")
    for name in ("P2", "P3", "P3A"):
        row = rows[name]
        if row.get("status") != "passed" or row.get("gate_verdict") != "pass":
            raise RuntimeError(f"C06 requires {name} status=passed and gate_verdict=pass")
        for section in ("inputs", "evidence"):
            entries = row.get(section, [])
            if not isinstance(entries, list):
                raise RuntimeError(f"{name}.{section} must be a list")
            if not entries:
                raise RuntimeError(f"C06 requires nonempty {name}.{section} hash bindings")
            seen_paths = set()
            for item in entries:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    raise RuntimeError(f"{name}.{section} has malformed path/hash row")
                path = contained(ROOT / item["path"], ROOT)
                if str(path).casefold() in seen_paths:
                    raise RuntimeError(f"{name}.{section} contains duplicate path binding: {item['path']}")
                seen_paths.add(str(path).casefold())
                if not path.is_file() or not isinstance(item.get("sha256"), str) or sha256(path) != item["sha256"]:
                    raise RuntimeError(f"{name} input/evidence hash mismatch: {item.get('path')}")
    if rows["P4"].get("status") != "running":
        raise RuntimeError("C06 replay requires P4 running")
    return {"statuses": {name: row.get("status") for name, row in rows.items()},
            "pipeline_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            "pipeline_state_path": str(state_path), "raw": raw}


def domain_manifest_from_metadata(datasets, split: str) -> dict:
    """Build strict ordered split/domain identity without touching targets."""
    if not datasets:
        raise ValueError("Domain manifest requires a nonempty ordered collection")
    placements = []
    for index, data in enumerate(datasets):
        bounds = split_history.split_history_range(data, split)
        placements.append({**bounds, "trajectory_index": index})
    return split_history._expected_collection_domain_manifest(datasets, placements)


def write_domain_manifest(path, datasets, split: str, *, root=ROOT) -> Path:
    """Create the immutable metadata/domain declaration with exclusive create."""
    target = contained(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = domain_manifest_from_metadata(datasets, split)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return target


def _target_dataset_fingerprint(dataset_path) -> dict:
    return engine.dataset_fingerprint(dataset_path)


def build_freeze_extras(*, checkpoint, ckpt, policy, datasets, dataset_path,
                        domain_manifest_path, recipe, split, authorized_recipes=None,
                        root=ROOT) -> dict:
    """Freeze helper/domain/recipe/policy/preprocessing/data bytes before replay.

    The caller can merge this object with validation evidence via
    `engine.freeze_checkpoint(..., extra=...)`. It does not predict or inspect
    outcomes. The checkpoint's core test-freeze separately binds model weights,
    training dataset, normalization statistics, and this exact policy digest.
    """
    checkpoint = contained(checkpoint, root)
    dataset_path = contained(dataset_path, root)
    domain_path = contained(domain_manifest_path, root)
    paths, metadata_datasets, read_audit = domain_preflight.load_metadata_collection(dataset_path, split)
    stored_domain = json.loads(domain_path.read_text(encoding="utf-8"))
    expected_domain = domain_manifest_from_metadata(metadata_datasets, split)
    if stored_domain != expected_domain:
        raise RuntimeError("Frozen domain manifest differs from metadata-only ordered collection")
    # If a caller already loaded PreparedData, compare only its identity/mask
    # domain to the metadata proxy; no target is consulted during freeze build.
    if datasets is not None:
        if domain_manifest_from_metadata(datasets, split) != expected_domain:
            raise RuntimeError("PreparedData collection differs from metadata-only domain")
    wrapper_path = Path(split_history.__file__).resolve()
    canonical = robustness.canonical_recipe(recipe)
    recipes = list(authorized_recipes or [canonical])
    robust_bundle = robustness.robustness_freeze_bundle_extras(recipes)
    if robustness.recipe_digest(canonical) not in robust_bundle["allowed_robustness_recipe_digests"]:
        raise ValueError("Selected recipe is not in the frozen robustness recipe bundle")
    # The metadata-only domain schema is recipe-independent. For time_block,
    # also freeze the exact randomized block placement/rate for each trajectory.
    domain_recipe = robustness.canonical_recipe({"schema_version": 1, "seed": 2027,
        "missing": {"kind": "time_block", "rate": 0.0}, "count_noise_max_abs": 0,
        "event_time_jitter_bins": 0})
    domain_bundle = split_history.split_history_freeze_bundle_extras(
        wrapper_path=wrapper_path, domain_manifest_path=domain_path, datasets=metadata_datasets,
        recipe=domain_recipe, split=split)
    time_block_bundle = None
    if canonical["missing"]["kind"] == "time_block":
        time_block_bundle = split_history.split_history_freeze_bundle_extras(
            wrapper_path=wrapper_path, domain_manifest_path=domain_path, datasets=metadata_datasets,
            recipe=canonical, split=split)
    source_hashes = _source_hashes(root)
    descriptor = domain_preflight.collection_descriptor(paths, metadata_datasets, read_audit, domain_path, split)
    if descriptor["domain_manifest_sha256"] != sha256(domain_path):
        raise RuntimeError("Metadata preflight domain descriptor digest mismatch")
    is_synthetic = all(data.event_kind == "synthetic" for data in metadata_datasets)
    expected = {
        "evaluator_source_sha256": source_hashes["edsgn/staging/frozen_split_history_robustness_v4.py"],
        "source_sha256": source_hashes,
        "domain_manifest_path": str(domain_path),
        "domain_manifest_sha256": sha256(domain_path),
        "ordered_domain_bundle": domain_bundle,
        "time_block_placement_bundle": time_block_bundle,
        "selected_recipe_sha256": robustness.recipe_digest(canonical),
        "policy_sha256": engine.policy_digest(policy),
        "policy": engine.policy_digest(policy),
        "checkpoint_sha256": sha256(checkpoint),
        "target_dataset_path": str(dataset_path),
        "target_dataset_fingerprint": _target_dataset_fingerprint(dataset_path),
        "normalization_statistics_sha256": _canonical_sha(ckpt["statistics"]),
        "model_task_sha256": _canonical_sha(ckpt["task"]),
        "metadata_preflight_descriptor": descriptor,
        "normalization_strategy": ("checkpoint-frozen-synthetic-normalizer" if is_synthetic
                                   else "require-prepared-normalization-equals-checkpoint"),
        "normalization_source_sha256": {
            "edsgn/src/edsgn/data.py": source_hashes["edsgn/src/edsgn/data.py"],
            "edsgn/src/edsgn/prepare.py": source_hashes["edsgn/src/edsgn/prepare.py"],
            "edsgn/scripts/frozen_synthetic_stress.py": source_hashes["edsgn/scripts/frozen_synthetic_stress.py"],
        },
        "selected_split": "val" if split == "validation" else split,
        "full_origins_required": True,
        "batch_size": 1,
        "dtype": "float32",
    }
    return {**robust_bundle, FREEZE_EXTRA_KEY: expected}


def validate_frozen_authorization(*, checkpoint, model, ckpt, policy, datasets,
                                 dataset_path, domain_manifest_path, recipe, split,
                                 freeze=None, root=ROOT) -> dict:
    """Fail closed unless ordinary and C06-specific freezes match exact bytes."""
    freeze = freeze or engine.validate_freeze(checkpoint, model, ckpt, policy)
    if not freeze.get("test_accessed_before_freeze") is False:
        raise RuntimeError("Freeze must declare test_accessed_before_freeze=false")
    authorization = robustness.validate_robustness_bundle_extra(freeze.get("extra"), recipe)
    expected = build_freeze_extras(checkpoint=checkpoint, ckpt=ckpt, policy=policy,
        datasets=datasets, dataset_path=dataset_path, domain_manifest_path=domain_manifest_path,
        recipe=recipe, split=split, authorized_recipes=[recipe],
        root=root)[FREEZE_EXTRA_KEY]
    actual = freeze.get("extra", {}).get(FREEZE_EXTRA_KEY)
    if actual != expected:
        raise RuntimeError("Frozen C06 domain/source/normalization/recipe/policy binding differs")
    if int(expected["batch_size"]) != 1 or expected["dtype"] != "float32":
        raise RuntimeError("C06 evaluation contract requires batch size 1 and float32")
    return {"robustness_authorization": authorization, "c06_authorization": actual}


def _check_prepared_views(clean, wrapped, split, recipe):
    pairs = engine.index_origins(wrapped, split)
    expected = [(i, int(origin)) for i, data in enumerate(wrapped) for origin in data.origins(split)]
    if pairs != expected or not pairs:
        raise RuntimeError("Evaluation origin collection is empty, incomplete, or reordered")
    qc = []
    for index, data in enumerate(wrapped):
        item = data.perturbation_summary
        expected_mask_hash = hashlib.sha256(np.ascontiguousarray(data._input_mask).tobytes()).hexdigest()
        if item.get("trajectory_input_mask_sha256") != expected_mask_hash:
            raise RuntimeError("Perturbed trajectory mask digest disagrees with actual wrapper mask")
        age_checks = 0
        event_checks = 0
        for origin in data.origins(split):
            window = data.window(int(origin))
            start = int(origin) - int(data.history) + 1
            initial = data._obs_age[start - 1] if start > 0 else np.zeros_like(data._obs_age[0])
            expected_age = robustness._compute_age(window["input_mask"], initial)
            if not np.array_equal(window["obs_age"], expected_age):
                raise RuntimeError("Observation age is inconsistent with the perturbed causal mask")
            age_checks += 1
            records = np.asarray(window["events"], dtype=np.int64).reshape((-1, 3))
            if len(records) and (np.any(records[:, 0] < start) or np.any(records[:, 0] > int(origin))):
                raise RuntimeError("Perturbed event records escaped the current causal input window")
            event_checks += len(records)
        clean_data = clean[index]
        for field in ("node_ids", "horizons"):
            if not np.array_equal(getattr(data, field), getattr(clean_data, field)):
                raise RuntimeError(f"Robustness wrapper changed clean {field}")
        row = {"trajectory_index": index, "n_origins": int(len(data.origins(split))),
               "age_windows_checked": age_checks, "causal_event_records_checked": event_checks,
               "source_input_mask_sha256": item.get("split_history_placement", {}).get("source_input_mask_sha256", hashlib.sha256(np.ascontiguousarray(clean_data.input_mask).tobytes()).hexdigest()),
               "perturbed_input_mask_sha256": expected_mask_hash,
               "perturbation_summary": item}
        if recipe["missing"]["kind"] == "time_block":
            placement = item.get("split_history_placement")
            if not isinstance(placement, dict):
                raise RuntimeError("time_block wrapper omitted split-history placement QC")
            if placement["origin_count"] != age_checks:
                raise RuntimeError("time_block QC origin count differs from full replay coverage")
            row["time_block_window_history_missing_fraction_after"] = placement["window_history_missing_fraction_after"]
            row["time_block_window_history_incremental_missing_fraction"] = placement["window_history_incremental_missing_fraction_of_original_observed"]
            row["time_block_union_realized_rate"] = placement["realized_history_union_rate"]
        qc.append(row)
    return pairs, qc


def _validate_clean_targets(clean, pairs, targets, masks):
    for row, (index, origin) in enumerate(pairs):
        raw = engine.get_target(clean[index], int(origin))
        expected_target, expected_mask = raw
        if not np.array_equal(np.asarray(targets[row]), np.asarray(expected_target)):
            raise RuntimeError(f"Robustness wrapper changed clean target at row {row}")
        if not np.array_equal(np.asarray(masks[row], dtype=bool), np.asarray(expected_mask, dtype=bool)):
            raise RuntimeError(f"Robustness wrapper changed clean target mask at row {row}")


def _validate_collection_task(ckpt, datasets, training_data):
    """Match a split collection to the real training task/checkpoint schema."""
    task = ckpt.get("task")
    required = {"n_nodes", "n_features", "history", "hmax", "count_data"}
    if not isinstance(task, dict) or not required.issubset(task):
        raise RuntimeError("Checkpoint task must declare n_nodes/n_features/history/hmax/count_data")
    if not training_data:
        raise RuntimeError("Checkpoint training collection is empty")
    train = training_data[0]
    for index, data in enumerate(datasets):
        actual = {"n_nodes": int(data.values.shape[1]), "n_features": int(data.values.shape[2]),
                  "history": int(data.history), "hmax": int(data.hmax),
                  "count_data": bool(data.count_data)}
        expected = {"n_nodes": int(task["n_nodes"]), "n_features": int(task["n_features"]),
                    "history": int(task["history"]), "hmax": int(task["hmax"]),
                    "count_data": bool(task["count_data"])}
        if actual != expected:
            raise RuntimeError(f"PreparedData task mismatch for trajectory {index}: {actual} != {expected}")
        if (not np.array_equal(data.node_ids, train.node_ids)
                or not np.array_equal(data.node_ids, datasets[0].node_ids)):
            raise RuntimeError(f"PreparedData node ordering differs from checkpoint training task at trajectory {index}")
        if tuple(data.horizons) != tuple(train.horizons) or data.event_kind != train.event_kind:
            raise RuntimeError(f"PreparedData horizons/event kind differ from checkpoint training task at trajectory {index}")
    return {"task": task, "collection_shapes": [list(data.values.shape) for data in datasets],
            "horizons": list(datasets[0].horizons), "event_kind": datasets[0].event_kind,
            "node_order_sha256": hashlib.sha256(np.ascontiguousarray(datasets[0].node_ids).tobytes()).hexdigest()}


def _normalise_collection_from_training(clean, ckpt, training_data):
    """Use the checkpoint's train-only transform or fail on stat disagreement."""
    if not training_data:
        raise RuntimeError("Cannot validate normalization without the checkpoint's training collection")
    reference_nodes = training_data[0].node_ids
    stats = ckpt["statistics"]
    mean = np.asarray(stats["norm_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(stats["norm_std"], dtype=np.float32).reshape(-1)
    residual = np.asarray(stats["residual_scale"], dtype=np.float32)
    normalized = []
    metadata = []
    for index, data in enumerate(clean):
        if data.event_kind == "synthetic":
            # This is the same frozen normalizer used by the accepted stress path;
            # target arrays are copied through unchanged by its shallow wrapper.
            result = synthetic_stress.with_frozen_normalization(data, ckpt, reference_nodes)
            expected = apply_normalizer(data.raw_values, data.input_mask, mean, std, data.count_data)
            if not np.array_equal(result.values, expected):
                raise RuntimeError("Frozen synthetic normalizer output differs from prepare.apply_normalizer")
            mode = "checkpoint-frozen-synthetic-normalizer"
        else:
            data_mean = np.asarray(data.mean, dtype=np.float32).reshape(-1)
            data_std = np.asarray(data.std, dtype=np.float32).reshape(-1)
            data_residual = np.asarray(data.scale, dtype=np.float32)
            if (not np.array_equal(data_mean, mean) or not np.array_equal(data_std, std)
                    or not np.array_equal(data_residual, residual)):
                raise RuntimeError(f"Prepared normalization differs from checkpoint statistics at trajectory {index}")
            result = data
            mode = "prepared-normalization-matches-checkpoint"
        normalized.append(result)
        metadata.append({"trajectory_index": index, "mode": mode,
            "norm_mean_sha256": _canonical_sha(mean.tolist()),
            "norm_std_sha256": _canonical_sha(std.tolist()),
            "residual_scale_sha256": _canonical_sha(residual.tolist()),
            "normalized_values_sha256": hashlib.sha256(np.ascontiguousarray(result.values).tobytes()).hexdigest()})
    return normalized, metadata


def _write_npz_exclusive(path, **arrays):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path, value):
    tmp = Path(path).with_name(Path(path).name + f".{uuid.uuid4().hex}.tmp")
    try:
        _write_json(tmp, value, exclusive=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _acquire_leases(root, identifier):
    locks = [Path(root) / "logs/model-resource.lock",
             Path(root) / "logs/offline-phase-wall/campaign.lock"]
    acquired = []
    try:
        for lock in locks:
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump({"pid": os.getpid(), "job_id": identifier,
                           "phase": "C06-frozen-robustness-evaluation"}, stream)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            acquired.append(lock)
        return acquired
    except BaseException:
        for lock in reversed(acquired):
            lock.unlink(missing_ok=True)
        raise


def _budget_guard(root, output, receipt_path, receipt, started):
    from aggregate_budget import aggregate
    from baseline_campaign import WALL_BUDGET_S
    receipt["wall_seconds"] = time.perf_counter() - started
    _atomic_json(receipt_path, receipt)
    budget = aggregate(root, exclude_active_output=str(output))
    if not budget.get("coverage_complete"):
        raise RuntimeError(f"Cumulative 72-hour budget coverage incomplete: {budget.get('coverage_gaps')}")
    if float(budget.get("spent_seconds", 0.0)) + receipt["wall_seconds"] >= float(WALL_BUDGET_S):
        raise RuntimeError("Cumulative 72-hour stop-loss reached")
    return budget


@torch.no_grad()
def evaluate_frozen(*, checkpoint, dataset_path, domain_manifest_path, recipe,
                    output_dir, split="test", policy=None, device_name="cuda",
                    root=ROOT, state_path=STATE_PATH, process_scan=None,
                    budget_fn=None):
    """Run one full, ordered, frozen B1 replay; no smoke/limit mode exists."""
    root = Path(root).resolve()
    checkpoint = contained(checkpoint, root)
    dataset_path = contained(dataset_path, root)
    domain_path = contained(domain_manifest_path, root)
    output = contained(output_dir, root)
    key = "val" if split == "validation" else str(split).lower()
    if key not in {"val", "test"}:
        raise ValueError("split must be validation/val or test")
    canonical = robustness.canonical_recipe(recipe)
    policy = policy or Policy()
    if not isinstance(policy, Policy):
        raise TypeError("policy must be an edsgn.scheduling.Policy")
    if output.exists():
        raise FileExistsError("C06 output path is immutable and exclusive")
    receipt_path = root / "logs/offline-phase-wall" / f"c06-history-{uuid.uuid4().hex}.json"
    identifier = receipt_path.stem
    started = time.perf_counter()
    receipt = {"scope": "offline-experiment-phase", "phase": "evaluation", "id": identifier,
        "job_id": identifier, "status": "running", "split": key, "output_dir": str(output),
        "wall_seconds": 0.0, "batch_size": 1, "dtype": "float32", "test_accessed": False,
        "optimizer_steps": 0, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    _write_json(receipt_path, receipt, exclusive=True)
    leases = []
    original_error = None
    budget = budget_fn or _budget_guard
    try:
        stages = require_phase_gate(key, state_path)
        receipt["pipeline_stages_before"] = stages
        leases = _acquire_leases(root, identifier)
        if process_scan is None:
            from candidate_gpu_fullpath_probe_v2 import _owned_model_processes
            process_scan = lambda: _owned_model_processes(root)
        active = process_scan()
        if active:
            raise RuntimeError(f"Owned model/resource process already active: {active}")
        budget(root, output, receipt_path, receipt, started)
        input_hashes_before = _source_hashes(root)
        recipe_sha_before = _canonical_sha(canonical)
        checkpoint_sha_before = sha256(checkpoint)
        freeze_path = checkpoint.parent / "test-freeze.json"
        if not freeze_path.is_file():
            raise RuntimeError("C06 requires an immutable checkpoint/test-freeze.json")
        freeze_sha_before = sha256(freeze_path)
        domain_sha_before = sha256(domain_path)
        device = device_for(device_name)
        model, ckpt = engine.load_checkpoint(checkpoint, device)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("C06 frozen replay requires float32 model parameters")
        freeze = engine.validate_freeze(checkpoint, model, ckpt, policy)
        robustness.validate_robustness_bundle_extra(freeze.get("extra"), canonical)
        training_reference = engine.datasets_at(ckpt["config"]["dataset_path"], "train")
        # Validate the external target collection and C06 extras before decoding
        # targets or values from any target-split archive.
        metadata_paths, metadata_clean, metadata_reads = domain_preflight.load_metadata_collection(dataset_path, key)
        if not metadata_clean:
            raise ValueError("Selected split has no ordered metadata bundles")
        metadata_task_qc = _validate_collection_task(ckpt, metadata_clean, training_reference)
        binding = validate_frozen_authorization(checkpoint=checkpoint, model=model, ckpt=ckpt,
            policy=policy, datasets=metadata_clean, dataset_path=dataset_path,
            domain_manifest_path=domain_path, recipe=canonical, split=key,
            freeze=freeze, root=root)
        descriptor = binding["c06_authorization"]["metadata_preflight_descriptor"]
        if descriptor["ordered_collection"] != domain_preflight.collection_descriptor(
                metadata_paths, metadata_clean, metadata_reads, domain_path, key)["ordered_collection"]:
            raise RuntimeError("Pre-test ordered target collection differs from frozen descriptor")
        # Set before the first normal target-split load attempt: even a failed
        # read is an access attempt and must never be reported as untouched.
        receipt["test_accessed"] = key == "test"
        clean_raw = engine.datasets_at(dataset_path, key)
        if not clean_raw:
            raise ValueError("Selected split has no ordered trajectories")
        preflight_match = domain_preflight.verify_after_normal_load(
            descriptor, domain_path, dataset_path, clean_raw, key)
        clean, normalization_qc = _normalise_collection_from_training(clean_raw, ckpt, training_reference)
        task_qc = _validate_collection_task(ckpt, clean, training_reference)
        if canonical["missing"]["kind"] == "related_group" and canonical["missing"]["rate"] > 0:
            wrapper_training_reference = training_reference
        else:
            wrapper_training_reference = None
        target_fingerprint_before = binding["c06_authorization"]["target_dataset_fingerprint"]
        normalization_sha_before = binding["c06_authorization"]["normalization_statistics_sha256"]
        wrappers = [split_history.build_robustness_view(data, canonical, split=key,
                    training_reference=wrapper_training_reference, trajectory_index=i)
                    for i, data in enumerate(clean)]
        pairs, qc = _check_prepared_views(clean, wrappers, key, canonical)
        receipt.update(checkpoint_sha256=checkpoint_sha_before, test_freeze_sha256=freeze_sha_before,
            domain_manifest_sha256=domain_sha_before, wrapper_input_sha256=input_hashes_before,
            input_recipe_sha256=robustness.recipe_digest(canonical), policy_sha256=engine.policy_digest(policy),
            metadata_task_qc=metadata_task_qc, task_qc=task_qc,
            metadata_preflight_match=preflight_match, normalization_qc=normalization_qc,
            n_origins_expected=len(pairs), trajectory_qc=qc, pipeline_gate=stages)
        budget(root, output, receipt_path, receipt, started)
        predictions, targets, masks, counter_rows = [], [], [], []
        before = parameter_hash(model)
        for index, origin in pairs:
            budget(root, output, receipt_path, receipt, started)
            batch = engine.batch_at(wrappers, [(index, int(origin))], device)
            expected_target, expected_mask = engine.get_target(clean[index], int(origin))
            if not np.array_equal(batch["targets"][0].detach().cpu().numpy(), np.asarray(expected_target)):
                raise RuntimeError(f"Clean target changed at origin {(index, int(origin))}")
            if not np.array_equal(batch["target_mask"][0].detach().cpu().numpy(), np.asarray(expected_mask, dtype=bool)):
                raise RuntimeError(f"Clean target mask changed at origin {(index, int(origin))}")
            result = engine.predict(model, batch, policy, trace=False)
            pred = result["predictions"]
            if pred.dtype != torch.float32 or not torch.isfinite(pred).all():
                raise RuntimeError("C06 predictions must be finite float32")
            predictions.append(pred.detach().cpu().numpy())
            targets.append(batch["targets"].detach().cpu().numpy())
            masks.append(batch["target_mask"].detach().cpu().numpy())
            counter_rows.append(serialize_counters(result.get("counters", {})))
            del batch, result
            receipt["origins_completed"] = len(predictions)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("Frozen weights changed during C06 replay")
        result = {"predictions": np.concatenate(predictions), "targets": np.concatenate(targets),
                  "target_mask": np.concatenate(masks), "pairs": pairs,
                  "counter_rows": counter_rows}
        _validate_clean_targets(clean, pairs, result["targets"], result["target_mask"])
        expected_sources = _source_hashes(root)
        if expected_sources != input_hashes_before:
            raise RuntimeError("C06 evaluator/resource/source bytes changed during replay")
        target_fingerprint_after = _target_dataset_fingerprint(dataset_path)
        normalization_sha_after = _canonical_sha(ckpt["statistics"])
        if (sha256(checkpoint) != checkpoint_sha_before or sha256(freeze_path) != freeze_sha_before
                or sha256(domain_path) != domain_sha_before or _canonical_sha(canonical) != recipe_sha_before
                or target_fingerprint_after != target_fingerprint_before
                or normalization_sha_after != normalization_sha_before
                or engine.policy_digest(policy) != binding["c06_authorization"]["policy_sha256"]):
            raise RuntimeError("Checkpoint/freeze/domain/recipe/data fingerprints changed during replay")
        stages_after = require_phase_gate(key, state_path)
        if stages_after != stages:
            raise RuntimeError("Pipeline gate statuses changed during frozen replay")
        engine.validate_freeze(checkpoint, model, ckpt, policy)
        if parameter_hash(model) != before:
            raise RuntimeError("Frozen parameters differ after final freeze validation")
        budget(root, output, receipt_path, receipt, started)
        if not output.parent.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        serialization_start = time.perf_counter()
        _write_npz_exclusive(output / "predictions.npz", predictions=result["predictions"],
            targets=result["targets"], target_mask=result["target_mask"],
            origins=np.asarray(result["pairs"], dtype=np.int64),
            seed_id=np.asarray(int(ckpt["config"].get("seed", 0)), dtype=np.int64))
        rows = engine.metric_rows(result, clean[0].horizons, ckpt["config"]["variant"],
                                  int(ckpt["config"].get("seed", 0)), key,
                                  ckpt.get("train_activity"))
        csv_rows(output / "metrics_by_horizon.csv", rows)
        _write_json(output / "counter-rows.json", result["counter_rows"], exclusive=True)
        manifest = {"schema_version": 1, "scope": "full_protocol", "split": key,
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha_before,
            "test_freeze_sha256": freeze_sha_before, "domain_manifest_sha256": domain_sha_before,
            "input_recipe": canonical, "input_recipe_sha256": robustness.recipe_digest(canonical),
            "policy": engine.policy_digest(policy), "policy_sha256": engine.policy_digest(policy),
            "dataset_fingerprint": binding["c06_authorization"]["target_dataset_fingerprint"],
            "dataset_fingerprint_before": target_fingerprint_before,
            "dataset_fingerprint_after": target_fingerprint_after,
            "normalization_statistics_sha256": binding["c06_authorization"]["normalization_statistics_sha256"],
            "model_task_sha256": binding["c06_authorization"]["model_task_sha256"],
            "normalization_statistics_sha256_after": normalization_sha_after,
            "normalization_qc": normalization_qc,
            "metadata_preflight_descriptor": descriptor,
            "pipeline_state_sha256": stages["pipeline_state_sha256"],
            "source_sha256_before": input_hashes_before, "source_sha256_after": expected_sources,
            "model_parameters_sha256_before": before, "model_parameters_sha256_after": after,
            "weights_unchanged": before == after, "n_origins": len(pairs),
            "checkpoint_task_qc": task_qc,
            "all_origins_complete_and_ordered": True, "targets_and_target_mask": "copied unchanged from clean PreparedData",
            "trajectory_qc": qc, "accuracy_rows": rows, "counter_rows": counter_rows,
            "optimizer_steps": 0, "serialization_seconds": time.perf_counter() - serialization_start,
            "scientific_timing_claim": False}
        _write_json(output / "perturbation-manifest.json", manifest, exclusive=True)
        # Serialization can take long enough for a source/data/pipeline change;
        # only mark complete after a second post-write fence.
        final_stages = require_phase_gate(key, state_path)
        final_sources = _source_hashes(root)
        final_dataset_fingerprint = _target_dataset_fingerprint(dataset_path)
        final_normalization_sha = _canonical_sha(ckpt["statistics"])
        engine.validate_freeze(checkpoint, model, ckpt, policy)
        if (final_stages != stages or final_sources != input_hashes_before
                or final_dataset_fingerprint != target_fingerprint_before
                or final_normalization_sha != normalization_sha_before
                or sha256(checkpoint) != checkpoint_sha_before
                or sha256(freeze_path) != freeze_sha_before
                or sha256(domain_path) != domain_sha_before
                or engine.policy_digest(policy) != binding["c06_authorization"]["policy_sha256"]
                or parameter_hash(model) != before):
            raise RuntimeError("Post-serialization freeze/source/data/pipeline fence failed")
        # The artifact serialization and final integrity fence consume the same
        # cumulative budget as replay. Never mark this phase completed before
        # the live guard has passed after every output byte has been written.
        budget(root, output, receipt_path, receipt, started)
        receipt.update(status="completed", origins_completed=len(pairs),
            outputs={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
            model_parameters_sha256_before=before, model_parameters_sha256_after=after,
            final_source_sha256=final_sources, final_dataset_fingerprint=final_dataset_fingerprint,
            final_normalization_statistics_sha256=final_normalization_sha,
            pipeline_stages_after=final_stages)
        return manifest
    except BaseException as exc:
        original_error = exc
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            _atomic_json(receipt_path, receipt)
        finally:
            for lock in reversed(leases):
                lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--domain-manifest", required=True)
    parser.add_argument("--recipe-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "validation", "test"), required=True)
    parser.add_argument("--policy", choices=("full", "periodic", "random", "adaptive", "off"), default="full")
    parser.add_argument("--policy-config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    if args.batch_size != 1:
        raise ValueError("Frozen C06 replay requires batch_size=1")
    from edsgn.profile import build_policy
    recipe = json.loads(Path(args.recipe_json).read_text(encoding="utf-8-sig"))
    evaluate_frozen(checkpoint=args.checkpoint, dataset_path=args.dataset_path,
        domain_manifest_path=args.domain_manifest, recipe=recipe, output_dir=args.output,
        split=args.split, policy=build_policy(args.policy, args.policy_config), device_name=args.device)


if __name__ == "__main__":
    main()

