"""Fail-closed, metadata-only validator for frozen endpoint plans and input manifests.

This staging validator does not read forecast arrays, inspect endpoint values,
compute inferential quantities, emit endpoint reports, or decide claims. It
returns only a validated path/hash proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRUSTED_PYTHON = Path(r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe")
VALIDATOR = ROOT / "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py"
STAGE_ORDER = ["P0", "P1", "P1A", "P1B", "P2", "P3", "P3A", "P4", "P5", "P5A"] + [f"P{i}" for i in range(6, 22)]
SEEDS = [0, 1, 2, 3, 4]
OUTCOMES = {
    "forecast_mae_superiority": {"claims": {"H1", "H2"}, "artifacts": ("checkpoint", "freeze", "input_binding", "driver_report", "phase_receipt", "dispatch_receipt", "archive")},
    "forecast_mae_noninferiority": {"claims": {"H1", "H3"}, "artifacts": ("checkpoint", "freeze", "input_binding", "driver_report", "phase_receipt", "dispatch_receipt", "archive")},
    "graph_tv_noise_superiority": {"claims": {"H1"}, "artifacts": ("checkpoint", "freeze", "input_binding", "driver_report", "phase_receipt", "dispatch_receipt", "trace")},
    "graph_tv_response_superiority": {"claims": {"H1"}, "artifacts": ("checkpoint", "freeze", "input_binding", "driver_report", "phase_receipt", "dispatch_receipt", "trace")},
    "paired_window_cost_superiority": {"claims": {"H3"}, "artifacts": ("checkpoint", "freeze", "input_binding", "driver_report", "phase_receipt", "dispatch_receipt", "csv")},
}
SLICES = {"full_post_switch", "post4L", "paired_pre_post4L", "full_split"}


class PlanError(ValueError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_no_duplicates(path: Path) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise PlanError(f"duplicate JSON key {key!r} in {path}")
            out[key] = value
        return out
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(PlanError(f"invalid JSON constant {x}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError(f"expected JSON object in {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PlanError("artifact path must be a nonempty string")
    raw = Path(value)
    result = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if not result.is_relative_to(root.resolve()):
        raise PlanError(f"artifact path escapes RUN: {value}")
    return result


def _digest_row(root: Path, row: dict, *, label: str) -> tuple[Path, str]:
    if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
        raise PlanError(f"{label} must contain exactly path and sha256")
    path = _resolve(root, row["path"])
    digest = row["sha256"]
    if not path.is_file() or not isinstance(digest, str) or len(digest) != 64 or _sha(path) != digest:
        raise PlanError(f"{label} missing or SHA-256 mismatch: {row.get('path')}")
    return path, digest


def _bind_map(stage: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for field in ("inputs", "evidence"):
        rows = stage.get(field)
        if not isinstance(rows, list):
            raise PlanError(f"passed stage lacks list {field}")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
                raise PlanError(f"malformed passed-stage {field} entry")
            rel = Path(row["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise PlanError("canonical stage artifact path must be RUN-relative")
            key = rel.as_posix()
            if key in out and out[key] != row["sha256"]:
                raise PlanError(f"conflicting canonical stage binding: {key}")
            out[key] = row["sha256"]
    return out


def canonical_predecessor(root: Path, plan_path: Path, plan_sha: str, registry: dict,
                          registry_path: Path, registry_sha: str, helper_rows: list[dict],
                          acf_row: dict, epsilon_row: dict | None, *, official=True) -> dict:
    root = root.resolve()
    pipeline = root / "pipeline-state.json"
    if not pipeline.is_file():
        raise PlanError("canonical RUN/pipeline-state.json is missing")
    pipeline_bytes = pipeline.read_bytes()
    state = _json_no_duplicates(pipeline)
    stages_list = state.get("stages")
    if not isinstance(stages_list, list):
        raise PlanError("canonical pipeline state lacks stages")
    ids = [row.get("id") if isinstance(row, dict) else None for row in stages_list]
    if ids != STAGE_ORDER:
        raise PlanError("canonical stage list is duplicated, incomplete, or out of order")
    stages = {row["id"]: row for row in stages_list}
    for name in ("P2", "P3", "P3A"):
        row = stages[name]
        if row.get("status") != "passed" or row.get("gate_verdict") != "pass":
            raise PlanError(f"canonical {name} must be passed/pass")
        if row.get("contract_version") != state.get("contract_version"):
            raise PlanError(f"canonical {name} contract version mismatch")
    if stages["P4"].get("status") != "running":
        raise PlanError("canonical P4 must be running")

    contract = state.get("contract")
    if not isinstance(contract, list) or not contract:
        raise PlanError("canonical contract binding is empty")
    contract_paths = set()
    for row in contract:
        path, digest = _digest_row(root, row, label="contract")
        contract_paths.add(path)
    bound: dict[str, str] = {}
    for name in STAGE_ORDER[:STAGE_ORDER.index("P4")]:
        row = stages[name]
        if row.get("status") != "passed" or row.get("gate_verdict") != "pass":
            raise PlanError(f"stage before P4 is not passed/pass: {name}")
        if row.get("contract_version") != state.get("contract_version"):
            raise PlanError(f"stale contract version in {name}")
        for key, digest in _bind_map(row).items():
            path = _resolve(root, key)
            if not path.is_file() or _sha(path) != digest:
                raise PlanError(f"canonical predecessor hash mismatch: {key}")
            if key in bound and bound[key] != digest:
                raise PlanError(f"conflicting predecessor hash binding: {key}")
            bound[key] = digest

    def rel(path: Path) -> str:
        return path.resolve().relative_to(root).as_posix()
    required = {rel(plan_path): plan_sha, rel(registry_path): registry_sha,
                rel(_resolve(root, acf_row["path"])): acf_row["sha256"]}
    if epsilon_row is not None:
        required[rel(_resolve(root, epsilon_row["path"]))] = epsilon_row["sha256"]
    for row in helper_rows:
        required[rel(_resolve(root, row["path"]))] = row["sha256"]
    p3a = _bind_map(stages["P3A"])
    for key, digest in required.items():
        if p3a.get(key) != digest:
            raise PlanError(f"P3A does not hash-bind required endpoint input: {key}")
    if official:
        proc = subprocess.run([str(TRUSTED_PYTHON), str(VALIDATOR), str(pipeline), "--before", "P4"],
                              cwd=str(root), capture_output=True, text=True, shell=False)
        if proc.returncode:
            raise PlanError(f"official --before P4 validator rejected canonical state: {proc.stdout[-1000:]} {proc.stderr[-1000:]}")
    if pipeline.read_bytes() != pipeline_bytes:
        raise PlanError("canonical pipeline state changed during predecessor validation")
    return {"pipeline_path": rel(pipeline), "pipeline_sha256": hashlib.sha256(pipeline_bytes).hexdigest(),
            "contract_paths": sorted(rel(x) for x in contract_paths), "passed_stage_count": 7,
            "p3a_bound_required_inputs": sorted(required)}


def _strict_int(x, label: str, *, minimum=0) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < minimum:
        raise PlanError(f"{label} must be an integer >= {minimum}")
    return x


def _validate_collection(root: Path, collection: dict) -> list[dict]:
    if not isinstance(collection, dict) or set(collection) != {"root", "files"}:
        raise PlanError("dataset_collection requires exactly root and files")
    base = _resolve(root, collection["root"])
    if not base.is_dir():
        raise PlanError("dataset_collection.root must be an existing directory")
    rows = collection["files"]
    if not isinstance(rows, list) or not rows:
        raise PlanError("dataset_collection.files must be a nonempty ordered list")
    paths = []
    seen = set()
    for i, row in enumerate(rows):
        path, digest = _digest_row(root, row, label=f"dataset file {i}")
        if not path.is_relative_to(base) or path in seen:
            raise PlanError("dataset file is outside collection root or duplicated")
        seen.add(path); paths.append((path, digest))
    actual = {p.resolve() for p in base.rglob("*") if p.is_file()}
    if actual != seen:
        raise PlanError(f"dataset collection membership changed (missing={len(actual-seen)}, extra={len(seen-actual)})")
    return [{"path": p.relative_to(root).as_posix(), "sha256": h} for p, h in paths]


def _validate_plan(root: Path, path: Path, plan: dict, registry_path: Path, registry: dict,
                   registry_sha: str) -> tuple[dict, dict[str, str]]:
    required = {"schema_version", "endpoint_id", "claim", "outcome_type", "seed_ids",
                "required_horizons", "history", "max_horizon", "block_length",
                "acf_dependence_length", "acf_evidence", "epsilon_evidence", "slice",
                "domain", "dataset_collection", "producer_sources", "left", "right", "switch_indices"}
    if set(plan) != required or plan.get("schema_version") != 1:
        raise PlanError("endpoint plan keys/schema must match typed endpoint plan v1 exactly")
    if not isinstance(plan["endpoint_id"], str) or not plan["endpoint_id"]:
        raise PlanError("endpoint_id must be nonempty")
    outcome = plan["outcome_type"]
    if outcome not in OUTCOMES or plan["claim"] not in OUTCOMES[outcome]["claims"]:
        raise PlanError("outcome_type and claim are unsupported or inconsistent")
    seeds = plan["seed_ids"]
    if not isinstance(seeds, list) or any(type(s) is not int for s in seeds) or seeds != SEEDS:
        raise PlanError("model seed_ids must be exactly ordered integer seeds 0..4")
    horizons = plan["required_horizons"]
    if (not isinstance(horizons, list) or not horizons or any(type(h) is not int or h < 1 for h in horizons)
            or len(set(horizons)) != len(horizons)):
        raise PlanError("required_horizons must be nonempty unique positive integers")
    history = _strict_int(plan["history"], "history", minimum=1)
    hmax = _strict_int(plan["max_horizon"], "max_horizon", minimum=1)
    if max(horizons) > hmax:
        raise PlanError("required horizon exceeds max_horizon")
    acf_len = _strict_int(plan["acf_dependence_length"], "acf_dependence_length", minimum=1)
    block = _strict_int(plan["block_length"], "block_length", minimum=1)
    if block != max(history, hmax, acf_len):
        raise PlanError("block_length must equal max(history,max_horizon,ACF dependence length)")
    if plan["slice"] not in SLICES:
        raise PlanError("unsupported temporal slice")

    domain = plan["domain"]
    if not isinstance(domain, dict) or set(domain) != {"dataset_id", "split", "path", "sha256", "bounds"}:
        raise PlanError("domain requires dataset_id/split/path/sha256/bounds")
    if domain["split"] != "test" or not isinstance(domain["dataset_id"], str) or not domain["dataset_id"]:
        raise PlanError("endpoint domain must identify a test split and dataset")
    dpath, dsha = _digest_row(root, {"path": domain["path"], "sha256": domain["sha256"]}, label="domain manifest")
    domain_doc = _json_no_duplicates(dpath)
    if domain_doc.get("schema") != "edsgn.split-history-domain.v1":
        raise PlanError("domain manifest must use the accepted strict split-history domain schema")
    bounds = domain["bounds"]
    if not isinstance(bounds, list) or not bounds:
        raise PlanError("domain bounds must list every trajectory")
    tids = []
    for row in bounds:
        if not isinstance(row, dict) or set(row) != {"trajectory_index", "split_start", "split_stop", "first_origin", "last_origin", "origin_count"}:
            raise PlanError("malformed domain trajectory bounds")
        tid = _strict_int(row["trajectory_index"], "trajectory_index")
        start = _strict_int(row["split_start"], "split_start")
        stop = _strict_int(row["split_stop"], "split_stop", minimum=1)
        first = _strict_int(row["first_origin"], "first_origin")
        last = _strict_int(row["last_origin"], "last_origin")
        count = _strict_int(row["origin_count"], "origin_count", minimum=1)
        if stop <= start or first != max(history-1, start-1) or last != stop-hmax-1 or count != last-first+1:
            raise PlanError("domain origins do not match full legal split support for declared history/horizon")
        if block > count:
            raise PlanError("frozen dependence block exceeds a trajectory's legal test-origin support")
        tids.append(tid)
    if tids != list(range(len(tids))):
        raise PlanError("trajectory bounds must enumerate the complete ordered collection")
    trajectories = domain_doc.get("trajectories")
    data_domain = domain_doc.get("dataset")
    if not isinstance(trajectories, list) or len(trajectories) != len(bounds) or not isinstance(data_domain, dict):
        raise PlanError("strict domain manifest trajectory/dataset schema is malformed")
    if data_domain.get("source_name") != domain["dataset_id"]:
        raise PlanError("domain dataset identity differs from endpoint plan")
    if data_domain.get("history_length") != history or data_domain.get("hmax") != hmax:
        raise PlanError("domain manifest history/hmax differs from endpoint plan")
    shape = data_domain.get("shape_tnf")
    if not isinstance(shape, list) or len(shape) != 3 or any(type(x) is not int or x < 1 for x in shape):
        raise PlanError("domain manifest must declare a valid [time,node,feature] shape")
    if data_domain.get("split_bounds_start_inclusive") != bounds[0]["split_start"] or data_domain.get("split_bounds_end_exclusive") != bounds[0]["split_stop"]:
        raise PlanError("domain split bounds differ from plan")
    if bounds[0]["split_stop"] > shape[0]:
        raise PlanError("domain split exceeds declared trajectory length")
    for declared, actual in zip(bounds, trajectories):
        expected = {"trajectory_index": declared["trajectory_index"], "split": "test",
                    "origin_first": declared["first_origin"], "origin_last": declared["last_origin"],
                    "origin_count": declared["origin_count"], "history_length": history,
                    "history_union_start_inclusive": declared["first_origin"]-history+1,
                    "history_union_end_exclusive": declared["last_origin"]+1,
                    "domain_bounds_start_inclusive": declared["first_origin"]-history+1,
                    "domain_bounds_end_exclusive": declared["last_origin"]+1}
        if any(actual.get(k) != v for k, v in expected.items()):
            raise PlanError("domain manifest trajectory bounds differ from endpoint plan")
    if plan["slice"] in {"post4L", "paired_pre_post4L", "full_post_switch"}:
        switch_map = plan.get("switch_indices")
        if not isinstance(switch_map, dict) or set(switch_map) != {str(i) for i in tids}:
            raise PlanError("switch-relative endpoint needs one frozen switch index per trajectory")
        for row in bounds:
            switch = switch_map.get(str(row["trajectory_index"]))
            _strict_int(switch, "switch index")
            if plan["slice"] == "post4L" and switch + 4*history-1 > row["last_origin"]:
                raise PlanError("post4L interval is unsupported; endpoint must be unidentifiable, not truncated")
            if plan["slice"] == "paired_pre_post4L" and switch-4*history < row["first_origin"]:
                raise PlanError("paired pre/post 4L interval is unsupported")

    collection_rows = _validate_collection(root, plan["dataset_collection"])
    if domain_doc.get("ordered_collection"):
        declared = [{"trajectory_index": x.get("trajectory_index"), "path": x.get("path"), "archive_sha256": x.get("archive_sha256")}
                    for x in domain_doc["ordered_collection"]]
        if len(declared) != len(bounds):
            raise PlanError("domain ordered collection trajectory count differs from declared origin bounds")
        if [x["trajectory_index"] for x in declared] != list(range(len(declared))):
            raise PlanError("domain ordered collection indices are incomplete or reordered")
        collection_by_path = {x["path"]: x["sha256"] for x in collection_rows}
        if any(collection_by_path.get(x["path"]) != x["archive_sha256"] for x in declared):
            raise PlanError("domain manifest ordered collection differs from hash-only dataset inventory")
        non_oracle_npz = sorted(x["path"] for x in collection_rows
                                if x["path"].lower().endswith(".npz") and not x["path"].lower().endswith(".oracle.npz"))
        if non_oracle_npz != [x["path"] for x in declared]:
            raise PlanError("domain manifest does not enumerate every forecast collection bundle")

    acf_path, acf_sha = _digest_row(root, plan["acf_evidence"], label="ACF evidence")
    acf_doc = _json_no_duplicates(acf_path)
    acf_declared = acf_doc.get("dependence_length", acf_doc.get("acf_dependence_length", acf_doc.get("block_length_windows")))
    if acf_declared != acf_len:
        raise PlanError("ACF evidence dependence length differs from endpoint plan")
    if acf_doc.get("history_length") != history or acf_doc.get("hmax") != hmax:
        raise PlanError("ACF evidence history/hmax differs from endpoint plan")
    if acf_doc.get("selection_status") not in {"available-for-prospective-use", "accepted-validation-only"}:
        raise PlanError("ACF evidence is unavailable or not accepted for prospective block selection")
    eps = plan["epsilon_evidence"]
    epsilon_ref = None
    if outcome == "forecast_mae_noninferiority":
        if not isinstance(eps, dict) or set(eps) != {"status", "value_raw", "path", "sha256", "scope", "dataset_id", "metric", "horizons"}:
            raise PlanError("noninferiority epsilon evidence has incomplete schema")
        if eps["status"] == "accepted":
            val = eps["value_raw"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise PlanError("accepted raw-unit epsilon must be finite and nonnegative")
            if eps["scope"] != "validation_only" or eps["dataset_id"] != domain["dataset_id"] or eps["metric"] != "raw_mae" or eps["horizons"] != horizons:
                raise PlanError("epsilon domain/metric/horizons do not match this endpoint")
            epsilon_doc = _json_no_duplicates(_resolve(root, eps["path"]))
            if (epsilon_doc.get("status") != "accepted" or epsilon_doc.get("value_raw") != val
                    or epsilon_doc.get("scope") != "validation_only"
                    or epsilon_doc.get("dataset_id") != domain["dataset_id"]
                    or epsilon_doc.get("metric") != "raw_mae"
                    or epsilon_doc.get("horizons") != horizons):
                raise PlanError("epsilon artifact contents do not match validation-bound plan fields")
        elif eps["status"] == "unidentifiable":
            if eps["value_raw"] is not None:
                raise PlanError("unidentifiable epsilon must not invent a numeric value")
            epsilon_doc = _json_no_duplicates(_resolve(root, eps["path"]))
            if (epsilon_doc.get("status") != "unidentifiable" or epsilon_doc.get("scope") != "validation_only"
                    or epsilon_doc.get("dataset_id") != domain["dataset_id"]
                    or epsilon_doc.get("metric") != "raw_mae" or epsilon_doc.get("horizons") != horizons):
                raise PlanError("unidentifiable margin evidence does not match validation-only plan identity")
        else:
            raise PlanError("epsilon status must be accepted or unidentifiable")
        epsilon_ref = {"path": eps["path"], "sha256": eps["sha256"]}
        _digest_row(root, epsilon_ref, label="epsilon evidence")
    elif eps is not None:
        raise PlanError("epsilon is allowed only for forecast noninferiority endpoints")

    sources = plan["producer_sources"]
    if not isinstance(sources, list) or not sources:
        raise PlanError("producer_sources must be nonempty source path/hash rows")
    source_hashes = {}
    for i, row in enumerate(sources):
        p, digest = _digest_row(root, row, label=f"producer source {i}")
        rel = p.relative_to(root).as_posix()
        if rel in source_hashes:
            raise PlanError("duplicate producer source path")
        source_hashes[rel] = digest
    consumed: dict[str, str] = {}
    all_refs = [{"path": str(path.relative_to(root).as_posix()), "sha256": _sha(path)},
                {"path": str(registry_path.relative_to(root).as_posix()), "sha256": registry_sha},
                {"path": domain["path"], "sha256": dsha}, plan["acf_evidence"], *sources,
                *collection_rows]
    if epsilon_ref:
        all_refs.append(epsilon_ref)
    for side in ("left", "right"):
        desc = plan[side]
        if not isinstance(desc, dict) or set(desc) != {"method_id", "producer_source_sha256", "seeds"}:
            raise PlanError(f"{side} descriptor requires method_id/producer_source_sha256/seeds")
        if not isinstance(desc["method_id"], str) or not desc["method_id"] or desc["producer_source_sha256"] not in source_hashes.values():
            raise PlanError(f"{side} method/source descriptor is not bound to declared producer source")
        runs = desc["seeds"]
        if not isinstance(runs, list) or len(runs) != 5 or [r.get("seed_id") if isinstance(r, dict) else None for r in runs] != SEEDS:
            raise PlanError(f"{side} requires five ordered distinct model seed descriptors 0..4")
        required_artifacts = set(OUTCOMES[outcome]["artifacts"])
        for seed in runs:
            if not isinstance(seed, dict) or set(seed) != {"seed_id", "artifacts"}:
                raise PlanError("seed descriptor must contain exactly seed_id and artifacts")
            artifacts = seed["artifacts"]
            if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts:
                raise PlanError(f"{side} seed artifacts do not match outcome schema")
            for role, ref in artifacts.items():
                p, digest = _digest_row(root, ref, label=f"{side} seed {seed['seed_id']} {role}")
                if role in {"archive", "trace", "csv", "driver_report", "phase_receipt", "dispatch_receipt"}:
                    # JSON/CSV/NPZ payloads are bound by bytes only. No arrays are opened here.
                    pass
                all_refs.append(ref)
    for row in all_refs:
        p = _resolve(root, row["path"])
        digest = row["sha256"]
        rel = p.relative_to(root).as_posix()
        if rel in consumed and consumed[rel] != digest:
            raise PlanError(f"same consumed path has conflicting hashes: {rel}")
        consumed[rel] = digest
    # A prepared registry is only a locator/hash-bound registry here; it cannot upgrade scope.
    if registry.get("schema_version") != 1 or registry.get("split") != "test":
        raise PlanError("endpoint registry must be schema v1 for test")
    if registry.get("artifact_status") != "accepted_pretest_registry":
        raise PlanError("endpoint registry must be explicitly root-accepted before test")
    plans = registry.get("endpoint_plans")
    rel_plan = path.relative_to(root).as_posix()
    if not isinstance(plans, list) or len(plans) != 49:
        raise PlanError("accepted endpoint registry must enumerate all 49 primary rows")
    expected_counts = {"H1": 9, "H2": 16, "H3": 24}
    counts = {claim: 0 for claim in expected_counts}
    ids = set()
    current_found = False
    for row in plans:
        if not isinstance(row, dict) or set(row) != {"endpoint_id", "claim", "path", "sha256"}:
            raise PlanError("registry endpoint plan entries require endpoint_id/claim/path/sha256")
        if row["claim"] not in counts or not isinstance(row["endpoint_id"], str) or not row["endpoint_id"] or row["endpoint_id"] in ids:
            raise PlanError("registry has invalid/duplicate endpoint identity or claim")
        ids.add(row["endpoint_id"]); counts[row["claim"]] += 1
        _digest_row(root, {"path": row["path"], "sha256": row["sha256"]}, label="registered endpoint plan")
        if row == {"endpoint_id": plan["endpoint_id"], "claim": plan["claim"],
                   "path": rel_plan, "sha256": _sha(path)}:
            current_found = True
    if counts != expected_counts:
        raise PlanError("accepted endpoint registry family counts must be H1=9, H2=16, H3=24")
    if not current_found:
        raise PlanError("registry does not bind this endpoint plan path/hash/id")
    for row in plans:
        registered = _resolve(root, row["path"])
        rel = registered.relative_to(root).as_posix()
        if rel in consumed and consumed[rel] != row["sha256"]:
            raise PlanError(f"registry plan hash conflicts with another consumed binding: {rel}")
        consumed[rel] = row["sha256"]
    return {"dataset_files": collection_rows, "domain_sha256": dsha, "acf_sha256": acf_sha,
            "epsilon_status": None if outcome != "forecast_mae_noninferiority" else eps["status"],
            "source_sha256": source_hashes}, consumed


def _validate_manifest(root: Path, manifest_path: Path, manifest: dict, plan_path: Path,
                       plan_sha: str, consumed: dict[str, str]) -> dict:
    if set(manifest) != {"schema_version", "plan_path", "plan_sha256", "consumed_files"} or manifest.get("schema_version") != 1:
        raise PlanError("postrun manifest must contain only v1 plan binding and consumed file hashes")
    if manifest["plan_path"] != plan_path.relative_to(root).as_posix() or manifest["plan_sha256"] != plan_sha:
        raise PlanError("postrun manifest plan binding mismatch")
    rows = manifest["consumed_files"]
    if not isinstance(rows, list):
        raise PlanError("postrun consumed_files must be an ordered list")
    actual = {}
    for row in rows:
        p, digest = _digest_row(root, row, label="postrun consumed file")
        rel = p.relative_to(root).as_posix()
        if rel in actual:
            raise PlanError("duplicate path in postrun manifest")
        actual[rel] = digest
    if actual != consumed:
        missing = set(consumed) - set(actual); extra = set(actual) - set(consumed)
        changed = {k for k in set(actual) & set(consumed) if actual[k] != consumed[k]}
        raise PlanError(f"postrun manifest does not equal plan-consumed input set (missing={len(missing)}, extra={len(extra)}, changed={len(changed)})")
    return {"manifest_path": manifest_path.relative_to(root).as_posix(), "manifest_sha256": _sha(manifest_path),
            "consumed_file_count": len(actual), "consumed_paths": sorted(actual)}


def validate(plan_path: Path, registry_path: Path, *, manifest_path: Path | None = None,
             root: Path = ROOT, official=True) -> dict:
    root = root.resolve()
    plan_path = plan_path.resolve(); registry_path = registry_path.resolve()
    if not plan_path.is_relative_to(root) or not registry_path.is_relative_to(root):
        raise PlanError("plan and registry must remain within RUN")
    plan_sha = _sha(plan_path); registry_sha = _sha(registry_path)
    plan = _json_no_duplicates(plan_path); registry = _json_no_duplicates(registry_path)
    proof, consumed = _validate_plan(root, plan_path, plan, registry_path, registry, registry_sha)
    helper_rows = plan["producer_sources"]
    predecessor = canonical_predecessor(root, plan_path, plan_sha, registry, registry_path,
        registry_sha, helper_rows, plan["acf_evidence"], plan["epsilon_evidence"], official=official)
    manifest_proof = None
    if manifest_path is not None:
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_relative_to(root):
            raise PlanError("postrun manifest must remain within RUN")
        manifest = _json_no_duplicates(manifest_path)
        manifest_proof = _validate_manifest(root, manifest_path, manifest, plan_path, plan_sha, consumed)
    # Final TOCTOU barrier for every bound input and canonical predecessor.
    if _sha(plan_path) != plan_sha or _sha(registry_path) != registry_sha:
        raise PlanError("plan or registry changed during validation")
    for rel, digest in consumed.items():
        if _sha(_resolve(root, rel)) != digest:
            raise PlanError(f"consumed input changed during validation: {rel}")
    state_path = root / "pipeline-state.json"
    ledger_sha = _sha(state_path)
    if manifest_path is not None and _sha(manifest_path) != manifest_proof["manifest_sha256"]:
        raise PlanError("postrun manifest changed during validation")
    return {"scope": "typed endpoint path/hash validation only; no arrays or outcome values read",
            "endpoint_id": plan["endpoint_id"], "outcome_type": plan["outcome_type"],
            "epsilon_status": proof["epsilon_status"], "plan_path": plan_path.relative_to(root).as_posix(),
            "plan_sha256": plan_sha, "registry_path": registry_path.relative_to(root).as_posix(),
            "registry_sha256": registry_sha, "canonical_predecessor": predecessor,
            "dataset_file_count": len(proof["dataset_files"]), "dataset_sha256_by_path": proof["dataset_files"],
            "domain_sha256": proof["domain_sha256"], "acf_sha256": proof["acf_sha256"],
            "producer_source_sha256": proof["source_sha256"], "postrun_manifest": manifest_proof,
            "canonical_pipeline_sha256_final": ledger_sha}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--postrun-manifest")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    proof = validate(Path(args.plan), Path(args.registry),
                     manifest_path=Path(args.postrun_manifest) if args.postrun_manifest else None,
                     root=Path(args.root))
    print(json.dumps(proof, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
