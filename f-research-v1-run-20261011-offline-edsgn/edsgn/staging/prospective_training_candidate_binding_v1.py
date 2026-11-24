"""Strict metadata-only prospective candidate and launch-binding checks.

This module does not train, inspect validation arrays, or select a scientific
winner. It validates immutable P3A candidate metadata and links a later
root-accepted validation-only selection receipt to an actual at-launch binding.
"""
from __future__ import annotations

import hashlib
import json
import math
import datetime as dt
from pathlib import Path
from typing import Any

try:
    import bind_training_inputs as _binding
except ImportError:  # imported as a staging package from repository root
    from edsgn.scripts import bind_training_inputs as _binding


class CandidateBindingError(ValueError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def no_dups(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise CandidateBindingError(f"duplicate JSON key {k!r}: {path}")
            out[k] = v
        return out
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=no_dups,
                         parse_constant=lambda x: (_ for _ in ()).throw(CandidateBindingError(f"invalid JSON constant {x}")))
    except (OSError, json.JSONDecodeError) as e:
        raise CandidateBindingError(f"cannot read JSON {path}: {e}") from e
    if not isinstance(doc, dict):
        raise CandidateBindingError(f"expected JSON object: {path}")
    return doc


def _resolve(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CandidateBindingError("path must be a nonempty string")
    p = Path(value)
    p = p.resolve() if p.is_absolute() else (root / p).resolve()
    if not p.is_relative_to(root.resolve()):
        raise CandidateBindingError(f"path escapes RUN: {value}")
    return p


def _ref(root: Path, obj: Any, label: str) -> tuple[Path, str]:
    if not isinstance(obj, dict) or set(obj) != {"path", "sha256"}:
        raise CandidateBindingError(f"{label} must have exactly path and sha256")
    p = _resolve(root, obj["path"])
    digest = obj["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise CandidateBindingError(f"{label} sha256 is invalid")
    if not p.is_file() or _sha(p) != digest:
        raise CandidateBindingError(f"{label} is missing or hash-mismatched: {obj['path']}")
    return p, digest


def _strict_int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CandidateBindingError(f"{label} must be an integer >= {minimum}")
    return value


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _entry_map(rows: Any, label: str) -> dict[str, str]:
    if not isinstance(rows, list):
        raise CandidateBindingError(f"{label} must be a list")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise CandidateBindingError(f"malformed {label} entry")
        path, digest = row["path"], row["sha256"]
        if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
            raise CandidateBindingError(f"malformed {label} path/hash")
        if path in result:
            raise CandidateBindingError(f"duplicate {label} path: {path}")
        result[path] = digest
    return result


def _blueprint_expected(root: Path, config_path: Path, *, comparison_id: str,
                        candidate_id: str, seed_id: int, dataset_id: str,
                        expected_variant: str) -> tuple[dict, dict[str, str]]:
    """Derive static identity via unchanged binding file-set helpers, no NPZ decode."""
    old_root = _binding.ROOT
    _binding.ROOT = root.resolve()
    try:
        config, source, config_sha = _binding._load_config(config_path)
        if source is None or not source.is_file():
            raise CandidateBindingError("candidate config must be an existing file")
        if not isinstance(config.get("dataset_path"), str) or not isinstance(config.get("variant"), str):
            raise CandidateBindingError("config requires dataset_path and variant")
        if type(config.get("seed")) is not int or config["seed"] != seed_id:
            raise CandidateBindingError("candidate config seed differs from enclosing seed_id")
        if config.get("variant") != expected_variant:
            raise CandidateBindingError("candidate config variant differs from frozen candidate pool")
        dataset_root, data_paths = _binding._dataset_source_files(config["dataset_path"])
        data_paths = sorted(set(data_paths + _binding._fixed_prior_paths(config, check_hashes=True)))
        code_paths = _binding._code_paths(config["variant"])
        data_entries = _binding._entries(data_paths)
        code_entries = _binding._entries(code_paths)
        config_entry = {"path": str(source), "sha256": config_sha,
                        "canonical_config_sha256": _binding._canonical_hash(config), "config": config}
        doc = {
            "schema_version": 1,
            "scope": "prospective-input-blueprint",
            "status": "prospective-input-blueprint",
            "comparison_id": comparison_id,
            "candidate_id": candidate_id,
            "dataset_id": dataset_id,
            "variant": expected_variant,
            "seed_id": seed_id,
            "dataset_root": str(dataset_root),
            "dataset_sources": data_entries,
            "config_source": config_entry,
            "variant": str(config["variant"]),
            "code_sources": code_entries,
            "hash_policy": "SHA-256 of all .npz/.json files under dataset directory; filename labels do not exclude files; hashes do not load array contents",
        }
        inputs = {str(source): config_sha}
        inputs.update({r["path"]: r["sha256"] for r in data_entries})
        inputs.update({r["path"]: r["sha256"] for r in code_entries})
        return doc, inputs
    finally:
        _binding.ROOT = old_root


def write_input_blueprint(root: str | Path, config_ref: dict, *, comparison_id: str,
                          candidate_id: str, seed_id: int, dataset_id: str,
                          output_path: str | Path) -> dict:
    """Create exclusive static blueprint. Intended for prospective preparation only."""
    root = Path(root).resolve()
    cfg, _ = _ref(root, config_ref, "config")
    if not comparison_id or not candidate_id or dataset_id not in {"synthetic", "email", "mooc"}:
        raise CandidateBindingError("comparison_id/candidate_id and supported dataset_id are required")
    seed_id = _strict_int(seed_id, "seed_id")
    cfgdoc = _json(cfg)
    variant = cfgdoc.get("variant")
    if variant not in {"V2", "V3", "V4", "V9"} or type(cfgdoc.get("seed")) is not int or cfgdoc["seed"] != seed_id:
        raise CandidateBindingError("blueprint requires a V2/V3/V4/V9 config with matching seed")
    output = _resolve(root, str(output_path))
    doc, _ = _blueprint_expected(root, cfg, comparison_id=comparison_id,
                                 candidate_id=candidate_id, seed_id=seed_id,
                                 dataset_id=dataset_id, expected_variant=variant)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return {"path": _rel(root, output), "sha256": _sha(output)}


def _parse_pool(root: Path, prep: dict, *, allow_postrun_outputs: bool = False) -> tuple[dict, Path, str, dict[str, str]]:
    if not isinstance(prep, dict) or set(prep) != {"schema_version", "candidate_pool", "comparison_id"}:
        raise CandidateBindingError("schema4 preparation must have exactly schema_version/candidate_pool/comparison_id")
    if prep["schema_version"] != 4 or not isinstance(prep["comparison_id"], str) or not prep["comparison_id"]:
        raise CandidateBindingError("invalid schema4 preparation identity")
    pool_path, pool_sha = _ref(root, prep["candidate_pool"], "candidate_pool")
    pool = _json(pool_path)
    required = {"schema_version", "scope", "comparison_id", "dataset_id", "variant", "selector", "candidates"}
    if set(pool) != required or pool.get("schema_version") != 1 or pool.get("scope") != "prospective_training_candidate_pool":
        raise CandidateBindingError("candidate pool schema/scope mismatch")
    if pool.get("comparison_id") != prep["comparison_id"]:
        raise CandidateBindingError("candidate pool comparison_id mismatch")
    if pool.get("dataset_id") not in {"synthetic", "email", "mooc"} or pool.get("variant") not in {"V2", "V3", "V4", "V9"}:
        raise CandidateBindingError("candidate pool must declare a supported dataset_id and variant")
    selector = pool["selector"]
    if not isinstance(selector, dict) or set(selector) != {"rule_id", "source"} or selector["rule_id"] != "normalized-syn-mooc-v4-seed0-v1":
        raise CandidateBindingError("selector must use frozen normalized-syn-mooc-v4-seed0-v1 rule")
    selector_path, selector_sha = _ref(root, selector["source"], "selector source")
    current_selector = Path(__file__).resolve()
    if selector_sha != _sha(current_selector):
        raise CandidateBindingError("selector source is not the exact current helper implementation")
    candidates = pool["candidates"]
    if not isinstance(candidates, list) or not (1 <= len(candidates) <= 12):
        raise CandidateBindingError("candidate pool must contain 1..12 ordered choices")
    ids, pairs = [], []
    consumed = {_rel(root, pool_path): pool_sha, _rel(root, selector_path): selector_sha}
    for ci, c in enumerate(candidates):
        if not isinstance(c, dict) or set(c) != {"candidate_id", "graph_candidate_id", "memory_candidate_id", "seeds"}:
            raise CandidateBindingError(f"candidate {ci} has invalid schema")
        cid, gid, mid = c["candidate_id"], c["graph_candidate_id"], c["memory_candidate_id"]
        if any(not isinstance(x, str) or not x for x in (cid, gid, mid)):
            raise CandidateBindingError("candidate IDs must be nonempty strings")
        ids.append(cid); pairs.append((gid, mid))
        seeds = c["seeds"]
        if not isinstance(seeds, list) or [x.get("seed_id") if isinstance(x, dict) else None for x in seeds] != [0, 1, 2, 3, 4]:
            raise CandidateBindingError(f"candidate {cid} must declare ordered seeds 0..4")
        for s in seeds:
            if set(s) != {"seed_id", "config", "input_blueprint", "training_binding_path", "selection_receipt_path"}:
                raise CandidateBindingError(f"candidate {cid} seed row schema mismatch")
            for field in ("config", "input_blueprint"):
                p, h = _ref(root, s[field], f"{cid} seed {s['seed_id']} {field}")
                consumed[_rel(root, p)] = h
            for field in ("training_binding_path", "selection_receipt_path"):
                p = _resolve(root, s[field])
                if not allow_postrun_outputs and (p.exists() or p.is_symlink()):
                    raise CandidateBindingError(f"prospective output already exists: {s[field]}")
            cfgdoc = _json(_resolve(root, s["config"]["path"]))
            if type(cfgdoc.get("seed")) is not int or cfgdoc["seed"] != s["seed_id"]:
                raise CandidateBindingError("candidate config seed differs from enclosing seed_id")
            if cfgdoc.get("variant") != pool["variant"]:
                raise CandidateBindingError("candidate config variant differs from candidate pool")
            if not isinstance(cfgdoc.get("output_dir"), str) or not cfgdoc["output_dir"]:
                raise CandidateBindingError("candidate config must declare output_dir")
            old_root = _binding.ROOT
            _binding.ROOT = root
            try:
                expected_bind = _binding._resolve_path(cfgdoc["output_dir"]) / _binding.BINDING_NAME
            finally:
                _binding.ROOT = old_root
            if _resolve(root, s["training_binding_path"]) != expected_bind.resolve():
                raise CandidateBindingError("training_binding_path must be config.output_dir/input-binding-at-launch.json")
            data_path = str(_binding._resolve_path(cfgdoc["dataset_path"]))
            prior_data_path = c.setdefault("_dataset_path_for_validation", data_path)
            if prior_data_path != data_path:
                raise CandidateBindingError("candidate config dataset_path must be identical across all five seeds")
        c.pop("_dataset_path_for_validation", None)
    if len(set(ids)) != len(ids) or len(set(pairs)) != len(pairs):
        raise CandidateBindingError("candidate IDs and joint graph/memory pairs must be unique")
    return pool, pool_path, pool_sha, consumed


def _validate_preparation(root: Path, prep: dict, seed: int, *, allow_postrun_outputs: bool) -> dict[str, str]:
    root = Path(root).resolve(); seed = _strict_int(seed, "seed_id")
    if seed not in range(5):
        raise CandidateBindingError("seed_id must be 0..4")
    pool, _, _, consumed = _parse_pool(root, prep, allow_postrun_outputs=allow_postrun_outputs)
    for c in pool["candidates"]:
        row = c["seeds"][seed]
        cfg, _ = _ref(root, row["config"], "config")
        expected, refs = _blueprint_expected(root, cfg, comparison_id=pool["comparison_id"],
                                             candidate_id=c["candidate_id"], seed_id=seed,
                                             dataset_id=pool["dataset_id"], expected_variant=pool["variant"])
        bp_path, _ = _ref(root, row["input_blueprint"], "input blueprint")
        actual = _json(bp_path)
        if actual != expected:
            raise CandidateBindingError(f"input blueprint content differs from current static inputs: {row['input_blueprint']['path']}")
        for path, digest in refs.items():
            rel = Path(path).resolve().relative_to(root).as_posix()
            consumed[rel] = digest
    return dict(sorted(consumed.items()))


def validate_preparation(root: str | Path, prep: dict, seed: int, *, allow_postrun_outputs: bool = False) -> dict[str, str]:
    """Validate P3A inputs; post-P4 callers must pass allow_postrun_outputs=True.

    This flag only relaxes exclusive-output absence checks. It does not waive any
    source/hash/config checks or establish phase authorization; callers must
    independently verify the official phase gate before setting it.
    """
    if type(allow_postrun_outputs) is not bool:
        raise CandidateBindingError("allow_postrun_outputs must be bool")
    return _validate_preparation(Path(root).resolve(), prep, seed, allow_postrun_outputs=allow_postrun_outputs)


_CANDIDATE_EVIDENCE_ROLES = {"pilot_config", "input_binding", "phase_receipt", "validation_archive",
                             "independent_numeric_review", "root_acceptance"}
_RANK_HORIZONS = {"synthetic": [1, 3, 7], "mooc": [1, 6, 24]}


def _consume_evidence(root: Path, evidence: Any, roles: set[str], label: str) -> dict[str, str]:
    if not isinstance(evidence, dict) or set(evidence) != roles:
        raise CandidateBindingError(f"{label} evidence roles must be exactly {sorted(roles)}")
    out = {}
    for role in sorted(roles):
        p, digest = _ref(root, evidence[role], f"{label} {role}")
        out[_rel(root, p)] = digest
    return out


def _positive_finite(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise CandidateBindingError(f"{label} must be finite")
    if value < 0 or (value == 0 and not allow_zero):
        raise CandidateBindingError(f"{label} must be {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def _selection_doc(root: Path, ref: dict, pool: dict, pool_path: Path, pool_sha: str, seed: int) -> tuple[dict, Path, str]:
    receipt_path, receipt_sha = _ref(root, ref, "candidate selection receipt")
    receipt = _json(receipt_path)
    required = {"schema_version", "scope", "status", "accepted_by", "test_feedback_used",
                "comparison_id", "pool", "selector", "candidates", "selected_candidate_id", "gru_reference"}
    if set(receipt) != required or receipt.get("schema_version") != 1:
        raise CandidateBindingError("selection receipt schema mismatch")
    if (receipt["scope"] != "root_validation_only_candidate_selection" or receipt["status"] != "accepted"
            or receipt["accepted_by"] != "root" or receipt["test_feedback_used"] is not False):
        raise CandidateBindingError("selection receipt is not a root-accepted train/validation-only record")
    if receipt["comparison_id"] != pool["comparison_id"]:
        raise CandidateBindingError("selection receipt comparison_id mismatch")
    if receipt["pool"] != {"path": _rel(root, pool_path), "sha256": pool_sha}:
        raise CandidateBindingError("selection receipt does not bind exact pool bytes")
    if receipt["selector"] != pool["selector"]:
        raise CandidateBindingError("selection receipt selector differs from pool")
    rows = receipt["candidates"]
    ids = [r.get("candidate_id") if isinstance(r, dict) else None for r in rows] if isinstance(rows, list) else []
    expected_ids = [c["candidate_id"] for c in pool["candidates"]]
    if ids != expected_ids:
        raise CandidateBindingError("selection receipt must retain every candidate in pool order")
    selected = receipt["selected_candidate_id"]
    if selected not in expected_ids:
        raise CandidateBindingError("selected candidate is not in immutable pool")
    selected_row = rows[expected_ids.index(selected)]
    if selected_row.get("status") != "completed":
        raise CandidateBindingError("selected candidate lacks completed status")
    for candidate, result in zip(pool["candidates"], rows):
        if not isinstance(result, dict):
            raise CandidateBindingError("candidate selection ledger row schema mismatch")
        if result["status"] not in {"completed", "failed", "unsupported"}:
            raise CandidateBindingError("candidate selection ledger has unsupported status")
        if result["status"] == "completed":
            if set(result) != {"candidate_id", "status", "validation", "email_descriptive_status"}:
                raise CandidateBindingError("completed selection ledger row schema mismatch")
            v = result["validation"]
            if not isinstance(v, dict) or set(v) != {"synthetic", "mooc"}:
                raise CandidateBindingError("completed candidate must contain both rankable regimes")
            for regime in ("synthetic", "mooc"):
                item = v[regime]
                if not isinstance(item, dict) or set(item) != {"variant", "model_seed", "split", "required_horizons",
                                                                "required_horizon_mae", "wall_seconds", "evidence"}:
                    raise CandidateBindingError("completed validation row schema mismatch")
                if item["variant"] != "V4" or type(item["model_seed"]) is not int or item["model_seed"] != 0 or item["split"] != "validation":
                    raise CandidateBindingError("rank evidence must be V4 pilot seed0 validation only")
                horizons = item["required_horizons"]
                if horizons != _RANK_HORIZONS[regime]:
                    raise CandidateBindingError(f"{regime} required_horizons must be exactly {_RANK_HORIZONS[regime]}")
                item["required_horizon_mae"] = _positive_finite(item["required_horizon_mae"], "required_horizon_mae", allow_zero=True)
                item["wall_seconds"] = _positive_finite(item["wall_seconds"], "validation wall_seconds", allow_zero=True)
                consumed_evidence = _consume_evidence(root, item["evidence"], _CANDIDATE_EVIDENCE_ROLES,
                                                      f"{candidate['candidate_id']} {regime}")
                result.setdefault("_consumed_evidence", {}).update(consumed_evidence)
            if not isinstance(result["email_descriptive_status"], str) or not result["email_descriptive_status"]:
                raise CandidateBindingError("email descriptive status must remain explicit")
        else:
            if set(result) != {"candidate_id", "status", "validation", "email_descriptive_status", "reason", "evidence"} or result["validation"] is not None:
                raise CandidateBindingError("failed/unsupported candidate must keep explicit null validation, reason and evidence")
            if not isinstance(result["reason"], str) or not result["reason"]:
                raise CandidateBindingError("failed/unsupported candidate requires a reason")
            if not isinstance(result["email_descriptive_status"], str) or not result["email_descriptive_status"]:
                raise CandidateBindingError("email descriptive status must remain explicit")
            if not isinstance(result["evidence"], list) or not result["evidence"]:
                raise CandidateBindingError("failed/unsupported candidate requires evidence refs")
            for i, evidence in enumerate(result["evidence"]):
                p, digest = _ref(root, evidence, f"{candidate['candidate_id']} failure evidence {i}")
                result.setdefault("_consumed_evidence", {})[_rel(root, p)] = digest
    refs = receipt["gru_reference"]
    if not isinstance(refs, dict) or set(refs) != {"synthetic", "mooc"}:
        raise CandidateBindingError("GRU references must contain exactly synthetic and mooc")
    for regime, item in refs.items():
        if not isinstance(item, dict) or set(item) != {"mae", "required_horizons", "evidence"}:
            raise CandidateBindingError("GRU reference must include MAE, domain horizons and numeric/root-accepted evidence")
        if item["required_horizons"] != _RANK_HORIZONS[regime]:
            raise CandidateBindingError(f"GRU {regime} horizons do not match the frozen domain horizons")
        item["mae"] = _positive_finite(item["mae"], f"GRU {regime} MAE")
        receipt.setdefault("_consumed_evidence", {}).update(_consume_evidence(
            root, item["evidence"], {"numeric_review", "root_acceptance"}, f"GRU {regime}"))
    rank_rows = []
    for result in rows:
        if result["status"] == "completed":
            rank_rows.append({"candidate_id": result["candidate_id"], "status": "completed",
                              "validation": {r: {"required_horizon_mae": result["validation"][r]["required_horizon_mae"],
                                                 "wall_seconds": result["validation"][r]["wall_seconds"]}
                                             for r in ("synthetic", "mooc")}})
        else:
            rank_rows.append({"candidate_id": result["candidate_id"], "status": result["status"], "validation": None})
    ranking = select_validation_candidate(pool, rank_rows, {r: refs[r]["mae"] for r in refs})
    if ranking["selected_candidate_id"] != selected:
        raise CandidateBindingError("receipt selection does not match frozen deterministic validation rule")
    seed_spec = next(x for x in pool["candidates"] if x["candidate_id"] == selected)["seeds"][seed]
    if _resolve(root, seed_spec["selection_receipt_path"]) != receipt_path:
        raise CandidateBindingError("selection receipt path differs from prospective pool path")
    return receipt, receipt_path, receipt_sha


def validate_selected_training(root: str | Path, prep: dict, seed: int, row: dict) -> dict[str, str]:
    """Verify root selection and actual launch binding for one selected seed row.

    `row` is intentionally an exact projection of run metadata:
    candidate_selection/config/training_binding refs only.
    """
    root = Path(root).resolve(); seed = _strict_int(seed, "seed_id")
    if seed not in range(5):
        raise CandidateBindingError("seed_id must be 0..4")
    if not isinstance(row, dict) or set(row) != {"candidate_selection", "config", "training_binding"}:
        raise CandidateBindingError("selected run row must have exactly candidate_selection/config/training_binding")
    pool, pool_path, pool_sha, consumed = _parse_pool(root, prep, allow_postrun_outputs=True)
    selection, selection_path, selection_sha = _selection_doc(root, row["candidate_selection"], pool, pool_path, pool_sha, seed)
    selected_id = selection["selected_candidate_id"]
    c = next(x for x in pool["candidates"] if x["candidate_id"] == selected_id)
    spec = c["seeds"][seed]
    config_path, config_sha = _ref(root, row["config"], "selected config")
    if row["config"] != spec["config"]:
        raise CandidateBindingError("selected config ref differs from immutable pool")
    expected_binding_path = _resolve(root, spec["training_binding_path"])
    binding_path, binding_sha = _ref(root, row["training_binding"], "actual training binding")
    if binding_path != expected_binding_path:
        raise CandidateBindingError("actual training binding path differs from prospective expected path")
    blueprint_path, _ = _ref(root, spec["input_blueprint"], "input blueprint")
    blueprint = _json(blueprint_path)
    binddoc = _json(binding_path)
    try:
        verified = _binding.verify_binding(binding_path)
    except Exception as e:
        raise CandidateBindingError(f"actual at-launch binding does not verify: {e}") from e
    if not isinstance(verified, dict) or verified.get("status") != "verified":
        raise CandidateBindingError("actual at-launch binding did not return verified status")
    actual_projection = {k: binddoc.get(k) for k in (
        "dataset_root", "dataset_sources", "config_source", "variant", "code_sources", "hash_policy")}
    expected_projection = {k: blueprint.get(k) for k in actual_projection}
    if actual_projection != expected_projection:
        raise CandidateBindingError("actual launch identity differs from P3A static blueprint")
    if (binddoc.get("binding_type") != "at-launch" or binddoc.get("status") != "captured-before-training"
            or not isinstance(binddoc.get("recorded_utc"), str) or not binddoc["recorded_utc"]):
        raise CandidateBindingError("actual training binding lacks valid launch-only fields")
    try:
        recorded = dt.datetime.fromisoformat(binddoc["recorded_utc"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as e:
        raise CandidateBindingError("actual training binding recorded_utc is not valid ISO-8601") from e
    if recorded.tzinfo is None or recorded.utcoffset() is None:
        raise CandidateBindingError("actual training binding recorded_utc must be timezone-aware")
    if binddoc["config_source"].get("path") != str(config_path) or binddoc["config_source"].get("sha256") != config_sha:
        raise CandidateBindingError("actual launch config bytes differ from selected exact config")
    if not isinstance(selection.get("gru_reference"), dict):
        raise CandidateBindingError("selection receipt lacks fixed GRU reference evidence")
    for regime in ("synthetic", "mooc"):
        r = selection["gru_reference"].get(regime, {}).get("mae")
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(float(r)) or float(r) <= 0:
            raise CandidateBindingError(f"selection receipt GRU {regime} reference must be positive finite MAE")
    consumed.update(_validate_preparation(root, prep, seed, allow_postrun_outputs=True))
    consumed[_rel(root, selection_path)] = selection_sha
    consumed[_rel(root, binding_path)] = binding_sha
    consumed.update(selection.get("_consumed_evidence", {}))
    for candidate_row in selection["candidates"]:
        consumed.update(candidate_row.get("_consumed_evidence", {}))
    return dict(sorted(consumed.items()))


def select_validation_candidate(pool: dict, results: list[dict], gru_reference: dict) -> dict:
    """Pure deterministic validation-only ranker; never reads files or arrays.

    This computes a diagnostic selected ID only. It does not issue a root
    acceptance receipt and cannot certify the saved validation evidence.
    """
    if not isinstance(pool, dict) or not isinstance(pool.get("candidates"), list):
        raise CandidateBindingError("pool candidates are required")
    ids = [c.get("candidate_id") for c in pool["candidates"]]
    if not 1 <= len(ids) <= 12 or len(set(ids)) != len(ids):
        raise CandidateBindingError("pool must have 1..12 unique candidates")
    if not isinstance(results, list) or [r.get("candidate_id") if isinstance(r, dict) else None for r in results] != ids:
        raise CandidateBindingError("results must contain every candidate exactly once in pool order")
    if set(gru_reference) != {"synthetic", "mooc"}:
        raise CandidateBindingError("GRU references must contain exactly synthetic and mooc")
    refs = {}
    for regime, value in gru_reference.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)) or value <= 0:
            raise CandidateBindingError(f"invalid positive GRU reference for {regime}")
        refs[regime] = float(value)
    ranked = []
    for row in results:
        if row.get("status") != "completed":
            continue
        val = row.get("validation")
        if not isinstance(val, dict) or set(val) != {"synthetic", "mooc"}:
            continue
        normalized, total_wall = [], 0.0
        valid = True
        for regime in ("synthetic", "mooc"):
            item = val[regime]
            if not isinstance(item, dict) or set(item) != {"required_horizon_mae", "wall_seconds"}:
                valid = False; break
            mae, wall = item["required_horizon_mae"], item["wall_seconds"]
            if (isinstance(mae, bool) or not isinstance(mae, (int, float)) or not math.isfinite(float(mae)) or mae < 0
                    or isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(float(wall)) or wall < 0):
                valid = False; break
            normalized.append(float(mae) / refs[regime]); total_wall += float(wall)
        if valid:
            ranked.append((sum(normalized) / 2.0, total_wall, row["candidate_id"]))
    if not ranked:
        raise CandidateBindingError("no completed candidate has complete finite synthetic/MOOC validation evidence")
    ranked.sort()
    return {"selected_candidate_id": ranked[0][2], "selection_status": "validation_rank_diagnostic_only",
            "singleton_not_optimality_claim": len(ids) == 1,
            "ranked_scores": [{"candidate_id": cid, "normalized_equal_mean": score, "validation_wall_seconds": wall}
                              for score, wall, cid in ranked]}
