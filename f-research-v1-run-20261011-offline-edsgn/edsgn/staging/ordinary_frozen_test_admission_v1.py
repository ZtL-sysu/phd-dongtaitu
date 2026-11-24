"""Metadata-only admission checks for ordinary frozen test jobs.

This module does not load checkpoints, import torch, or decode NPZ arrays. It
binds an already completed training run and a frozen matrix/job to the existing
global test queue; it is not a scientific acceptance or matrix-completeness
claim.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

try:
    import bind_training_inputs as binding
    import prospective_training_candidate_binding_v1 as candidates
    import global_frozen_test_completion_v1 as global_completion
except ImportError:  # package-style imports from repository root
    from edsgn.scripts import bind_training_inputs as binding
    from edsgn.staging import prospective_training_candidate_binding_v1 as candidates
    from edsgn.staging import global_frozen_test_completion_v1 as global_completion


SOURCE_REL = "edsgn/staging/ordinary_frozen_test_admission_v1.py"
GLOBAL_COMPLETION_REL = "edsgn/staging/global_frozen_test_completion_v1.py"
DISPATCHER_REL = "edsgn/scripts/verified_frozen_dispatch_v9.py"
PIPELINE_STATE_REL = "pipeline-state.json"
PIPELINE_VALIDATOR_REL = "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py"
PIPELINE_DEFINITION_REL = "edsgn/scripts/frozen-pipeline-validator/assets/pipeline-definition.json"
PIPELINE_SIBLINGS = (
    "edsgn/scripts/frozen-pipeline-validator/scripts/citation_requirements.py",
    "edsgn/scripts/frozen-pipeline-validator/scripts/delivery_readiness.py",
)
DRIVER_SOURCES = {
    "graph_v8": "edsgn/staging/frozen_graph_state_diagnostic_v8.py",
    "prediction_v5": "edsgn/staging/frozen_prediction_guarded_v5.py",
    "h2_v4": "edsgn/staging/frozen_h2_matched_retention_v4.py",
    "profile_v5": "edsgn/staging/frozen_profile_guarded_v5.py",
    "c06_v5": "edsgn/staging/frozen_split_history_robustness_v5.py",
}
SUPPORTED_VARIANTS = {
    *(f"V{i}" for i in range(10)), "SeqSNN", "GRU", "STID", "DGCRN",
    "GraphWaveNet", "AGCRN", "STAEformer", "DCRNN",
}
DATASET_IDS = {"synthetic", "mooc", "email", "email-episode"}


class AdmissionError(ValueError):
    """Raised when a frozen test job is not linked to its admitted inputs."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AdmissionError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(AdmissionError(f"invalid JSON constant {x}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"JSON object required: {path}")
    return value


def _resolve(root: Path, raw: str, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise AdmissionError("path must be a nonempty string")
    p = Path(raw)
    lexical = p if p.is_absolute() else root / p
    # Inspect the lexical path before resolution: checking only the resolved
    # components would miss a symlink that points back inside RUN.
    try:
        lexical_abs = lexical.absolute()
        if lexical_abs.is_relative_to(root.absolute()):
            current_lex = root.absolute()
            for part in lexical_abs.relative_to(root.absolute()).parts:
                current_lex = current_lex / part
                if current_lex.is_symlink():
                    raise AdmissionError(f"symlink forbidden: {raw}")
    except OSError as exc:
        raise AdmissionError(f"invalid path {raw}: {exc}") from exc
    value = lexical.resolve(strict=must_exist)
    if not value.is_relative_to(root.resolve()):
        raise AdmissionError(f"path escapes RUN: {raw}")
    current = root.resolve()
    try:
        parts = value.relative_to(root.resolve()).parts
    except ValueError as exc:
        raise AdmissionError(f"path escapes RUN: {raw}") from exc
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise AdmissionError(f"symlink forbidden: {raw}")
    return value


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _ref(root: Path, value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise AdmissionError(f"{label} must have exactly path/sha256")
    try:
        path = _resolve(root, value["path"])
    except (OSError, RuntimeError, ValueError) as exc:
        raise AdmissionError(f"{label} path is unavailable: {value.get('path')}") from exc
    digest = value["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or not path.is_file() or _sha(path) != digest):
        raise AdmissionError(f"{label} missing or hash mismatch: {value.get('path')}")
    return path, digest


def _ref_doc(root: Path, path: Path) -> dict[str, str]:
    return {"path": _rel(root, path), "sha256": _sha(path)}


def _add(consumed: dict[str, str], root: Path, path: Path, digest: str) -> None:
    rel = _rel(root, path)
    old = consumed.get(rel)
    if old is not None and old != digest:
        raise AdmissionError(f"conflicting hashes for {rel}")
    consumed[rel] = digest


def _strict_refs(root: Path, rows: Any, label: str) -> dict[str, str]:
    if not isinstance(rows, list):
        raise AdmissionError(f"{label} must be a ref list")
    result = {}
    for row in rows:
        path, digest = _ref(root, row, label)
        rel = _rel(root, path)
        if rel in result:
            raise AdmissionError(f"duplicate {label} path: {rel}")
        result[rel] = digest
    return result


def _dataset_identity(dataset_id: str, dataset_path: str) -> bool:
    text = dataset_path.replace("\\", "/").lower()
    if dataset_id == "email":
        return "email" in text and "episode" not in text
    return dataset_id in text


def _derive_static_blueprint(root: Path, config_path: Path, *, dataset_id: str,
                             comparison_id: str) -> tuple[dict, dict[str, str]]:
    if dataset_id not in DATASET_IDS or not comparison_id:
        raise AdmissionError("unsupported dataset_id or empty comparison_id")
    old_root = binding.ROOT
    binding.ROOT = root.resolve()
    try:
        config, config_source, config_sha = binding._load_config(config_path)
        if config_source is None or config_source.resolve() != config_path.resolve():
            raise AdmissionError("static config must have a file-backed config source")
        variant = config.get("variant")
        dataset_path = config.get("dataset_path")
        if variant not in SUPPORTED_VARIANTS or not isinstance(dataset_path, str):
            raise AdmissionError("config must name a supported training_variant and dataset_path")
        if not _dataset_identity(dataset_id, dataset_path):
            raise AdmissionError("dataset_id does not match the bound config dataset_path")
        dataset_root, dataset_paths = binding._dataset_source_files(dataset_path)
        prior_paths = sorted(set(binding._fixed_prior_paths(config, check_hashes=True)))
        code_paths = binding._code_paths(str(variant))
        doc = {
            "schema_version": 1,
            "scope": "prospective-ordinary-static-input-blueprint",
            "status": "prospective-input-blueprint",
            "comparison_id": comparison_id,
            "dataset_id": dataset_id,
            "config_source": {
                "path": str(config_source), "sha256": config_sha,
                "canonical_config_sha256": binding._canonical_hash(config), "config": config,
            },
            "dataset_root": str(dataset_root),
            "dataset_sources": binding._entries(sorted(set(dataset_paths + prior_paths))),
            "variant": str(variant),
            "fixed_prior_sources": binding._entries(prior_paths),
            "code_sources": binding._entries(code_paths),
            "hash_policy": "SHA-256 of all .npz/.json files under dataset directory; filename labels do not exclude files; hashes do not load array contents",
        }
        refs = {str(config_source): config_sha}
        for field in ("dataset_sources", "fixed_prior_sources", "code_sources"):
            for row in doc[field]:
                refs[row["path"]] = row["sha256"]
        return doc, refs
    finally:
        binding.ROOT = old_root


def write_static_input_blueprint(root: str | Path, config_ref: dict,
                                 output_path: str | Path, dataset_id: str,
                                 comparison_id: str) -> dict[str, str]:
    """Write one immutable metadata-only blueprint from an existing config."""
    root = Path(root).resolve(strict=True)
    config_path, _ = _ref(root, config_ref, "config")
    doc, _ = _derive_static_blueprint(root, config_path, dataset_id=dataset_id,
                                      comparison_id=comparison_id)
    output = _resolve(root, str(output_path), must_exist=False)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(doc, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return _ref_doc(root, output)


def _official_gate(root: Path, matrix_ref: dict, consumed: dict[str, str]) -> dict:
    state_path = _resolve(root, PIPELINE_STATE_REL)
    validator_path = _resolve(root, PIPELINE_VALIDATOR_REL)
    definition_path = _resolve(root, PIPELINE_DEFINITION_REL)
    sibling_paths = [_resolve(root, p) for p in PIPELINE_SIBLINGS]
    spec = importlib.util.spec_from_file_location("_ordinary_pipeline_state", validator_path)
    if spec is None or spec.loader is None:
        raise AdmissionError("cannot import official pipeline validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = _json(state_path)
    errors = module.validate(state, root, before="P4")
    if errors:
        raise AdmissionError("official canonical pipeline gate failed: " + "; ".join(errors))
    stages = {row.get("id"): row for row in state.get("stages", [])}
    for stage in ("P2", "P3", "P3A"):
        row = stages.get(stage)
        if not isinstance(row, dict) or row.get("status") != "passed" or row.get("gate_verdict") != "pass":
            raise AdmissionError(f"canonical {stage} must be passed/pass")
    p4 = stages.get("P4")
    if not isinstance(p4, dict) or p4.get("status") != "running":
        raise AdmissionError("canonical P4 must be running")
    matrix_path, matrix_sha = _ref(root, matrix_ref, "admission matrix")
    p3a = stages["P3A"]
    matrix_rel = _rel(root, matrix_path)
    found = []
    for role in ("inputs", "evidence"):
        rows = p3a.get(role)
        if not isinstance(rows, list):
            continue
        for item in rows:
            if isinstance(item, dict) and item.get("path") == matrix_rel:
                found.append(item.get("sha256"))
    if found != [matrix_sha]:
        raise AdmissionError("exact admission matrix path/hash must be bound once in P3A inputs or evidence")
    for path in [state_path, validator_path, definition_path, *sibling_paths]:
        _add(consumed, root, path, _sha(path))
    # The official validator requires hashes for every passed stage input and evidence.
    for row in state["stages"]:
        if row.get("status") != "passed":
            continue
        for item in [*row.get("inputs", []), *row.get("evidence", [])]:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise AdmissionError(f"malformed canonical phase ref: {row.get('id')}")
            path, digest = _ref(root, item, f"canonical {row['id']} evidence")
            _add(consumed, root, path, digest)
    return state


def _matrix_doc(root: Path, matrix_ref: dict, consumed: dict[str, str]) -> tuple[Path, dict]:
    path, digest = _ref(root, matrix_ref, "admission matrix")
    doc = _json(path)
    if (set(doc) != {"schema_version", "scope", "status", "source_refs", "jobs"}
            or doc.get("schema_version") != 1
            or doc.get("scope") != "prospective_offline_test_job_admission_matrix"
            or doc.get("status") != "accepted_before_training_test"):
        raise AdmissionError("admission matrix schema/scope/status mismatch")
    _add(consumed, root, path, digest)
    source_refs = _strict_refs(root, doc["source_refs"], "matrix source")
    required_sources = {SOURCE_REL, GLOBAL_COMPLETION_REL, DISPATCHER_REL}
    if not required_sources.issubset(source_refs):
        raise AdmissionError("P3A matrix source_refs must pin admission, global completion and dispatcher9 sources")
    for rel, source_sha in source_refs.items():
        _add(consumed, root, _resolve(root, rel), source_sha)
    rows = doc["jobs"]
    fields = {"job_id", "driver", "method_id", "seed_id", "training_variant", "training_preparation",
              "expected_driver_arguments", "output_dir", "phase_receipt_alias_path"}
    if not isinstance(rows, list) or not rows:
        raise AdmissionError("matrix jobs must be nonempty")
    ids, outputs, aliases, checkpoint_parents = set(), set(), set(), set()
    for row in rows:
        if not isinstance(row, dict) or frozenset(row) not in {frozenset(fields), frozenset(fields | {"checkpoint_route"})}:
            raise AdmissionError("matrix job row does not match exact schema")
        if (not all(isinstance(row[k], str) and row[k] for k in ("job_id", "driver", "method_id", "training_variant"))
                or type(row["seed_id"]) is not int or row["seed_id"] < 0):
            raise AdmissionError("matrix job identity fields malformed")
        if row["driver"] not in DRIVER_SOURCES or row["training_variant"] not in SUPPORTED_VARIANTS:
            raise AdmissionError("unsupported driver or training variant in matrix")
        route = row.get("checkpoint_route")
        if "checkpoint_route" in row and route is None:
            raise AdmissionError("checkpoint_route may be omitted but cannot be null")
        if route is not None:
            if (not isinstance(route, dict) or set(route) != {"mode", "source_path", "execution_path"}
                    or route.get("mode") != "byte_identical_copy"
                    or not isinstance(route.get("source_path"), str)
                    or not isinstance(route.get("execution_path"), str)):
                raise AdmissionError("checkpoint_route must be an exact byte_identical_copy descriptor")
            execution = _resolve(root, route["execution_path"], must_exist=False)
            if not _rel(root, execution).startswith("runs/frozen-replay-checkpoints-v1/"):
                raise AdmissionError("byte-identical execution checkpoint must be under the frozen replay root")
            parent = str(execution.parent).casefold()
            if parent in checkpoint_parents:
                raise AdmissionError("each replay checkpoint copy must have a distinct parent directory")
            checkpoint_parents.add(parent)
            route_prep = row.get("training_preparation")
            if not isinstance(route_prep, dict) or route_prep.get("mode") != "static_config":
                raise AdmissionError("byte-identical replay route requires a static training config")
            route_config_path, route_config_sha = _ref(root, route_prep.get("config"),
                                                       "checkpoint route training config")
            route_config = _json(route_config_path)
            route_source = _resolve(root, route["source_path"], must_exist=False)
            expected_source = _resolve(root, str(Path(route_config.get("output_dir", "")) / "checkpoint.pt"),
                                       must_exist=False)
            if route_source != expected_source:
                raise AdmissionError("checkpoint_route source must equal config output_dir/checkpoint.pt")
            route_flags = _flag_map(row["expected_driver_arguments"])
            if "--checkpoint" not in route_flags or _resolve(root, route_flags["--checkpoint"], must_exist=False) != execution:
                raise AdmissionError("checkpoint_route execution_path must equal matrix --checkpoint")
            _add(consumed, root, route_config_path, route_config_sha)
        if row["job_id"] in ids:
            raise AdmissionError("duplicate matrix job_id")
        ids.add(row["job_id"])
        out = _resolve(root, row["output_dir"], must_exist=False)
        alias = _resolve(root, row["phase_receipt_alias_path"], must_exist=False)
        if not _rel(root, out).startswith("runs/") or not _rel(root, alias).startswith("runs/") or alias.parent.name != "evidence":
            raise AdmissionError("matrix output/phase alias must be under RUN runs/... and alias under evidence")
        if str(out).casefold() in outputs or str(alias).casefold() in aliases:
            raise AdmissionError("duplicate matrix output or phase alias path")
        outputs.add(str(out).casefold()); aliases.add(str(alias).casefold())
        argv = row["expected_driver_arguments"]
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise AdmissionError("expected_driver_arguments must be a nonempty exact argv string list")
        if not isinstance(row["training_preparation"], dict):
            raise AdmissionError("training_preparation must be an object")
    return path, doc


def _flag_map(argv: list[str]) -> dict[str, str]:
    if len(argv) % 2:
        raise AdmissionError("driver argv must consist of flag/value pairs")
    flags = {}
    for key, value in zip(argv[0::2], argv[1::2]):
        if not key.startswith("--") or key in flags:
            raise AdmissionError("driver argv has invalid or duplicate flags")
        flags[key] = value
    return flags


def _validate_static_preparation(root: Path, row: dict, config_ref: dict,
                                  blueprint_ref: dict, binding_path_ref: str,
                                  consumed: dict[str, str]) -> tuple[dict, Path]:
    prep = row["training_preparation"]
    expected = {"mode", "config", "input_blueprint", "expected_training_binding_path"}
    if set(prep) != expected or prep.get("mode") != "static_config":
        raise AdmissionError("static training_preparation schema mismatch")
    if prep["config"] != config_ref or prep["input_blueprint"] != blueprint_ref:
        raise AdmissionError("static config/blueprint refs differ from admission binding")
    config_path, config_sha = _ref(root, config_ref, "training config")
    blueprint_path, blueprint_sha = _ref(root, blueprint_ref, "static input blueprint")
    config = _json(config_path)
    if config.get("variant") != row["training_variant"] or config.get("seed") != row["seed_id"]:
        raise AdmissionError("static config variant/seed differs from matrix row")
    out = config.get("output_dir")
    if not isinstance(out, str) or not out:
        raise AdmissionError("static training config must declare output_dir")
    expected_binding = _resolve(root, str(Path(out) / "input-binding-at-launch.json"), must_exist=False)
    if (_resolve(root, prep["expected_training_binding_path"], must_exist=False) != expected_binding
            or _resolve(root, binding_path_ref) != expected_binding):
        raise AdmissionError("training binding path differs from static config output")
    blueprint = _json(blueprint_path)
    comparison_id = row["method_id"]
    if blueprint.get("comparison_id") != comparison_id:
        raise AdmissionError("static blueprint comparison_id must equal matrix method_id")
    dataset_id = blueprint.get("dataset_id")
    derived, refs = _derive_static_blueprint(root, config_path, dataset_id=dataset_id,
                                             comparison_id=comparison_id)
    if blueprint != derived:
        raise AdmissionError("static blueprint bytes/content differ from current config/data/code inventories")
    for rel, digest in refs.items():
        _add(consumed, root, _resolve(root, rel), digest)
    _add(consumed, root, config_path, config_sha)
    _add(consumed, root, blueprint_path, blueprint_sha)
    return config, expected_binding


def _validate_selected_preparation(root: Path, row: dict, config_ref: dict,
                                   training_binding_ref: dict, selection_ref: dict,
                                   consumed: dict[str, str]) -> tuple[dict, Path]:
    prep = row["training_preparation"]
    if set(prep) != {"mode", "preparation"} or prep.get("mode") != "selected_candidate":
        raise AdmissionError("selected training_preparation schema mismatch")
    candidate_prep = prep["preparation"]
    if candidate_prep.get("comparison_id") != row["method_id"]:
        raise AdmissionError("selected candidate comparison_id must equal matrix method_id")
    try:
        old_root = candidates._binding.ROOT
        candidates._binding.ROOT = root.resolve()
        try:
            candidate_consumed = candidates.validate_preparation(root, candidate_prep, row["seed_id"],
                                                                   allow_postrun_outputs=True)
            candidate_row = {"candidate_selection": selection_ref, "config": config_ref,
                             "training_binding": training_binding_ref}
            selected_consumed = candidates.validate_selected_training(root, candidate_prep,
                                                                        row["seed_id"], candidate_row)
        finally:
            candidates._binding.ROOT = old_root
    except Exception as exc:
        raise AdmissionError(f"selected candidate linkage failed: {exc}") from exc
    for rel, digest in {**candidate_consumed, **selected_consumed}.items():
        _add(consumed, root, _resolve(root, rel), digest)
    config_path, _ = _ref(root, config_ref, "selected config")
    config = _json(config_path)
    if config.get("seed") != row["seed_id"] or config.get("variant") != row["training_variant"]:
        raise AdmissionError("selected config identity differs from matrix row")
    if not _dataset_identity(_dataset_id_from_config(config), config.get("dataset_path", "")):
        raise AdmissionError("selected config dataset identity is malformed")
    bind_spec = candidate_prep
    pool_path, _ = candidates._ref(root, candidate_prep["candidate_pool"], "candidate pool")
    pool = candidates._json(pool_path)
    seed_specs = [spec for cand in pool["candidates"] for spec in cand["seeds"] if spec.get("seed_id") == row["seed_id"]]
    if len(seed_specs) != 1:
        raise AdmissionError("selected candidate pool lacks unique seed row")
    expected_binding = _resolve(root, seed_specs[0]["training_binding_path"])
    if _resolve(root, str(Path(config["output_dir"]) / "input-binding-at-launch.json")) != expected_binding:
        raise AdmissionError("selected training binding path differs from candidate pool")
    return config, expected_binding


def _validate_training_provenance(root: Path, *, config_path: Path, config: dict,
                                  training_binding_path: Path, phase_ref: dict,
                                  summary_ref: dict, checkpoint_ref: dict,
                                  root_acceptance_ref: dict, consumed: dict[str, str]) -> dict:
    # The unchanged binder reads JSON and hashes source bytes; it never opens NPZ members.
    actual_binding = _json(training_binding_path)
    try:
        old_root = binding.ROOT
        binding.ROOT = root.resolve()
        try:
            status = binding.verify_binding(training_binding_path)
        finally:
            binding.ROOT = old_root
    except Exception as exc:
        raise AdmissionError(f"at-launch training binding verification failed: {exc}") from exc
    if status.get("status") != "verified":
        raise AdmissionError("at-launch training binding did not verify")
    if (actual_binding.get("binding_type") != "at-launch"
            or actual_binding.get("status") != "captured-before-training"):
        raise AdmissionError("training binding must be captured-before-training at launch")
    import datetime as _dt
    try:
        recorded = _dt.datetime.fromisoformat(actual_binding.get("recorded_utc", "").replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AdmissionError("training binding recorded_utc must be a timezone-aware ISO timestamp") from exc
    if recorded.tzinfo is None or recorded.utcoffset() is None:
        raise AdmissionError("training binding recorded_utc must be timezone-aware")
    if actual_binding.get("variant") != config.get("variant"):
        raise AdmissionError("launch binding variant differs from training config")
    cfg_source = actual_binding.get("config_source")
    if (not isinstance(cfg_source, dict) or cfg_source.get("path") != str(config_path)
            or cfg_source.get("sha256") != _sha(config_path)
            or cfg_source.get("config") != config):
        raise AdmissionError("launch binding config projection differs from frozen config bytes")
    phase_path, phase_sha = _ref(root, phase_ref, "training phase receipt")
    phase = _json(phase_path)
    if (phase.get("scope") != "offline-experiment-phase" or phase.get("phase") != "training"
            or phase.get("status") != "completed" or phase.get("returncode") != 0):
        raise AdmissionError("training phase receipt must be completed with returncode 0")
    if _resolve(root, phase.get("output_dir", "")) != _resolve(root, config.get("output_dir", "")):
        raise AdmissionError("training phase output_dir differs from resolved config")
    if _resolve(root, phase.get("binding_path", "")) != training_binding_path:
        raise AdmissionError("training phase receipt does not own actual training binding")
    phase_inputs = _strict_refs(root, phase.get("inputs"), "training receipt input")
    phase_outputs = _strict_refs(root, phase.get("outputs"), "training receipt output")
    config_rel = _rel(root, config_path)
    binding_rel = _rel(root, training_binding_path)
    if config_rel not in phase_inputs or phase_inputs[config_rel] != _sha(config_path):
        raise AdmissionError("training receipt inputs do not bind exact config")
    if not phase_outputs:
        raise AdmissionError("training receipt must bind output hashes")
    summary_path, summary_sha = _ref(root, summary_ref, "training summary")
    checkpoint_path, checkpoint_sha = _ref(root, checkpoint_ref, "checkpoint")
    for path, digest, role in ((summary_path, summary_sha, "training summary"),
                               (checkpoint_path, checkpoint_sha, "checkpoint")):
        rel = _rel(root, path)
        if phase_outputs.get(rel) != digest:
            raise AdmissionError(f"training phase receipt does not bind exact {role}")
    if config_rel not in phase_inputs or binding_rel not in phase_inputs and phase.get("binding_path") != str(training_binding_path):
        # The standard campaign records binding_path separately; some versions
        # also include it in inputs. The exact path/hash itself is consumed below.
        raise AdmissionError("training receipt is not tied to its launch binding")
    summary = _json(summary_path)
    if summary.get("scope") != "full_protocol" or summary.get("test_accessed") is not False:
        raise AdmissionError("training summary must be full_protocol and explicitly test_accessed=false")
    if summary.get("checkpoint_sha256") != checkpoint_sha:
        raise AdmissionError("training summary checkpoint hash differs from admitted checkpoint")
    if phase.get("output_dir") and checkpoint_path.parent != _resolve(root, phase["output_dir"]):
        raise AdmissionError("checkpoint is outside training phase output directory")

    acceptance_path, acceptance_sha = _ref(root, root_acceptance_ref, "root numeric acceptance")
    acceptance = _json(acceptance_path)
    if (acceptance.get("accepted_by") != "root" or acceptance.get("verdict") != "pass"
            or acceptance.get("scope") != "completed-development-seed-validation"
            or acceptance.get("pipeline_advanced") is not False
            or acceptance.get("scientific_hypothesis_accepted") is not False
            or acceptance.get("formal_five_seed_result") is not False
            or acceptance.get("final_test_result") is not False):
        raise AdmissionError("root acceptance must be validation-only and explicitly non-scientific/non-test")
    if (acceptance.get("dataset") != _dataset_id_from_config(config)
            or acceptance.get("variant") != config.get("variant")
            or acceptance.get("model_seed") != config.get("seed")):
        raise AdmissionError("root numeric acceptance dataset/variant/seed identity mismatch")
    evidence = acceptance.get("evidence_sha256")
    if not isinstance(evidence, dict) or not evidence:
        raise AdmissionError("root numeric acceptance evidence_sha256 is missing")
    acceptance_evidence = {}
    for raw, digest in evidence.items():
        path = _resolve(root, raw)
        if not isinstance(digest, str) or _sha(path) != digest:
            raise AdmissionError(f"root numeric acceptance evidence is stale: {raw}")
        acceptance_evidence[_rel(root, path)] = digest
        _add(consumed, root, path, digest)
    required_evidence = {
        _rel(root, config_path): _sha(config_path),
        _rel(root, training_binding_path): _sha(training_binding_path),
        _rel(root, phase_path): phase_sha,
        _rel(root, summary_path): summary_sha,
        _rel(root, checkpoint_path): checkpoint_sha,
    }
    if any(acceptance_evidence.get(path) != digest for path, digest in required_evidence.items()):
        raise AdmissionError("root acceptance must cite exact config/binding/phase/summary/checkpoint bytes")
    numeric_reports = []
    for rel, digest in acceptance_evidence.items():
        path = _resolve(root, rel)
        if path.suffix.lower() != ".json" or path in {acceptance_path, phase_path, summary_path}:
            continue
        try:
            doc = _json(path)
        except AdmissionError:
            continue
        if doc.get("scope") == "full_bound_validation_numeric_audit":
            if doc.get("status") != "passed" or doc.get("split") != "validation" or doc.get("test_arrays_or_test_windows_read") is not False:
                raise AdmissionError("root acceptance cites a numeric report that is not validation-only")
            if (doc.get("seed") != config.get("seed") or doc.get("variant") != config.get("variant")
                    or doc.get("dataset_path") is None
                    or _resolve(root, doc["dataset_path"]) != _resolve(root, config["dataset_path"])
                    or doc.get("checkpoint_sha256") != checkpoint_sha
                    or type(doc.get("n_validation_origins")) is not int or doc["n_validation_origins"] <= 0
                    or doc.get("origins_complete_and_ordered") is not True
                    or doc.get("saved_targets_and_masks_match_validation_source") is not True):
                raise AdmissionError("root numeric audit does not match this complete validation run")
            numeric_reports.append((path, doc))
    if not numeric_reports:
        raise AdmissionError("root acceptance must cite a passed validation-only numeric audit")

    # Include every at-launch binding input and every actual receipt/ref byte.
    for key in ("config_source",):
        value = actual_binding.get(key)
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise AdmissionError("at-launch binding config source malformed")
        p = _resolve(root, value["path"])
        _add(consumed, root, p, value.get("sha256"))
    for field in ("dataset_sources", "code_sources"):
        rows = actual_binding.get(field)
        if not isinstance(rows, list):
            raise AdmissionError(f"at-launch binding {field} missing")
        refs = _strict_refs(root, rows, f"at-launch {field}")
        for rel, digest in refs.items():
            _add(consumed, root, _resolve(root, rel), digest)
    # Verify every phase receipt input/output hash and retain it in the snapshot.
    for refs in (phase_inputs, phase_outputs):
        for rel, digest in refs.items():
            _add(consumed, root, _resolve(root, rel), digest)
    for path, digest in ((training_binding_path, _sha(training_binding_path)), (phase_path, phase_sha),
                         (summary_path, summary_sha), (checkpoint_path, checkpoint_sha),
                         (acceptance_path, acceptance_sha)):
        _add(consumed, root, path, digest)
    return {"summary": summary, "phase": phase, "acceptance": acceptance,
            "checkpoint_sha256": checkpoint_sha, "numeric_report_paths": [_rel(root, p) for p, _ in numeric_reports]}


def _dataset_id_from_config(config: dict) -> str:
    path = str(config.get("dataset_path", "")).replace("\\", "/").lower()
    for dataset_id in ("email-episode", "synthetic", "mooc", "email"):
        if _dataset_identity(dataset_id, path):
            return dataset_id
    raise AdmissionError("cannot identify config dataset family from dataset_path")


def _validate_frozen_policy(root: Path, driver: str, flags: dict[str, str],
                            freeze_path: Path, policy_config_path: Path | None) -> None:
    """Match the requested policy descriptor against frozen metadata, no model load."""
    freeze = _json(freeze_path)
    policies = freeze.get("policies")
    allowed = freeze.get("allowed_policy_digests")
    if not isinstance(policies, dict) or not isinstance(allowed, dict):
        raise AdmissionError("freeze lacks policy descriptors/digests")
    kind = flags.get("--policy", "full")
    if driver == "h2_v4":
        kind = "full"  # H2's reviewed wrapper hardcodes the full policy.
    options = {}
    if policy_config_path is not None:
        options = _json(policy_config_path)
        declared = options.pop("kind", kind)
        if declared != kind:
            raise AdmissionError("policy config kind differs from driver argv")
    matched = []
    for name, descriptor in policies.items():
        if not isinstance(descriptor, dict) or descriptor.get("kind") != kind:
            continue
        if any(key not in descriptor or descriptor[key] != value for key, value in options.items()):
            continue
        digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if allowed.get(name) == digest:
            matched.append(name)
    if len(matched) != 1:
        raise AdmissionError("requested policy descriptor is not uniquely authorized by checkpoint freeze")


def _validate_checkpoint_route(root: Path, route: dict, config: dict,
                               execution_ref: dict, training_ref: dict) -> tuple[Path, str, Path, str]:
    if (not isinstance(route, dict) or set(route) != {"mode", "source_path", "execution_path"}
            or route.get("mode") != "byte_identical_copy"):
        raise AdmissionError("checkpoint_route must be an exact byte_identical_copy descriptor")
    source, source_sha = _ref(root, training_ref, "training checkpoint")
    execution, execution_sha = _ref(root, execution_ref, "execution checkpoint copy")
    expected_source = _resolve(root, str(Path(config["output_dir"]) / "checkpoint.pt"))
    if _resolve(root, route["source_path"]) != expected_source or source != expected_source:
        raise AdmissionError("checkpoint_route source must be config output_dir/checkpoint.pt")
    if _resolve(root, route["execution_path"], must_exist=False) != execution:
        raise AdmissionError("execution checkpoint ref differs from prospective checkpoint_route path")
    if not _rel(root, execution).startswith("runs/frozen-replay-checkpoints-v1/"):
        raise AdmissionError("byte-identical execution checkpoint must be under the frozen replay root")
    if execution_sha != source_sha:
        raise AdmissionError("execution checkpoint copy is not byte-identical to original training checkpoint")
    if execution == source or execution.parent == source.parent:
        raise AdmissionError("execution checkpoint copy must have a distinct path and parent")
    return execution, execution_sha, source, source_sha


def validate_admission(root: str | Path, binding_ref: dict, job_doc: dict,
                       job_path: str | Path) -> dict[str, str]:
    """Validate actual ordinary admission, training provenance, and queue/job link.

    Returns a normalized path -> SHA-256 snapshot map for dispatcher fencing.
    Call only from a post-training, before-test job validator.
    """
    root = Path(root).resolve(strict=True)
    consumed: dict[str, str] = {}
    source_path = _resolve(root, SOURCE_REL)
    global_source = _resolve(root, GLOBAL_COMPLETION_REL)
    dispatch_path = _resolve(root, DISPATCHER_REL)
    for path in (source_path, global_source, dispatch_path):
        _add(consumed, root, path, _sha(path))
    admission_path, admission_sha = _ref(root, binding_ref, "ordinary admission binding")
    admission = _json(admission_path)
    expected_fields = {"schema_version", "scope", "status", "test_feedback_used", "matrix", "matrix_job_id",
                       "global_test_queue", "config", "training_binding", "training_phase_receipt",
                       "training_summary", "checkpoint", "freeze", "policy_config", "candidate_selection",
                       "root_numeric_acceptance", "input_refs"}
    if set(admission) not in {frozenset(expected_fields), frozenset(expected_fields | {"training_checkpoint"})} or admission.get("schema_version") != 1:
        raise AdmissionError("ordinary admission binding exact schema mismatch")
    if (admission.get("scope") != "ordinary_pretest_job_admission"
            or admission.get("status") != "accepted_by_root"
            or admission.get("test_feedback_used") is not False):
        raise AdmissionError("ordinary admission binding must be root-accepted with no test feedback")
    matrix_path, matrix_sha = _ref(root, admission["matrix"], "admission matrix")
    _official_gate(root, admission["matrix"], consumed)
    _, matrix = _matrix_doc(root, admission["matrix"], consumed)
    matches = [row for row in matrix["jobs"] if row["job_id"] == admission["matrix_job_id"]]
    if len(matches) != 1:
        raise AdmissionError("ordinary admission matrix_job_id must identify exactly one frozen row")
    matrix_row = matches[0]
    route = matrix_row.get("checkpoint_route")
    if (route is None) != ("training_checkpoint" not in admission):
        raise AdmissionError("training_checkpoint ref must be present exactly when matrix declares checkpoint_route")
    if not isinstance(job_doc, dict):
        raise AdmissionError("job_doc must be an object")
    job_path = _resolve(root, str(job_path))
    job_sha = _sha(job_path)
    disk_job = _json(job_path)
    if disk_job != job_doc:
        raise AdmissionError("provided job_doc differs from immutable job bytes")
    if (job_doc.get("schema_version") != 2 or job_doc.get("job_id") != matrix_row["job_id"]
            or job_doc.get("driver") != matrix_row["driver"]):
        raise AdmissionError("job spec identity/schema differs from prospective matrix")
    argv = job_doc.get("argv")
    if argv != matrix_row["expected_driver_arguments"]:
        raise AdmissionError("actual job argv differs from exact P3A-prospective arguments")
    flags = _flag_map(argv)
    for required in ("--checkpoint", "--dataset-path", "--output"):
        if required not in flags:
            raise AdmissionError(f"ordinary frozen test job lacks required {required}")
    if "--split" in flags and flags["--split"] != "test":
        raise AdmissionError("ordinary frozen job must target the full test split")
    if matrix_row["driver"] in {"prediction_v5", "h2_v4", "profile_v5", "c06_v5"} and "--split" not in flags:
        raise AdmissionError("selected driver requires an explicit test split")
    if flags["--output"] != matrix_row["output_dir"]:
        raise AdmissionError("ordinary job split/output differs from prospective matrix")
    current_output = _resolve(root, matrix_row["output_dir"], must_exist=False)
    current_alias = _resolve(root, matrix_row["phase_receipt_alias_path"], must_exist=False)
    if current_output.exists() or current_alias.exists():
        raise AdmissionError("current job output and phase receipt alias must be absent before child execution")
    if _resolve(root, flags["--checkpoint"]) != _resolve(root, admission["checkpoint"]["path"]):
        raise AdmissionError("job checkpoint argument differs from admission binding")
    if route is not None and _resolve(root, route["execution_path"]) != _resolve(root, flags["--checkpoint"]):
        raise AdmissionError("job checkpoint argument differs from prospective byte-identical execution path")
    if job_doc.get("phase_receipt_alias") != matrix_row["phase_receipt_alias_path"]:
        raise AdmissionError("job phase receipt alias differs from matrix")
    if not isinstance(job_doc, dict) or set(job_doc) != {
            "schema_version", "job_id", "driver", "argv", "sources",
            "pretest_execution_bindings", "phase_receipt_alias"}:
        raise AdmissionError("job spec must use dispatcher schema_version 2 exact fields")
    if not isinstance(job_doc.get("sources"), list):
        raise AdmissionError("job sources must be explicit path/hash refs")
    job_sources = _strict_refs(root, job_doc["sources"], "job source")
    matrix_sources = _strict_refs(root, matrix["source_refs"], "matrix source")
    if any(job_sources.get(rel) != digest for rel, digest in matrix_sources.items()):
        raise AdmissionError("job sources must preserve every P3A-frozen matrix source ref")
    expected_driver_source = DRIVER_SOURCES[matrix_row["driver"]]
    if expected_driver_source not in job_sources:
        raise AdmissionError("job sources do not pin the selected driver's exact source")
    for rel, digest in job_sources.items():
        if matrix.get("source_refs", []) and rel == expected_driver_source:
            if _sha(_resolve(root, rel)) != digest:
                raise AdmissionError("selected driver source hash is stale")
    for rel, digest in job_sources.items():
        _add(consumed, root, _resolve(root, rel), digest)
    _add(consumed, root, job_path, job_sha)
    _add(consumed, root, admission_path, admission_sha)
    _add(consumed, root, matrix_path, matrix_sha)

    # The queued job is unique and already frozen path-only; the actual binding
    # points back to those queue bytes, avoiding a hash cycle.
    queue_path, queue_sha = _ref(root, admission["global_test_queue"], "global test queue")
    queue_ref = {"path": _rel(root, queue_path), "sha256": queue_sha}
    if _ref(root, admission["global_test_queue"], "global test queue")[1] != queue_sha:
        raise AdmissionError("global queue bytes changed")
    queue, queue_rows = global_completion._load_queue(root, queue_path)
    qmatches = [row for row in queue_rows.values() if row["job_spec_path"] == _rel(root, job_path)]
    if len(qmatches) != 1:
        raise AdmissionError("job spec is not a unique queued job")
    qrow = qmatches[0]
    if (qrow["job_id"] != matrix_row["job_id"] or qrow["driver"] != matrix_row["driver"]
            or qrow["output_dir"] != matrix_row["output_dir"]
            or qrow["phase_receipt_alias_path"] != matrix_row["phase_receipt_alias_path"]):
        raise AdmissionError("global queue identity/output/alias differs from frozen admission matrix")
    bind_path, bind_sha = _ref(root, binding_ref, "ordinary admission binding")
    binding_paths = qrow["pretest_execution_binding_paths"]
    if binding_paths != [_rel(root, bind_path)]:
        raise AdmissionError("ordinary job queue must list exactly its actual admission binding path")
    if admission.get("global_test_queue") != queue_ref:
        raise AdmissionError("admission binding queue ref differs from current queue")
    job_bindings = job_doc.get("pretest_execution_bindings")
    if not isinstance(job_bindings, list) or job_bindings != [binding_ref]:
        raise AdmissionError("job must reference exactly its ordinary admission binding")
    if job_doc.get("phase_receipt_alias") != qrow["phase_receipt_alias_path"]:
        raise AdmissionError("job alias differs from queue")
    _add(consumed, root, queue_path, queue_sha)

    config_path, config_sha = _ref(root, admission["config"], "training config")
    training_binding_path, training_binding_sha = _ref(root, admission["training_binding"], "at-launch training binding")
    if _resolve(root, str(job_path)) != _resolve(root, qrow["job_spec_path"]):
        raise AdmissionError("queued job path mismatch")
    prep = matrix_row["training_preparation"]
    if prep.get("mode") == "static_config":
        config, expected_binding = _validate_static_preparation(
            root, matrix_row, admission["config"], prep.get("input_blueprint"),
            prep.get("expected_training_binding_path"), consumed)
    elif prep.get("mode") == "selected_candidate":
        if not isinstance(admission.get("candidate_selection"), dict):
            raise AdmissionError("selected candidate admission must reference a selection receipt")
        if admission["candidate_selection"].get("path") is None:
            raise AdmissionError("candidate selection ref malformed")
        config, expected_binding = _validate_selected_preparation(
            root, matrix_row, admission["config"], admission["training_binding"],
            admission["candidate_selection"], consumed)
    else:
        raise AdmissionError("unsupported training preparation mode")
    if prep.get("mode") == "static_config" and admission["config"] != prep.get("config"):
        raise AdmissionError("admission config differs from prospective static config")
    if config.get("variant") != matrix_row["training_variant"] or config.get("seed") != matrix_row["seed_id"]:
        raise AdmissionError("training config variant/seed differs from matrix identity")
    if expected_binding != training_binding_path:
        raise AdmissionError("actual training binding path differs from prospective preparation")
    if prep.get("mode") == "static_config":
        blueprint_path, _ = _ref(root, prep["input_blueprint"], "static input blueprint")
        blueprint = _json(blueprint_path)
        actual_binding = _json(training_binding_path)
        projection_fields = ("dataset_root", "dataset_sources", "config_source", "variant",
                             "code_sources", "hash_policy")
        if any(actual_binding.get(key) != blueprint.get(key) for key in projection_fields):
            raise AdmissionError("actual at-launch binding differs from the P3A static input blueprint")
        if _resolve(root, actual_binding.get("dataset_root", "")) != _resolve(root, config["dataset_path"]):
            raise AdmissionError("actual launch dataset_root differs from frozen config dataset_path")

    # Ordinary frozen drivers consume the checkpoint sibling test-freeze file.
    checkpoint_path, checkpoint_sha = _ref(root, admission["checkpoint"], "checkpoint")
    training_checkpoint_path, training_checkpoint_sha = checkpoint_path, checkpoint_sha
    if route is not None:
        checkpoint_path, checkpoint_sha, training_checkpoint_path, training_checkpoint_sha = _validate_checkpoint_route(
            root, route, config, admission["checkpoint"], admission["training_checkpoint"])
    freeze_path, freeze_sha = _ref(root, admission["freeze"], "checkpoint freeze")
    expected_freeze = checkpoint_path.parent / "test-freeze.json"
    if freeze_path != expected_freeze:
        raise AdmissionError("test freeze must be the checkpoint sibling test-freeze.json")
    freeze_doc = _json(freeze_path)
    if freeze_doc.get("checkpoint_sha256") != checkpoint_sha:
        raise AdmissionError("test freeze checkpoint hash differs from execution checkpoint bytes")
    if flags.get("--policy-config"):
        if not isinstance(admission.get("policy_config"), dict):
            raise AdmissionError("job policy-config requires a bound policy_config ref")
        policy_path, policy_sha = _ref(root, admission["policy_config"], "policy config")
        if _resolve(root, flags["--policy-config"]) != policy_path:
            raise AdmissionError("job policy-config argument differs from admission binding")
        _add(consumed, root, policy_path, policy_sha)
    elif admission.get("policy_config") is not None:
        raise AdmissionError("unused policy_config ref must be null")
    _validate_frozen_policy(root, matrix_row["driver"], flags, freeze_path,
                            policy_path if flags.get("--policy-config") else None)
    if _resolve(root, flags["--dataset-path"]) != _resolve(root, config["dataset_path"]):
        raise AdmissionError("test driver dataset path differs from trained config dataset")
    dataset_id = _dataset_id_from_config(config)
    if not _dataset_identity(dataset_id, flags["--dataset-path"]):
        raise AdmissionError("test dataset path differs from ordinary matrix domain")
    if row_prep_dataset := (matrix_row["training_preparation"].get("dataset_id") if isinstance(matrix_row["training_preparation"], dict) else None):
        if row_prep_dataset != dataset_id:
            raise AdmissionError("matrix training preparation dataset identity mismatch")
    if "--freeze" in flags and _resolve(root, flags["--freeze"]) != freeze_path:
        raise AdmissionError("job freeze argument differs from admitted freeze")
    if "--binding" in flags and _resolve(root, flags["--binding"]) != training_binding_path:
        raise AdmissionError("job binding argument differs from actual training binding")
    extra_job_inputs = {}
    if "--recipe-json" in flags:
        recipe_path = _resolve(root, flags["--recipe-json"])
        recipe_sha = _sha(recipe_path)
        extra_job_inputs[_rel(root, recipe_path)] = recipe_sha
        _add(consumed, root, recipe_path, recipe_sha)
    source_job_ref = job_sources[expected_driver_source]
    queue_source_refs = {item["path"]: item["sha256"] for item in qrow["pretest_refs"]}
    if queue_source_refs.get(expected_driver_source) != source_job_ref:
        raise AdmissionError("global queue pretest_refs do not bind actual driver source")
    if admission.get("candidate_selection") is not None and prep.get("mode") != "selected_candidate":
        raise AdmissionError("static-config admission cannot carry candidate_selection")
    _add(consumed, root, config_path, config_sha)
    _add(consumed, root, training_binding_path, training_binding_sha)
    _add(consumed, root, checkpoint_path, checkpoint_sha)
    _add(consumed, root, freeze_path, freeze_sha)

    provenance = _validate_training_provenance(
        root, config_path=config_path, config=config,
        training_binding_path=training_binding_path,
        phase_ref=admission["training_phase_receipt"],
        summary_ref=admission["training_summary"],
        checkpoint_ref=(admission["training_checkpoint"] if route is not None else admission["checkpoint"]),
        root_acceptance_ref=admission["root_numeric_acceptance"], consumed=consumed)
    if route is not None and provenance["checkpoint_sha256"] != checkpoint_sha:
        raise AdmissionError("checkpoint copy differs from phase-owned original checkpoint bytes")
    if provenance["summary"].get("test_accessed") is not False:
        raise AdmissionError("training summary has test feedback")

    input_refs = _strict_refs(root, admission["input_refs"], "admission input")
    # Full launch-binding inventories and exact job sources must enter the
    # dispatcher's immutable before/after snapshot.
    actual_training_binding = _json(training_binding_path)
    required_inputs = {}
    for field in ("dataset_sources", "code_sources"):
        for source_ref in actual_training_binding[field]:
            source_path, source_sha = _ref(root, source_ref, f"training binding {field}")
            required_inputs[_rel(root, source_path)] = source_sha
    config_source = actual_training_binding["config_source"]
    config_source_path = _resolve(root, config_source["path"])
    required_inputs[_rel(root, config_source_path)] = config_source["sha256"]
    required_inputs.update(job_sources)
    required_inputs.update(extra_job_inputs)
    if route is not None:
        required_inputs[_rel(root, training_checkpoint_path)] = training_checkpoint_sha
    for ref_name in ("checkpoint", "freeze"):
        ref_path, ref_sha = _ref(root, admission[ref_name], ref_name)
        required_inputs[_rel(root, ref_path)] = ref_sha
    if admission.get("policy_config") is not None:
        ref_path, ref_sha = _ref(root, admission["policy_config"], "policy config")
        required_inputs[_rel(root, ref_path)] = ref_sha
    if any(input_refs.get(rel) != digest for rel, digest in required_inputs.items()):
        raise AdmissionError("input_refs must contain every training input and job source at exact hash")
    for rel, digest in input_refs.items():
        _add(consumed, root, _resolve(root, rel), digest)

    # Queue's independently hashed refs must cover every pre-existing file
    # source passed to the job and every admission input, excluding the
    # post-freeze binding itself and the queue bytes that would form a cycle.
    qrefs = {item["path"]: item["sha256"] for item in qrow["pretest_refs"]}
    required_qrefs = dict(input_refs)
    required_qrefs.update(job_sources)
    required_qrefs.update(extra_job_inputs)
    for ref_name in ("matrix", "config", "training_binding", "training_phase_receipt", "training_summary",
                     "checkpoint", "freeze", "root_numeric_acceptance"):
        path, digest = _ref(root, admission[ref_name], ref_name)
        if path != admission_path and path != queue_path:
            required_qrefs[_rel(root, path)] = digest
    if route is not None:
        path, digest = _ref(root, admission["training_checkpoint"], "training checkpoint")
        required_qrefs[_rel(root, path)] = digest
    if admission.get("policy_config") is not None:
        path, digest = _ref(root, admission["policy_config"], "policy config")
        required_qrefs[_rel(root, path)] = digest
    if admission.get("candidate_selection") is not None:
        path, digest = _ref(root, admission["candidate_selection"], "candidate selection")
        required_qrefs[_rel(root, path)] = digest
    if any(qrefs.get(rel) != digest for rel, digest in required_qrefs.items()):
        raise AdmissionError("global queue pretest_refs omit or stale-bind an independent admission/job input")

    # Recheck every consumed byte immediately before returning the snapshot.
    for rel, digest in consumed.items():
        path = _resolve(root, rel)
        if not path.is_file() or _sha(path) != digest:
            raise AdmissionError(f"admission input changed during validation: {rel}")
    return dict(sorted(consumed.items()))
