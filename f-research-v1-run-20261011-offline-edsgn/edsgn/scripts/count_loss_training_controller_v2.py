"""Prepared-only serial controller for the matched V4 Poisson-loss control.

The scientific training loop remains the unchanged bound_phase_campaign loop,
called through private function globals. This script is not an execution
authorization: all candidate selection, root acceptance, P3/P3A gates, and the
official before-P4 validator must independently pass before a child is spawned.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = ROOT / "edsgn" / "staging"
PYTHON = Path(r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe")
ADAPTER = ROOT / "edsgn" / "staging" / "count_loss_training_adapter_v1.py"
BOUND_CONTROLLER = ROOT / "edsgn" / "scripts" / "bound_phase_campaign.py"
BASELINE_CONTROLLER = ROOT / "edsgn" / "scripts" / "baseline_campaign.py"
CATALOG = ROOT / "protocols" / "core-formal-default-candidate-catalog-v2.json"
ADAPTER_ACCEPTANCE = ROOT / "reviews" / "count-loss-training-adapter-root-engineering-v1.json"
BLUEPRINT_ACCEPTANCE = ROOT / "reviews" / "count-loss-config-blueprints-root-preparation-v3.json"
VALIDATOR = ROOT / "edsgn" / "scripts" / "frozen-pipeline-validator" / "scripts" / "pipeline_state.py"
RESOURCE_LOCK = ROOT / "logs" / "model-resource.lock"
CAMPAIGN_LOCK = ROOT / "logs" / "offline-phase-wall" / "campaign.lock"
WALL_BUDGET_S = 72 * 3600
EXPECTED_BOUND_SHA256 = "12ccfce1a6c9df0341ede4df14e37c8be04ac2cee78c96b4f75dde6d13d68bd3"
EXPECTED_POOL_PATHS = {
    "email": "protocols/core-formal-default-candidate-pools-v2/email-V4.json",
    "mooc": "protocols/core-formal-default-candidate-pools-v2/mooc-V4.json",
}


class ControllerError(RuntimeError):
    pass


def _staging_module(name: str):
    for directory in (ROOT / "edsgn" / "src", ROOT / "edsgn" / "scripts", STAGING_DIR):
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module(name)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ControllerError(f"duplicate JSON key {key!r}: {path}")
            value[key] = item
        return value
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                            parse_constant=lambda x: (_ for _ in ()).throw(ControllerError(f"invalid JSON constant {x}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ControllerError(f"expected JSON object: {path}")
    return result


def _resolve(root: Path, raw: str, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ControllerError("path must be a nonempty string")
    p = Path(raw)
    p = p if p.is_absolute() else root / p
    try:
        p = p.resolve(strict=must_exist)
    except OSError as exc:
        raise ControllerError(f"missing or invalid path: {raw}") from exc
    if not p.is_relative_to(root.resolve()):
        raise ControllerError(f"path escapes RUN root: {raw}")
    return p


def _ref(root: Path, value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ControllerError(f"{label} must contain exactly path and sha256")
    path = _resolve(root, value["path"])
    digest = value["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or sha256(path) != digest:
        raise ControllerError(f"{label} is missing or hash-mismatched")
    return path, digest


def _phase_snapshot(root: Path) -> dict[str, str]:
    """Pin canonical state plus every P2/P3/P3A input/evidence artifact."""
    state_path = root / "pipeline-state.json"
    state = _json(state_path)
    stages = state.get("stages")
    if not isinstance(stages, list):
        raise ControllerError("canonical pipeline state has no stages list")
    rows = [row for row in stages if isinstance(row, dict)]
    by_id = {row.get("id"): row for row in rows}
    if len(by_id) != len(rows):
        raise ControllerError("canonical pipeline state contains duplicate stage IDs")
    for name in ("P2", "P3", "P3A"):
        row = by_id.get(name, {})
        if row.get("status") != "passed" or row.get("gate_verdict") != "pass":
            raise ControllerError(f"{name} must be passed/pass")
    if by_id.get("P4", {}).get("status") != "running":
        raise ControllerError("P4 must be running")
    pinned = {state_path: sha256(state_path)}
    for name in ("P2", "P3", "P3A"):
        row = by_id[name]
        entries = [*row.get("inputs", []), *row.get("evidence", [])]
        if name == "P3A" and not entries:
            raise ControllerError("P3A must expose hash-bound inputs/evidence")
        for entry in entries:
            path, digest = _ref(root, entry, f"{name} input/evidence")
            pinned[path] = digest
    return {str(path): digest for path, digest in sorted(pinned.items(), key=lambda x: str(x[0]))}


def official_before_p4(root: str | Path, *, run=subprocess.run) -> dict[str, str]:
    root = Path(root).resolve()
    before = _phase_snapshot(root)
    command = [str(PYTHON), str(root / "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py"),
               str(root / "pipeline-state.json"), "--before", "P4"]
    proc = run(command, cwd=root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ControllerError(f"official --before P4 validator rejected canonical state: {proc.stdout} {proc.stderr}")
    after = _phase_snapshot(root)
    if before != after:
        raise ControllerError("canonical state or passed evidence changed during official before-P4 validation")
    return before


def _accepted_record(root: Path, ref: Any, *, scope: str, evidence_map: dict[str, str] | None = None) -> tuple[Path, str, dict]:
    path, digest = _ref(root, ref, f"root acceptance ({scope})")
    doc = _json(path)
    if doc.get("accepted_by") != "root" or doc.get("verdict") != "pass" or doc.get("scope") != scope:
        raise ControllerError(f"root acceptance is not accepted for {scope}")
    if doc.get("scientific_hypothesis_accepted") is not False or doc.get("pipeline_advanced") is not False:
        raise ControllerError("engineering acceptance must not claim scientific acceptance or phase advancement")
    if evidence_map is not None and doc.get("evidence_sha256") != evidence_map:
        raise ControllerError("root acceptance evidence map differs from controller inputs")
    return path, digest, doc


def _validate_job(root: Path, job: dict, acceptance_refs: dict[str, Any], *, adapter_module=None) -> dict[str, Path]:
    required = {"id", "phase", "dataset", "seed", "variant", "scope", "args", "config", "output_dir", "candidate_pool",
                "candidate_selection", "static_blueprint", "mse_numeric_acceptance", "mse_numeric_root_acceptance", "inputs"}
    if not isinstance(job, dict) or set(job) != required:
        raise ControllerError("job schema must be exact and include selected policy and MSE evidence")
    dataset = job["dataset"]
    if (dataset not in {"email", "mooc"} or job["variant"] != "V4"
            or type(job["seed"]) is not int or job["seed"] != 0 or job["phase"] != "training"):
        raise ControllerError("count-loss jobs are limited to original email/MOOC V4 seed 0")
    if job["scope"] != "formal":
        raise ControllerError("count-loss control requires formal scope")
    config_path, config_sha = _ref(root, job["config"], "adapter config")
    if job["args"] != ["--config", str(config_path)]:
        raise ControllerError("training command must pass exactly adapter --config <bound config>")
    cfg = _json(config_path)
    if cfg.get("dataset_path", "").casefold().replace("\\", "/").endswith(f"/{dataset}/{dataset}.npz") is False:
        raise ControllerError("adapter config dataset does not match queue row")
    if cfg.get("variant") != "V4" or type(cfg.get("seed")) is not int or cfg["seed"] != 0:
        raise ControllerError("adapter config identity mismatch")
    expected_pool = _resolve(root, EXPECTED_POOL_PATHS[dataset])
    pool_path, pool_sha = _ref(root, job["candidate_pool"], "candidate pool")
    if pool_path != expected_pool:
        raise ControllerError("candidate pool is not the frozen V4 pool for this dataset")
    pool = _json(pool_path)
    if (pool.get("scope") != "prospective_training_candidate_pool" or pool.get("dataset_id") != dataset
            or pool.get("variant") != "V4" or len(pool.get("candidates", [])) != 1
            or pool["candidates"][0].get("candidate_id") != "g00-m00"
            or pool["candidates"][0].get("graph_candidate_id") != "g00"
            or pool["candidates"][0].get("memory_candidate_id") != "m00"):
        raise ControllerError("only exact singleton g00/m00 V4 candidate pool is eligible")
    candidate = pool["candidates"][0]
    selection_path, selection_sha = _ref(root, job["candidate_selection"], "selected candidate evidence")
    selection = _json(selection_path)
    if (selection.get("accepted_by") != "root" or selection.get("status") != "accepted"
            or selection.get("scope") != "root_validation_only_candidate_selection"
            or selection.get("test_feedback_used") is not False
            or selection.get("comparison_id") != pool.get("comparison_id")
            or selection.get("selected_candidate_id") != "g00-m00"
            or selection.get("pool") != {"path": pool_path.relative_to(root).as_posix(), "sha256": pool_sha}):
        raise ControllerError("g00/m00 policy selection lacks an exact root-accepted validation-only receipt")
    if not isinstance(selection.get("candidates"), list) or [x.get("candidate_id") for x in selection["candidates"]] != ["g00-m00"]:
        raise ControllerError("selection receipt must retain the complete singleton candidate ledger")
    selected_config, selected_sha = _ref(root, candidate["seeds"][0]["config"], "selected MSE policy config")
    if _resolve(root, cfg["mse_reference_config"]["path"]) != selected_config:
        raise ControllerError("adapter MSE reference differs from selected g00/m00 config")
    # Reuse the accepted metadata helper for all candidate pool/config/blueprint
    # inputs, then independently compare the adapter config's effective fields.
    candidate_helper = (_staging_module("prospective_training_candidate_binding_v1")
                        if adapter_module is None else adapter_module)
    prep = {"schema_version": 4, "comparison_id": pool["comparison_id"],
            "candidate_pool": {"path": pool_path.relative_to(root).as_posix(), "sha256": pool_sha}}
    consumed = candidate_helper.validate_preparation(root, prep, 0, allow_postrun_outputs=True)
    selection_doc, selection_path_checked, selection_sha_checked = candidate_helper._selection_doc(
        root, job["candidate_selection"], pool, pool_path, pool_sha, 0)
    if (selection_path_checked != selection_path or selection_sha_checked != selection_sha
            or selection_doc.get("selected_candidate_id") != "g00-m00"):
        raise ControllerError("selection helper did not verify the exact g00/m00 receipt")
    # The helper attaches validated pilot/GRU evidence refs in-memory; bind every
    # consumed file into the child job input list as well as the receipt itself.
    for rel, digest in selection_doc.get("_consumed_evidence", {}).items():
        prior = consumed.get(rel)
        if prior is not None and prior != digest:
            raise ControllerError(f"conflicting selected-policy evidence hash: {rel}")
        consumed[rel] = digest
    mse_ref_path = _resolve(root, cfg["mse_reference_config"]["path"])
    if sha256(mse_ref_path) != cfg["mse_reference_config"]["sha256"] or sha256(mse_ref_path) != selected_sha:
        raise ControllerError("MSE reference bytes differ from selected policy config")
    if candidate_helper._json(selection_path).get("selected_candidate_id") != "g00-m00":
        raise ControllerError("selected candidate identity mismatch")
    # The controller config may change only output location and loss/reference
    # declaration; the accepted adapter performs the canonical equality check.
    loss_adapter = _staging_module("count_loss_training_adapter_v1")
    loss_adapter._validate_count_config(cfg, root)

    blueprint_path, blueprint_sha = _ref(root, job["static_blueprint"], "count-loss static blueprint")
    prep_path = root / "reviews/count-loss-config-blueprints-root-preparation-v3.json"
    prep_acceptance = _json(prep_path)
    if (prep_acceptance.get("accepted_by") != "root" or prep_acceptance.get("verdict") != "pass"
            or prep_acceptance.get("scope") != "paired_count_loss_configs_static_blueprints_preparation_only"
            or prep_acceptance.get("actual_training_completed") is not False
            or prep_acceptance.get("actual_admission_accepted") is not False):
        raise ControllerError("count-loss blueprint root preparation record is invalid")
    prepared_row = next((r for r in prep_acceptance.get("rows", [])
                         if r.get("dataset") == dataset and r.get("method_id") == "V4-Poisson" and r.get("seed_id") == 0), None)
    expected_config_ref = {"path": config_path.relative_to(root).as_posix(), "sha256": config_sha}
    if (not prepared_row or prepared_row.get("config") != expected_config_ref
            or prepared_row.get("blueprint") != job["static_blueprint"]
            or _resolve(root, prepared_row.get("training_binding_path", ""), must_exist=False)
            != (_resolve(root, cfg["output_dir"], must_exist=False) / "input-binding-at-launch.json")):
        raise ControllerError("count-loss config/blueprint differs from root-prepared V4-Poisson identity")
    for rel, digest in prep_acceptance.get("evidence_sha256", {}).items():
        if sha256(_resolve(root, rel)) != digest:
            raise ControllerError(f"count-loss blueprint preparation evidence changed: {rel}")
    ordinary = _staging_module("ordinary_frozen_test_admission_v1")
    expected_blueprint, blueprint_refs = ordinary._derive_static_blueprint(
        root, config_path, dataset_id=dataset, comparison_id="V4-Poisson")
    if _json(blueprint_path) != expected_blueprint:
        raise ControllerError("count-loss static blueprint differs from current config/data/code inputs")

    numeric_ref = job["mse_numeric_acceptance"]
    numeric_path, numeric_sha = _ref(root, numeric_ref, "full MSE numeric acceptance")
    numeric = _json(numeric_path)
    required_horizons = [1, 3, 7] if dataset == "email" else [1, 6, 24]
    mse_config = _json(selected_config)
    if (numeric.get("status") != "passed" or numeric.get("scope") != "full_bound_validation_numeric_audit"
            or numeric.get("variant") != "V4"
            or numeric.get("seed") != 0 or numeric.get("split") != "validation"
            or numeric.get("test_arrays_or_test_windows_read") is not False
            or numeric.get("config_path") != str(selected_config)
            or numeric.get("binding", {}).get("status") != "verified"
            or numeric.get("binding", {}).get("binding_path") != str(_resolve(root, mse_config.get("output_dir", "")) / "input-binding-at-launch.json")
            or numeric.get("checkpoint_sha256") != sha256(_resolve(root, mse_config.get("output_dir", "")) / "checkpoint.pt")
            or numeric.get("training", {}).get("summary_test_accessed") is not False
            or numeric.get("training", {}).get("checkpoint_metadata_loaded") is not False
            or numeric.get("dataset_path") != mse_config.get("dataset_path")
            or numeric.get("run_dir") != mse_config.get("output_dir")
            or numeric.get("required_horizons") != required_horizons
            or numeric.get("origins_complete_and_ordered") is not True
            or numeric.get("saved_targets_and_masks_match_validation_source") is not True
            or type(numeric.get("n_validation_origins")) is not int
            or numeric["n_validation_origins"] < 1):
        raise ControllerError("full MSE validation numeric audit is absent or mismatched")
    root_numeric_ref = job.get("mse_numeric_root_acceptance")
    numeric_root_path, numeric_root_sha = _ref(root, root_numeric_ref, "full MSE numeric root acceptance")
    numeric_root = _json(numeric_root_path)
    if (numeric_root.get("accepted_by") != "root" or numeric_root.get("verdict") != "pass"
            or numeric_root.get("scope") != "completed-development-seed-validation"
            or numeric_root.get("dataset") != dataset or numeric_root.get("variant") != "V4"
            or numeric_root.get("model_seed") != 0 or numeric_root.get("formal_five_seed_result") is not False
            or numeric_root.get("final_test_result") is not False
            or numeric_root.get("required_horizon_mean_mae") != numeric.get("required_horizon_mean_selection_mae")
            or numeric_root.get("evidence_sha256", {}).get(numeric_path.relative_to(root).as_posix()) != numeric_sha):
        raise ControllerError("root acceptance does not pin a full MSE validation numeric audit")
    for rel, digest in numeric_root.get("evidence_sha256", {}).items():
        p = _resolve(root, rel)
        if sha256(p) != digest:
            raise ControllerError(f"numeric root acceptance evidence changed: {rel}")
    expected_inputs = {
        "count_controller": {"path": str(root / "edsgn/scripts/count_loss_training_controller_v2.py"), "sha256": sha256(root / "edsgn/scripts/count_loss_training_controller_v2.py")},
        "count_adapter": {"path": str(root / "edsgn/staging/count_loss_training_adapter_v1.py"), "sha256": sha256(root / "edsgn/staging/count_loss_training_adapter_v1.py")},
        "bound_controller": {"path": str(root / "edsgn/scripts/bound_phase_campaign.py"), "sha256": sha256(root / "edsgn/scripts/bound_phase_campaign.py")},
        "baseline_controller": {"path": str(root / "edsgn/scripts/baseline_campaign.py"), "sha256": sha256(root / "edsgn/scripts/baseline_campaign.py")},
        "adapter_acceptance": {"path": str(root / "reviews/count-loss-training-adapter-root-engineering-v1.json"), "sha256": sha256(root / "reviews/count-loss-training-adapter-root-engineering-v1.json")},
        "candidate_pool": {"path": str(pool_path), "sha256": pool_sha},
        "candidate_selection": {"path": str(selection_path), "sha256": selection_sha},
        "static_blueprint": {"path": str(blueprint_path), "sha256": blueprint_sha},
        "blueprint_preparation_acceptance": {"path": str(prep_path), "sha256": sha256(prep_path)},
        "mse_numeric_acceptance": {"path": str(numeric_path), "sha256": numeric_sha},
        "mse_numeric_root_acceptance": {"path": str(numeric_root_path), "sha256": numeric_root_sha},
        "mse_reference": {"path": str(mse_ref_path), "sha256": sha256(mse_ref_path)},
        "config": {"path": str(config_path), "sha256": config_sha},
    }
    if not isinstance(job["inputs"], list):
        raise ControllerError("job.inputs must be a list of exact frozen input refs")
    supplied = {}
    for ref in job["inputs"]:
        path, digest = _ref(root, ref, "job input")
        supplied[str(path)] = digest
    for item in expected_inputs.values():
        if supplied.get(item["path"]) != item["sha256"]:
            raise ControllerError(f"mandatory job input missing or changed: {item['path']}")
    for rel, digest in prep_acceptance.get("evidence_sha256", {}).items():
        path = _resolve(root, rel)
        if supplied.get(str(path)) != digest:
            raise ControllerError(f"blueprint preparation acceptance evidence missing from job inputs: {rel}")
    for rel, digest in numeric_root.get("evidence_sha256", {}).items():
        path = _resolve(root, rel)
        if supplied.get(str(path)) != digest:
            raise ControllerError(f"full MSE numeric acceptance evidence missing from job inputs: {rel}")
    for rel, digest in consumed.items():
        path = _resolve(root, rel)
        if supplied.get(str(path)) != digest:
            raise ControllerError(f"selected candidate evidence input missing or changed: {rel}")
    for rel, digest in blueprint_refs.items():
        path = _resolve(root, rel)
        if supplied.get(str(path)) != digest:
            raise ControllerError(f"count-loss static blueprint input missing from job inputs: {rel}")
    output = _resolve(root, job["output_dir"], must_exist=False)
    if output != _resolve(root, cfg.get("output_dir"), must_exist=False):
        raise ControllerError("job output differs from count config output_dir")
    if output.exists() and any(output.iterdir()):
        raise ControllerError("count-loss output is nonempty; no overwrite/resume/retry")
    return {"config": config_path, "output": output, "pool": pool_path, "selection": selection_path,
            "numeric": numeric_path, "mse_reference": mse_ref_path, "binding": output / "input-binding-at-launch.json"}


def validate_queue(root: str | Path, queue_path: str | Path, *, validator_run=subprocess.run) -> tuple[dict, dict[str, str]]:
    root = Path(root).resolve()
    qpath = _resolve(root, str(queue_path))
    queue = _json(qpath)
    if queue.get("schema_version") != 1 or queue.get("scope") != "paired-count-loss-v4-seed0-controller":
        raise ControllerError("queue schema/scope mismatch")
    if not isinstance(queue.get("jobs"), list) or len(queue["jobs"]) != 2:
        raise ControllerError("queue must contain exactly email and MOOC controls")
    checked = {}
    for job in queue["jobs"]:
        if job.get("dataset") in checked:
            raise ControllerError("duplicate count-loss dataset job")
        checked[job["dataset"]] = _validate_job(root, job, queue.get("acceptances", {}))
    if set(checked) != {"email", "mooc"}:
        raise ControllerError("queue must include exactly email and MOOC")
    # Root adapter acceptance is engineering-only, but must not imply phase or
    # hypothesis acceptance. Its own evidence map is checked against live bytes.
    acc_path = root / "reviews/count-loss-training-adapter-root-engineering-v1.json"
    acc = _json(acc_path)
    if (acc.get("accepted_by") != "root" or acc.get("verdict") != "pass"
            or acc.get("scope") != "paired_v4_seed0_count_loss_adapter_engineering_only"
            or acc.get("scientific_hypothesis_accepted") is not False
            or acc.get("pipeline_advanced") is not False
            or acc.get("actual_training_completed") is not False
            or acc.get("actual_test_authorized") is not False):
        raise ControllerError("root engineering acceptance for adapter is invalid")
    for rel, digest in acc.get("evidence_sha256", {}).items():
        path = _resolve(root, rel)
        if sha256(path) != digest:
            raise ControllerError(f"adapter root acceptance evidence changed: {rel}")
    pinned = {str(qpath): sha256(qpath), str(root / "edsgn/scripts/count_loss_training_controller_v2.py"): sha256(root / "edsgn/scripts/count_loss_training_controller_v2.py"),
              str(root / "edsgn/staging/count_loss_training_adapter_v1.py"): sha256(root / "edsgn/staging/count_loss_training_adapter_v1.py"), str(root / "edsgn/scripts/bound_phase_campaign.py"): sha256(root / "edsgn/scripts/bound_phase_campaign.py"),
              str(root / "edsgn/scripts/baseline_campaign.py"): sha256(root / "edsgn/scripts/baseline_campaign.py"), str(acc_path): sha256(acc_path),
              str(root / "protocols/core-formal-default-candidate-catalog-v2.json"): sha256(root / "protocols/core-formal-default-candidate-catalog-v2.json")}
    for rel, digest in acc.get("evidence_sha256", {}).items():
        pinned[str(_resolve(root, rel))] = digest
    for job in queue["jobs"]:
        for ref in job["inputs"]:
            path, digest = _ref(root, ref, "queue input")
            pinned[str(path)] = digest
    # Phase evidence is inserted into each private run_queue receipt's input
    # manifest without changing the reviewed on-disk queue bytes.
    phase_pins = official_before_p4(root, run=validator_run)
    pinned.update(phase_pins)
    return queue, pinned


def _owned_processes(root: Path) -> list[dict[str, Any]]:
    import psutil
    active = []
    extra = {"count_loss_training_controller_v2.py", "count_loss_training_adapter_v1.py"}
    for proc in psutil.process_iter(["pid", "cmdline", "cwd", "exe"]):
        try:
            if proc.pid == os.getpid():
                continue
            argv = [str(x).replace("\\", "/").casefold() for x in (proc.info.get("cmdline") or [])]
            exe = str(proc.info.get("exe") or "")
            if not Path(exe).name.casefold().startswith("python"):
                continue
            owned = str(root).casefold() in " ".join(argv) or str(proc.info.get("cwd", "")).casefold() == str(root).casefold()
            is_job = any(Path(x).name in extra for x in argv)
            if owned and is_job:
                active.append({"pid": proc.pid})
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, TypeError):
            continue
    return active


def _private_bound_controller(root: Path = ROOT):
    """Load an isolated copy of the unchanged runner with private globals."""
    import importlib.util
    scripts = root / "edsgn" / "scripts"
    sys.path.insert(0, str(scripts))
    module_name = f"_count_loss_bound_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, scripts / "bound_phase_campaign.py")
    if spec is None or spec.loader is None:
        raise ControllerError("cannot load unchanged bound phase controller")
    bound = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = bound
    spec.loader.exec_module(bound)
    private = dict(bound.run_queue.__globals__)
    real_subprocess = private["subprocess"]
    def command_for(kind, args):
        if kind != "training":
            raise ControllerError("count-loss controller only accepts training jobs")
        config_path = args[args.index("--config") + 1]
        return [str(PYTHON), str(root / "edsgn/staging/count_loss_training_adapter_v1.py"), "--config", config_path]
    private["command_for"] = command_for
    private["_is_training_process"] = lambda: bool(_owned_processes(root))
    private["ROOT"] = root
    private["__file__"] = str(root / "edsgn/scripts/bound_phase_campaign.py")
    # Functions imported from the unchanged module retain their original globals.
    # Rebind those functions to this private namespace so every path/gate is RUN-local.
    for key, value in list(private.items()):
        if isinstance(value, types.FunctionType) and value.__module__ == module_name:
            private[key] = types.FunctionType(value.__code__, private, name=value.__name__,
                                              argdefs=value.__defaults__, closure=value.__closure__)
    runner = types.FunctionType(bound.run_queue.__code__, private, name="run_count_loss_queue",
                                argdefs=bound.run_queue.__defaults__, closure=bound.run_queue.__closure__)
    return runner, private, real_subprocess


def run_queue(queue_path: str | Path, *, root: str | Path = ROOT, validator_run=subprocess.run,
              aggregate_fn=None, child_popen=None):
    root = Path(root).resolve()
    qpath = _resolve(root, str(queue_path))
    queue, pinned = validate_queue(root, qpath, validator_run=validator_run)
    if sha256(root / "edsgn/scripts/bound_phase_campaign.py") != EXPECTED_BOUND_SHA256:
        raise ControllerError("unchanged bound controller does not match reviewed source hash")
    # P3/P3A status, their complete evidence bytes and the official validator
    # are checked before acquiring locks and again immediately before each child.
    phase_pins = official_before_p4(root, run=validator_run)
    if _owned_processes(root):
        raise ControllerError("owned model/training/resource process is active")
    from aggregate_budget import aggregate
    budget = (aggregate if aggregate_fn is None else aggregate_fn)(root)
    if not budget.get("coverage_complete") or budget.get("spent_seconds", WALL_BUDGET_S) >= WALL_BUDGET_S:
        raise ControllerError("cumulative 72-hour budget is incomplete or exhausted")
    resource = root / "logs" / "model-resource.lock"
    resource.parent.mkdir(parents=True, exist_ok=True)
    lease_record = {"pid": os.getpid(), "queue": str(qpath), "queue_sha256": sha256(qpath),
                    "lease_id": uuid.uuid4().hex}
    try:
        with resource.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(lease_record) + "\n")
    except FileExistsError as exc:
        raise ControllerError("model-resource lease already exists; manual verified recovery required") from exc
    owned_resource = True
    try:
        if sha256(qpath) != pinned[str(qpath)]:
            raise ControllerError("queue changed after initial snapshot")
        runner, private, real_subprocess = _private_bound_controller(root)
        original_popen = child_popen or real_subprocess.Popen
        allowed_commands = {
            (str(PYTHON), str(root / "edsgn/staging/count_loss_training_adapter_v1.py"),
             "--config", str(_resolve(root, job["args"][1])))
            for job in queue["jobs"]
        }
        def guarded_popen(command, *args, **kwargs):
            if (not isinstance(command, (list, tuple)) or len(command) != 4
                    or tuple(command) not in allowed_commands):
                raise ControllerError("refusing unexpected child command; exact adapter --config argv required")
            if sha256(qpath) != pinned[str(qpath)] or any(sha256(p) != h for p, h in pinned.items()):
                raise ControllerError("bound input bytes changed immediately before child")
            current_phase_pins = official_before_p4(root, run=validator_run)
            if any(current_phase_pins.get(path) != digest for path, digest in phase_pins.items()):
                raise ControllerError("canonical phase snapshot changed before child")
            if _owned_processes(root):
                raise ControllerError("owned model/training/resource process became active")
            current_budget = (aggregate if aggregate_fn is None else aggregate_fn)(root)
            if not current_budget.get("coverage_complete") or current_budget.get("spent_seconds", WALL_BUDGET_S) >= WALL_BUDGET_S:
                raise ControllerError("cumulative 72-hour budget gate refused child")
            return original_popen(command, *args, **kwargs)
        private["subprocess"] = types.SimpleNamespace(Popen=guarded_popen, STDOUT=real_subprocess.STDOUT)
        private["ROOT"] = root
        private["sha256"] = sha256
        private["aggregate"] = aggregate if aggregate_fn is None else aggregate_fn
        private["WALL_BUDGET_S"] = WALL_BUDGET_S
        original_input_paths = private["command_input_paths"]
        fenced_paths = {Path(path) for path in pinned}
        private["command_input_paths"] = lambda job: original_input_paths({**job, "inputs": []}) | {root / ref["path"] for ref in job.get("inputs", [])} | fenced_paths
        result = runner(qpath)
        return result
    finally:
        if owned_resource and resource.is_file():
            try:
                lease = _json(resource)
                if lease == lease_record:
                    resource.unlink()
            except Exception:
                # An unreadable/changed lease is preserved for manual recovery.
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    args = parser.parse_args(argv)
    run_queue(args.queue)


if __name__ == "__main__":
    main()
