"""Isolated V4 count-data training adapter for the preregistered Poisson control.

This wrapper runs unchanged engine.train_run code with a private function-global
namespace. It is preparation only; it does not authorize training or alter the
engine/package implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import types
from typing import Any

import torch
import torch.nn.functional as F

try:
    from edsgn import engine
except ImportError:  # PYTHONPATH=edsgn/src when invoked as a script
    import engine  # type: ignore


ADAPTER_SOURCE_KEY = "staging/count_loss_training_adapter_v1.py"
POISSON_RECIPE = {
    "kind": "masked_poisson_nll",
    "log_input": False,
    "full": False,
    "reduction": "none",
    "prediction_floor": 1e-8,
    "normalization": "valid_target_cells",
}
RUN_ROOT = Path(__file__).resolve().parents[2]
_FIXED_MODEL_VALUES = {
    "d_total": 64,
    "k": 8,
    "candidate_cap": 32,
    "tau_fast": 1.0,
    "tau_slow": 8.0,
    "beta_init": 1.0,
    "lambda_temporal": 0.01,
}


def masked_poisson_nll(pred: torch.Tensor, targets: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """Mean Poisson NLL over valid target cells, with safe masked entries."""
    if not all(torch.is_tensor(x) for x in (pred, targets, mask)):
        raise TypeError("prediction, targets and mask must be tensors")
    if pred.shape != targets.shape or pred.shape != mask.shape:
        raise ValueError("prediction, targets and mask shapes must match exactly")
    if mask.dtype is not torch.bool:
        raise TypeError("target mask must have bool dtype")
    if pred.device != targets.device or pred.device != mask.device:
        raise ValueError("prediction, targets and mask must share a device")
    if not torch.is_floating_point(pred) or not torch.is_floating_point(targets):
        raise TypeError("prediction and targets must have floating dtypes")
    valid_count = mask.sum()
    if not bool(valid_count):
        raise ValueError("No available supervision target in batch")

    valid_pred = pred.masked_select(mask)
    valid_targets = targets.masked_select(mask)
    if not bool(torch.isfinite(valid_pred).all()):
        raise ValueError("valid predictions must be finite")
    if bool((valid_pred < 0).any()):
        raise ValueError("valid predictions must be nonnegative count means")
    if not bool(torch.isfinite(valid_targets).all()) or bool((valid_targets < 0).any()):
        raise ValueError("valid Poisson targets must be finite and nonnegative")

    # Sanitize invalid cells before evaluating elementwise loss: masked NaN or
    # negative padding must not contaminate either values or gradients.
    safe_pred = torch.where(mask, pred, torch.ones_like(pred)).clamp_min(1e-8)
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    elementwise = F.poisson_nll_loss(safe_pred, safe_targets, log_input=False,
                                    full=False, reduction="none")
    return elementwise.masked_select(mask).sum() / valid_count


def validate_count_config(config: dict[str, Any]) -> None:
    """Fail closed unless this is the immutable V4 seed-0 email/MOOC control."""
    return _validate_count_config(config, RUN_ROOT)


def _resolve_under_run(root: Path, raw: str, *, must_exist: bool) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("MSE reference/output path must be a nonempty string")
    root = root.resolve(strict=True)
    given = Path(raw)
    lexical = given if given.is_absolute() else root / given
    if lexical.resolve(strict=must_exist) != lexical.absolute().resolve(strict=must_exist):
        raise ValueError(f"noncanonical path spelling: {raw}")
    lexical_abs = lexical.absolute()
    if not lexical_abs.is_relative_to(root):
        raise ValueError(f"path is outside canonical RUN: {raw}")
    cursor = root
    for part in lexical_abs.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"symlink forbidden in path: {raw}")
    resolved = lexical.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes canonical RUN: {raw}")
    return resolved


def _read_unique_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {x}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read MSE reference config: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("MSE reference config must be a JSON object")
    return value


def _validate_count_config(config: dict[str, Any], run_root: str | Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping")
    if config.get("variant") != "V4":
        raise ValueError("count-loss adapter variant must be V4")
    if type(config.get("seed")) is not int or config["seed"] != 0:
        raise ValueError("count-loss adapter seed must be exactly 0")
    dataset_path = config.get("dataset_path")
    if not isinstance(dataset_path, str):
        raise ValueError("count-loss adapter dataset_path must be a path string")
    dataset = Path(dataset_path)
    dataset_parts = [part.casefold() for part in dataset.parts]
    if not ((dataset.name.casefold() == "email.npz" and "email" in dataset_parts)
            or (dataset.name.casefold() == "mooc.npz" and "mooc" in dataset_parts)):
        raise ValueError("count-loss adapter is restricted to original email or MOOC count datasets")
    recipe = config.get("training_loss")
    if not isinstance(recipe, dict) or set(recipe) != set(POISSON_RECIPE):
        raise ValueError("training_loss recipe must exactly match the preregistered masked Poisson NLL")
    if (recipe.get("kind") != "masked_poisson_nll"
            or type(recipe.get("log_input")) is not bool or recipe["log_input"] is not False
            or type(recipe.get("full")) is not bool or recipe["full"] is not False
            or recipe.get("reduction") != "none"
            or type(recipe.get("prediction_floor")) is not float or recipe["prediction_floor"] != 1e-8
            or recipe.get("normalization") != "valid_target_cells"):
        raise ValueError("training_loss recipe must exactly match the preregistered masked Poisson NLL")
    if float(config.get("auxiliary_one_step_weight", 0.0)) != 0.0:
        raise ValueError("auxiliary_one_step_weight must be zero for the matched loss control")
    if config.get("resume", False):
        raise ValueError("resume is forbidden for the count-loss control")
    if any(config.get(key) is not None for key in ("smoke_train_windows", "smoke_val_windows")):
        raise ValueError("smoke subsets are forbidden for the count-loss control")
    if config.get("max_run_seconds") is not None:
        raise ValueError("max_run_seconds is forbidden; controller owns campaign budgeting")
    for key, expected in {"lr": 0.001, "weight_decay": 1e-4, "grad_clip": 1.0,
                          "epochs": 100, "patience": 10}.items():
        if config.get(key) != expected:
            raise ValueError(f"count-loss control must preserve V4 {key}={expected}")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("V4 model configuration is required")
    for key, expected in _FIXED_MODEL_VALUES.items():
        if model.get(key) != expected:
            raise ValueError(f"count-loss control must preserve V4 model.{key}={expected}")
    ref = config.get("mse_reference_config")
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("mse_reference_config must have exact path/sha256 fields")
    digest = ref.get("sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("MSE reference config SHA-256 is malformed")
    root = Path(run_root).resolve(strict=True)
    ref_path = _resolve_under_run(root, ref["path"], must_exist=True)
    if hashlib.sha256(ref_path.read_bytes()).hexdigest() != digest:
        raise ValueError("MSE reference config SHA-256 mismatch")
    dataset_id = "email" if dataset.name.casefold() == "email.npz" else "mooc"
    expected_ref = root / "configs" / "core-development-bound-v3" / f"{dataset_id}-V4-seed0.json"
    if ref_path != expected_ref.resolve(strict=False):
        raise ValueError("MSE reference config must be the canonical original V4 seed-0 config")
    reference = _read_unique_json(ref_path)
    if "training_loss" in reference or "mse_reference_config" in reference:
        raise ValueError("MSE reference must not contain adapter-only recipe/reference fields")
    output_raw = config.get("output_dir")
    reference_output_raw = reference.get("output_dir")
    if not isinstance(output_raw, str) or not output_raw or not isinstance(reference_output_raw, str):
        raise ValueError("count-loss and reference configs require output_dir")
    output_dir = _resolve_under_run(root, output_raw, must_exist=False)
    reference_output = _resolve_under_run(root, reference_output_raw, must_exist=False)
    if not output_dir.is_relative_to(root / "runs") or output_dir == reference_output:
        raise ValueError("count-loss output_dir must be distinct and contained under canonical RUN/runs")
    if config.get("variant") != reference.get("variant") or config.get("seed") != reference.get("seed"):
        raise ValueError("count-loss config identity differs from referenced MSE config")
    effective = {key: value for key, value in config.items()
                 if key not in {"output_dir", "training_loss", "mse_reference_config"}}
    reference_effective = {key: value for key, value in reference.items() if key != "output_dir"}
    if effective != reference_effective:
        raise ValueError("effective count-loss config differs from referenced MSE config")
    return {"path": ref_path, "sha256": digest, "config": reference}


def make_private_train_callable(*, engine_module=None, base_train_run=None, config=None, run_root=None):
    """Copy train_run with private loss/provenance/count-dataset guards."""
    engine_module = engine if engine_module is None else engine_module
    base_train_run = engine_module.train_run if base_train_run is None else base_train_run
    private_globals = dict(base_train_run.__globals__)
    original_source_manifest = engine_module.source_manifest
    original_build_model = engine_module.build_model
    reference = _validate_count_config(config, run_root or RUN_ROOT) if config is not None else None

    def augmented_source_manifest():
        result = dict(original_source_manifest())
        result[ADAPTER_SOURCE_KEY] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if reference is not None:
            if hashlib.sha256(reference["path"].read_bytes()).hexdigest() != reference["sha256"]:
                raise ValueError("MSE reference config changed while preparing training provenance")
            result["training-reference/" + reference["path"].relative_to(Path(run_root or RUN_ROOT).resolve()).as_posix()] = reference["sha256"]
        return result

    def guarded_build_model(config, ds, stats=None):
        _validate_count_config(config, run_root or RUN_ROOT)
        if getattr(ds, "count_data", None) is not True:
            raise ValueError("count-loss adapter requires a count_data dataset")
        return original_build_model(config, ds, stats)

    private_globals["masked_mse"] = masked_poisson_nll
    private_globals["source_manifest"] = augmented_source_manifest
    private_globals["build_model"] = guarded_build_model
    return types.FunctionType(base_train_run.__code__, private_globals,
                              name="train_count_run",
                              argdefs=base_train_run.__defaults__,
                              closure=base_train_run.__closure__)


def train_count_run(config: dict[str, Any]):
    _validate_count_config(config, RUN_ROOT)
    return make_private_train_callable(config=config)(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="immutable V4 seed-0 config plus training_loss recipe")
    args = parser.parse_args(argv)
    config = engine.load_config(args.config)
    result = train_count_run(config)
    print(str(result))


if __name__ == "__main__":
    main()
