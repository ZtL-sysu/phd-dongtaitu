import copy
import hashlib
import json
import math
from pathlib import Path
import types

import pytest
import torch

import count_loss_training_adapter_v1 as adapter
from edsgn import engine


def config(root, dataset_id="email", *, reference_changes=None, candidate_changes=None):
    dataset_path = root / "data" / "processed" / dataset_id / f"{dataset_id}.npz"
    reference = {
        "dataset_path": str(dataset_path),
        "variant": "V4",
        "seed": 0,
        "device": "cpu",
        "threads": 2,
        "batch_size": 8,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "epochs": 100,
        "patience": 10,
        "auxiliary_one_step_weight": 0.0,
        "model": {
            "d_total": 64,
            "k": 8,
            "candidate_cap": 32,
            "tau_fast": 1.0,
            "tau_slow": 8.0,
            "beta_init": 1.0,
            "lambda_temporal": 0.01,
            "training_prior_cache": True,
            "training_prior_cache_bytes": 1073741824,
            "training_candidate_tensor": True,
        },
        "output_dir": str(root / "runs" / "core-development-bound-v3" / dataset_id / "V4" / "seed0"),
    }
    candidate = copy.deepcopy(reference)
    for path, value in (reference_changes or {}).items():
        if path == "model.learn_tau":
            reference["model"]["learn_tau"] = value
        else:
            reference[path] = value
    reference_path = root / "configs" / "core-development-bound-v3" / f"{dataset_id}-V4-seed0.json"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps(reference, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    candidate["output_dir"] = str(root / "runs" / "count-loss-control-v1" / dataset_id / "V4" / "seed0")
    candidate["training_loss"] = copy.deepcopy(adapter.POISSON_RECIPE)
    candidate["mse_reference_config"] = {
        "path": reference_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }
    for path, value in (candidate_changes or {}).items():
        if path == "model.learn_tau":
            candidate["model"]["learn_tau"] = value
        elif path.startswith("model."):
            candidate["model"][path.split(".", 1)[1]] = value
        else:
            candidate[path] = value
    return candidate


def test_masked_poisson_matches_manual_scalar_and_reduces_valid_cells():
    pred = torch.tensor([[[2.0, 4.0], [99.0, 99.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 3.0], [0.0, 0.0]]])
    mask = torch.tensor([[[True, True], [False, False]]])
    expected = ((2.0 - math.log(2.0)) + (4.0 - 3.0 * math.log(4.0))) / 2.0
    loss = adapter.masked_poisson_nll(pred, target, mask)
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_masked_invalid_padding_is_ignored_but_valid_invalid_cells_reject():
    pred = torch.tensor([2.0, float("nan"), float("inf"), -4.0], requires_grad=True)
    target = torch.tensor([1.0, -3.0, float("nan"), float("inf")])
    mask = torch.tensor([True, False, False, False])
    loss = adapter.masked_poisson_nll(pred, target, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert pred.grad[1:].tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="valid Poisson targets"):
        adapter.masked_poisson_nll(torch.tensor([1.0]), torch.tensor([-1.0]), torch.tensor([True]))
    with pytest.raises(ValueError, match="valid predictions"):
        adapter.masked_poisson_nll(torch.tensor([float("nan")]), torch.tensor([1.0]), torch.tensor([True]))
    with pytest.raises(ValueError, match="valid predictions"):
        adapter.masked_poisson_nll(torch.tensor([-1.0]), torch.tensor([1.0]), torch.tensor([True]))
    with pytest.raises(ValueError, match="valid Poisson targets"):
        adapter.masked_poisson_nll(torch.tensor([1.0]), torch.tensor([float("inf")]), torch.tensor([True]))
    with pytest.raises(TypeError, match="floating dtypes"):
        adapter.masked_poisson_nll(torch.tensor([1], dtype=torch.int64), torch.tensor([1.0]), torch.tensor([True]))


def test_prediction_floor_is_finite_and_has_finite_gradient():
    pred = torch.tensor([0.0, 1e-10, 2.0], requires_grad=True)
    target = torch.tensor([0.0, 2.0, 3.0])
    mask = torch.ones(3, dtype=torch.bool)
    loss = adapter.masked_poisson_nll(pred, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:2].tolist() == [0.0, 0.0]
    assert pred.grad[2].item() == pytest.approx((1.0 - 3.0 / 2.0) / 3.0, rel=1e-6)


@pytest.mark.parametrize("change,match", [
    ({"variant": "V3"}, "variant must be V4"),
    ({"seed": 1}, "seed must be exactly 0"),
    ({"dataset_path": r"C:\run\data\processed\synthetic\synthetic.npz"}, "original email or MOOC"),
    ({"dataset_path": r"C:\run\data\processed\email-episode\email-episode.npz"}, "original email or MOOC"),
    ({"auxiliary_one_step_weight": 0.1}, "auxiliary_one_step_weight"),
    ({"resume": True}, "resume"),
    ({"smoke_train_windows": 4}, "smoke"),
    ({"max_run_seconds": 10}, "max_run_seconds"),
    ({"training_loss": {"kind": "masked_mse"}}, "training_loss recipe"),
])
def test_config_gate_rejects_wrong_scope_or_loss_recipe(tmp_path, change, match):
    value = config(tmp_path, candidate_changes=change)
    with pytest.raises(ValueError, match=match):
        adapter._validate_count_config(value, tmp_path)


@pytest.mark.parametrize("field,value", [("lr", 0.002), ("model.learn_tau", False), ("model.extra", 3)])
def test_effective_config_must_match_mse_reference_exactly(tmp_path, field, value):
    changed_reference = config(tmp_path, reference_changes={field: value})
    # The reference remains byte-valid, but its non-output training fields no
    # longer match the count-loss candidate.
    with pytest.raises(ValueError, match="effective count-loss config differs"):
        adapter._validate_count_config(changed_reference, tmp_path)


@pytest.mark.parametrize("dataset_id", ["email", "mooc"])
def test_exact_original_mse_reference_passes_and_is_returned(dataset_id, tmp_path):
    value = config(tmp_path, dataset_id)
    checked = adapter._validate_count_config(value, tmp_path)
    assert checked["path"].name == f"{dataset_id}-V4-seed0.json"
    assert checked["sha256"] == value["mse_reference_config"]["sha256"]
    assert checked["config"]["output_dir"] != value["output_dir"]


@pytest.mark.parametrize("change,match", [
    ({"training_loss": {"kind": "masked_poisson_nll", "log_input": 0, "full": False,
                        "reduction": "none", "prediction_floor": 1e-8,
                        "normalization": "valid_target_cells"}}, "training_loss recipe"),
    ({"training_loss": {"kind": "masked_poisson_nll", "log_input": False, "full": False,
                        "reduction": "none", "prediction_floor": 1,
                        "normalization": "valid_target_cells"}}, "training_loss recipe"),
])
def test_recipe_types_are_exact_not_boolean_integer_aliases(tmp_path, change, match):
    value = config(tmp_path, candidate_changes=change)
    with pytest.raises(ValueError, match=match):
        adapter._validate_count_config(value, tmp_path)


def test_reference_tamper_and_noncanonical_output_reject(tmp_path):
    value = config(tmp_path)
    reference = tmp_path / value["mse_reference_config"]["path"]
    reference.write_text(reference.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        adapter._validate_count_config(value, tmp_path)
    second = tmp_path / "second"
    value = config(second)
    value["output_dir"] = str(tmp_path / "outside")
    with pytest.raises(ValueError, match="outside canonical RUN"):
        adapter._validate_count_config(value, second)


def test_count_builder_rejects_noncount_dataset_before_underlying_factory(tmp_path):
    calls = []
    fake_engine = types.SimpleNamespace(
        build_model=lambda *args: calls.append(args) or "model",
        source_manifest=lambda: {},
    )
    value = config(tmp_path)
    wrapped = adapter.make_private_train_callable(engine_module=fake_engine, base_train_run=lambda c: c,
                                                  config=value, run_root=tmp_path)
    guarded = wrapped.__globals__["build_model"]
    with pytest.raises(ValueError, match="count_data dataset"):
        guarded(value, types.SimpleNamespace(count_data=False), None)
    assert calls == []
    assert guarded(value, types.SimpleNamespace(count_data=True), None) == "model"
    assert len(calls) == 1


def test_private_function_globals_do_not_mutate_engine_and_bind_adapter_and_reference(tmp_path):
    root = tmp_path
    value = config(root)
    old_loss, old_manifest, old_builder = engine.masked_mse, engine.source_manifest, engine.build_model
    private = adapter.make_private_train_callable(config=value, run_root=root)
    assert engine.masked_mse is old_loss
    assert engine.source_manifest is old_manifest
    assert engine.build_model is old_builder
    assert private.__globals__ is not engine.train_run.__globals__
    assert private.__globals__["masked_mse"] is adapter.masked_poisson_nll
    manifest = private.__globals__["source_manifest"]()
    expected = hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()
    assert manifest[adapter.ADAPTER_SOURCE_KEY] == expected
    ref_key = "training-reference/" + value["mse_reference_config"]["path"]
    assert manifest[ref_key] == value["mse_reference_config"]["sha256"]
    assert engine.source_manifest is old_manifest


def test_stub_train_function_observes_poisson_loss_and_forwarded_config(tmp_path):
    value = config(tmp_path)
    namespace = {"torch": torch, "ds": types.SimpleNamespace(count_data=True)}
    exec(
        "def stub_train(config):\n"
        "    pred = torch.tensor([2.0])\n"
        "    target = torch.tensor([1.0])\n"
        "    mask = torch.tensor([True])\n"
        "    loss = masked_mse(pred, target, mask)\n"
        "    model = build_model(config, ds, None)\n"
        "    return {'loss': loss, 'manifest': source_manifest(), 'model': model, 'config': config}\n",
        namespace, namespace,
    )
    stub = namespace["stub_train"]
    original = types.SimpleNamespace(
        build_model=lambda c, ds, stats: (c["variant"], ds.count_data),
        source_manifest=lambda: {"engine.py": "source-hash"},
    )
    private = adapter.make_private_train_callable(engine_module=original, base_train_run=stub,
                                                  config=value, run_root=tmp_path)
    result = private(value)
    assert result["loss"].item() == pytest.approx(2.0 - math.log(2.0), rel=1e-6)
    assert result["manifest"]["engine.py"] == "source-hash"
    assert result["manifest"][adapter.ADAPTER_SOURCE_KEY] == hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()
    assert result["manifest"]["training-reference/" + value["mse_reference_config"]["path"]] == value["mse_reference_config"]["sha256"]
    assert result["model"] == ("V4", True)
    assert result["config"] is value
