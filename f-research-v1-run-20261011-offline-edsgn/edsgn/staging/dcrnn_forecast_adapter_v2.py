"""Unregistered CPU-first adapter around the pinned author DCRNN encoder-decoder.

This staging module is deliberately not imported by the package factory.  It keeps
the author recurrence and graph convolution intact; only source namespace and
device handling are adapted for this offline forecast interface.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "edsgn" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from edsgn.baselines import _ForecastBaseline  # noqa: E402


def _load_exact_random_walk(source: Path):
    """Compile the unchanged author AST function without importing TF-heavy utils."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "calculate_random_walk_matrix"
    )
    module = types.ModuleType("_edsgn_dcrnn_isolated_utils")
    module.__dict__.update(np=np, sp=sp)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), module.__dict__)
    return module


def _load_author_modules():
    cached = sys.modules.get("_edsgn_dcrnn_author")
    if cached is not None:
        return cached.cell, cached.model
    source = _ROOT / "sources" / "DCRNN_PyTorch"
    utils = _load_exact_random_walk(source / "lib" / "utils.py")

    # The author modules use `from lib import utils` and `model.pytorch.*`.
    # Provide only these isolated namespace aliases, never importing trainer/data.
    lib_pkg = types.ModuleType("lib")
    lib_pkg.__path__ = [str(source / "lib")]
    lib_pkg.utils = utils
    lib_pkg.__package__ = "lib"
    model_pkg = types.ModuleType("model")
    model_pkg.__path__ = [str(source / "model")]
    model_pkg.__package__ = "model"
    pytorch_pkg = types.ModuleType("model.pytorch")
    pytorch_pkg.__path__ = [str(source / "model" / "pytorch")]
    pytorch_pkg.__package__ = "model.pytorch"
    previous = {k: sys.modules.get(k) for k in ("lib", "model", "model.pytorch")}
    # Refuse to shadow an unrelated in-process package.
    for key, expected in (("lib", source / "lib"), ("model", source / "model")):
        old = previous[key]
        if old is not None:
            paths = [Path(p).resolve() for p in getattr(old, "__path__", [])]
            if expected.resolve() not in paths:
                raise ImportError(f"Refusing to shadow unrelated {key!r} package")
    sys.modules.update({"lib": lib_pkg, "model": model_pkg, "model.pytorch": pytorch_pkg})
    try:
        cell_spec = importlib.util.spec_from_file_location(
            "model.pytorch.dcrnn_cell", source / "model" / "pytorch" / "dcrnn_cell.py"
        )
        cell = importlib.util.module_from_spec(cell_spec)
        sys.modules[cell_spec.name] = cell
        cell_spec.loader.exec_module(cell)
        model_spec = importlib.util.spec_from_file_location(
            "_edsgn_dcrnn_author_model", source / "model" / "pytorch" / "dcrnn_model.py"
        )
        model = importlib.util.module_from_spec(model_spec)
        sys.modules[model_spec.name] = model
        model_spec.loader.exec_module(model)
    finally:
        for key, old in previous.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old
        # Class objects retain the isolated utility and author module references.
        sys.modules.pop("model.pytorch.dcrnn_cell", None)
        sys.modules.pop("_edsgn_dcrnn_author_model", None)
    holder = types.SimpleNamespace(cell=cell, model=model)
    sys.modules["_edsgn_dcrnn_author"] = holder
    return cell, model


class DCRNNForecast(_ForecastBaseline):
    """Author DCRNN seq2seq with a checkpoint-persistent training graph prior."""

    def __init__(self, *, fixed_training_prior, **kwargs):
        super().__init__(
            n_nodes=kwargs["n_nodes"], n_features=kwargs["n_features"],
            history=kwargs["history"], hmax=kwargs["hmax"],
            count_data=kwargs.get("count_data", True), norm_mean=kwargs.get("norm_mean"),
            norm_std=kwargs.get("norm_std"), output_scale=kwargs.get("output_scale"),
            output_mean=kwargs.get("output_mean"),
        )
        prior = np.asarray(fixed_training_prior, dtype=np.float32)
        if prior.shape != (self.n_nodes, self.n_nodes) or not np.isfinite(prior).all():
            raise ValueError("fixed_training_prior must be a finite [N,N] matrix")
        if np.any(prior < 0):
            raise ValueError("dual random walk requires nonnegative prior weights")
        self.register_buffer("fixed_training_prior", torch.from_numpy(prior.copy()), persistent=True)
        cell, author = _load_author_modules()
        # Author's global allocation device auto-selects CUDA at import; staging
        # initializes safely on CPU and later updates these isolated globals in _apply.
        cell.device = torch.device("cpu")
        author.device = torch.device("cpu")
        options = dict(
            input_dim=self.n_features, output_dim=self.n_features,
            seq_len=self.history, horizon=self.hmax, num_nodes=self.n_nodes,
            rnn_units=int(kwargs.get("rnn_units", 64)),
            num_rnn_layers=int(kwargs.get("num_rnn_layers", 2)),
            max_diffusion_step=int(kwargs.get("max_diffusion_step", 2)),
            filter_type="dual_random_walk", cl_decay_steps=2000,
            use_curriculum_learning=False,
        )
        # Author defaults are dual_random_walk, diffusion=2, units=64, layers=2;
        # curriculum stays disabled, so labels are never available/used.
        self.network = author.DCRNNModel(
            self.fixed_training_prior.detach().cpu().numpy(),
            logging.getLogger("dcrnn-staging"), **options
        )
        self._author_cell_module = cell
        self._author_model_module = author
        # The source keeps supports as plain attributes; rebuild from independent
        # copies to ensure each direction is derived from the frozen graph itself.
        self._rebuild_author_supports()
        # Materialize all LayerParams before callers can count/save optimizer state.
        with torch.no_grad():
            dummy = torch.zeros(self.history, 1, self.n_nodes * self.n_features)
            was_training = self.network.training
            self.network.eval()
            self.network(dummy, labels=None, batches_seen=None)
            after_materialization = sum(p.numel() for p in self.parameters())
            self.network(dummy, labels=None, batches_seen=None)
            self.network.train(was_training)
        after_second_forward = sum(p.numel() for p in self.parameters())
        if after_materialization != after_second_forward:
            raise RuntimeError("DCRNN created new parameters after materialization")
        self.materialized_parameter_count = after_materialization

    def _apply(self, fn):
        result = super()._apply(fn)
        params = list(self.parameters())
        device = params[0].device if params else torch.device("cpu")
        self._author_cell_module.device = device
        self._author_model_module.device = device
        for module in self.modules():
            if module.__class__.__name__ == "DCGRUCell" and hasattr(module, "_supports"):
                module._supports = [fn(support) for support in module._supports]
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # The persistent buffer may replace the constructor graph. Rebuild the
        # author cells' plain-list supports from the loaded checkpoint graph.
        try:
            result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:  # compatibility with torch versions predating assign
            result = super().load_state_dict(state_dict, strict=strict)
        self._rebuild_author_supports()
        return result

    def _rebuild_author_supports(self):
        prior = self.fixed_training_prior.detach().cpu().numpy()
        random_walk = self._author_cell_module.utils.calculate_random_walk_matrix
        forward_support = random_walk(prior.copy()).T
        reverse_support = random_walk(prior.T.copy()).T
        supports = [forward_support, reverse_support]
        device = self.fixed_training_prior.device
        for module in self.modules():
            if module.__class__.__name__ == "DCGRUCell" and hasattr(module, "_supports"):
                module._supports = [module._build_sparse_matrix(s).to(device) for s in supports]

    def forward(self, x, mask=None, age=None, raw=None, policy=None, event_priors=None,
                return_trace=False):
        del age, raw, policy, event_priors, return_trace
        if x.ndim != 4 or tuple(x.shape[1:]) != (
            self.history, self.n_nodes, self.n_features
        ):
            raise ValueError("x must have shape [B,history,nodes,features]")
        if mask is not None:
            visible = torch.as_tensor(mask, device=x.device).bool()
            if visible.ndim == 3:
                visible = visible.unsqueeze(-1).expand_as(x)
            if visible.shape != x.shape:
                raise ValueError("mask must be [B,L,N] or [B,L,N,F]")
            x = x * visible.to(x.dtype)
        b = x.shape[0]
        author_input = x.permute(1, 0, 2, 3).reshape(
            self.history, b, self.n_nodes * self.n_features
        )
        logits = self.network(author_input, labels=None, batches_seen=None)
        logits = logits.reshape(self.hmax, b, self.n_nodes, self.n_features).permute(1, 0, 2, 3)
        predictions = self._raw_forecast(logits)
        recurrent_cell_calls = (self.history + self.hmax) * int(self.network.num_rnn_layers)
        diffusion_conv_calls = 2 * recurrent_cell_calls  # gate convolution + candidate convolution
        support_order_applications = (
            diffusion_conv_calls * 2 * int(self.network.max_diffusion_step)
        )
        return self._result(predictions, counters={
            "recurrent_cell_calls": recurrent_cell_calls,
            "diffusion_conv_calls": diffusion_conv_calls,
            "diffusion_support_order_applications": support_order_applications,
            "readout_calls": self.hmax,
        })


def _source_hashes():
    root = _ROOT / "sources" / "DCRNN_PyTorch"
    selected = [
        root / "model" / "pytorch" / "dcrnn_cell.py",
        root / "model" / "pytorch" / "dcrnn_model.py",
        root / "lib" / "utils.py", root / "LICENSE", root / "README.md",
        root / "data" / "model" / "dcrnn_la.yaml",
    ]
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in selected}


def smoke():
    """CPU diagnostic checks for signed/count targets and persistent graph state."""
    torch.set_num_threads(2)
    source_before = _source_hashes()
    cases = []
    for seed, (history, horizon, features, count_data) in enumerate((
        (24, 7, 1, True), (24, 7, 2, True),
        (24, 7, 1, False), (24, 7, 2, False),
    ), 11):
        torch.manual_seed(seed)
        n = 5
        prior = np.zeros((n, n), dtype=np.float32)
        prior[0, 1] = 1.0
        prior[1, 2] = 2.0
        prior[2, 3] = 0.5
        prior[3, 0] = 3.0
        prior[4, 1] = 0.25
        model = DCRNNForecast(
            fixed_training_prior=prior, n_nodes=n, n_features=features,
            history=history, hmax=horizon, count_data=count_data,
            output_scale=torch.ones(features),
        )
        model.to(torch.device("cpu"))
        device_ok = all(s.device.type == "cpu" for m in model.modules()
                        if m.__class__.__name__ == "DCGRUCell" for s in m._supports)
        cell_modules = [m for m in model.modules() if m.__class__.__name__ == "DCGRUCell"]
        # Compare after the author cell's unchanged sparse-conversion routine,
        # which defines the actual tensors consumed by its diffusion operator.
        build_sparse = model._author_cell_module.DCGRUCell._build_sparse_matrix
        expected_forward = build_sparse(
            model._author_cell_module.utils.calculate_random_walk_matrix(prior.copy()).T
        ).to_dense().numpy()
        expected_reverse = build_sparse(
            model._author_cell_module.utils.calculate_random_walk_matrix(prior.T.copy()).T
        ).to_dense().numpy()
        supports_ok = all(
            len(m._supports) == 2
            and torch.allclose(m._supports[0].to_dense(), torch.as_tensor(expected_forward, dtype=torch.float32))
            and torch.allclose(m._supports[1].to_dense(), torch.as_tensor(expected_reverse, dtype=torch.float32))
            for m in cell_modules
        )
        asymmetry_detected = not np.allclose(expected_forward, expected_reverse)
        parameter_count = sum(p.numel() for p in model.parameters())
        x = torch.rand(2, history, n, features)
        if not count_data:
            x = x * 2.0 - 1.0
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 2, 1, :] = False
        model.train()
        recurrent_observed = [0]
        diffusion_observed = [0]
        readout_observed = [0]
        hooks = [m.register_forward_hook(lambda *_args, c=recurrent_observed: c.__setitem__(0, c[0] + 1))
                 for m in cell_modules]
        hooks.append(model.network.decoder_model.projection_layer.register_forward_hook(
            lambda *_args, c=readout_observed: c.__setitem__(0, c[0] + 1)
        ))
        original_methods = []
        for cell in cell_modules:
            original = cell._gconv
            original_methods.append((cell, original))
            def counted_gconv(*args, _original=original, _counter=diffusion_observed, **kwargs):
                _counter[0] += 1
                return _original(*args, **kwargs)
            cell._gconv = counted_gconv
        out_result = model(x, mask=mask)
        out = out_result["predictions"]
        for hook in hooks:
            hook.remove()
        for cell, original in original_methods:
            cell._gconv = original
        loss = out.mean()
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        grad_ok = bool(grads) and all(g is not None and torch.isfinite(g).all().item() for g in grads)
        prediction_ok = tuple(out.shape) == (2, horizon, n, features) and torch.isfinite(out).all().item()
        after_forward_count = sum(p.numel() for p in model.parameters())
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        graph_serialized = "fixed_training_prior" in state and torch.equal(state["fixed_training_prior"], torch.as_tensor(prior))
        other_prior = np.roll(prior, 1, axis=1)
        restored = DCRNNForecast(
            fixed_training_prior=other_prior, n_nodes=n, n_features=features,
            history=history, hmax=horizon, count_data=count_data,
            output_scale=torch.ones(features),
        ).to("cpu")
        restored.load_state_dict(state)
        model.eval(); restored.eval()
        with torch.no_grad():
            pred_a = model(x, mask=mask)["predictions"]
            pred_b = restored(x, mask=mask)["predictions"]
        reload_equal = torch.allclose(pred_a, pred_b, rtol=0, atol=1e-7)
        checkpoint_graph = torch.equal(restored.fixed_training_prior, torch.as_tensor(prior))
        supports_after_load = all(
            torch.allclose(m._supports[0].to_dense(), torch.as_tensor(expected_forward, dtype=torch.float32))
            and torch.allclose(m._supports[1].to_dense(), torch.as_tensor(expected_reverse, dtype=torch.float32))
            for m in [x for x in restored.modules() if x.__class__.__name__ == "DCGRUCell"]
        )
        # Same seed and same inputs/prior must initialize and predict identically.
        torch.manual_seed(seed)
        repeated = DCRNNForecast(
            fixed_training_prior=prior, n_nodes=n, n_features=features,
            history=history, hmax=horizon, count_data=count_data,
            output_scale=torch.ones(features),
        ).eval()
        with torch.no_grad():
            repeat_pred = repeated(x, mask=mask)["predictions"]
        repeatable = torch.allclose(pred_a, repeat_pred, rtol=0, atol=1e-7)
        counters = out_result["counters"]
        cases.append({
            "N": n, "B": 2, "L": history, "H": horizon, "F": features,
            "count_data": count_data,
            "prediction_shape": list(out.shape), "finite_prediction": bool(prediction_ok),
            "finite_gradients_all_parameters": grad_ok,
            "parameters_materialized": model.materialized_parameter_count,
            "parameters_after_forward": after_forward_count,
            "parameter_count_stable": parameter_count == after_forward_count,
            "supports_cpu_after_apply": bool(device_ok),
            "both_asymmetric_random_walk_supports_match_author_math": bool(supports_ok and asymmetry_detected),
            "diffusion_order_2_executed": counters["diffusion_conv_calls"] > 0,
            "counter_recurrent_cells_matches_hooks": counters["recurrent_cell_calls"] == recurrent_observed[0],
            "counter_diffusion_convs_matches_hooks": counters["diffusion_conv_calls"] == diffusion_observed[0],
            "counter_readouts_matches_hooks": counters["readout_calls"] == readout_observed[0],
            "counter_values": counters,
            "training_prior_in_state_dict": bool(graph_serialized),
            "loaded_constructor_with_different_graph_uses_checkpoint_graph": bool(checkpoint_graph),
            "loaded_supports_rebuilt_from_checkpoint_graph": bool(supports_after_load),
            "state_reload_prediction_equal_atol_1e-7": bool(reload_equal),
            "seed_repeatable_atol_1e-7": bool(repeatable),
        })
    source_after = _source_hashes()
    return {
        "check_type": "CPU functionality/resource diagnostic only; no data, optimizer, GPU, or metrics",
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "source_hashes_unchanged": source_before == source_after,
        "cases": cases,
        "all_checks_pass": source_before == source_after and all(
            c["finite_prediction"] and c["finite_gradients_all_parameters"]
            and c["parameter_count_stable"] and c["supports_cpu_after_apply"]
            and c["both_asymmetric_random_walk_supports_match_author_math"]
            and c["diffusion_order_2_executed"]
            and c["counter_recurrent_cells_matches_hooks"]
            and c["counter_diffusion_convs_matches_hooks"]
            and c["counter_readouts_matches_hooks"]
            and c["training_prior_in_state_dict"]
            and c["loaded_constructor_with_different_graph_uses_checkpoint_graph"]
            and c["loaded_supports_rebuilt_from_checkpoint_graph"]
            and c["state_reload_prediction_equal_atol_1e-7"] and c["seed_repeatable_atol_1e-7"]
            for c in cases
        ),
    }


if __name__ == "__main__":
    print(json.dumps(smoke(), indent=2, sort_keys=True))
