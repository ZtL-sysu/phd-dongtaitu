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
    """Author DCRNN seq2seq, with an explicit fixed training-derived prior."""

    def __init__(self, *, fixed_training_prior, **kwargs):
        super().__init__(
            n_nodes=kwargs["n_nodes"], n_features=kwargs["n_features"],
            history=kwargs["history"], hmax=kwargs["hmax"],
            count_data=kwargs.get("count_data", True), norm_mean=kwargs.get("norm_mean"),
            norm_std=kwargs.get("norm_std"), output_scale=kwargs.get("output_scale"),
            output_mean=kwargs.get("output_mean"),
        )
        if not self.count_data:
            raise ValueError("DCRNNForecast adapter currently requires count targets")
        prior = np.asarray(fixed_training_prior, dtype=np.float32)
        if prior.shape != (self.n_nodes, self.n_nodes) or not np.isfinite(prior).all():
            raise ValueError("fixed_training_prior must be a finite [N,N] matrix")
        if np.any(prior < 0):
            raise ValueError("dual random walk requires nonnegative prior weights")
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
        self.network = author.DCRNNModel(prior, logging.getLogger("dcrnn-staging"), **options)
        self._author_cell_module = cell
        self._author_model_module = author
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
        return self._result(predictions, counters={
            "readout_calls": self.hmax, "graph_calls": self.history + self.hmax,
            "state_calls": self.history + self.hmax,
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
    """Small CPU-only architecture, gradient, state and device regression checks."""
    torch.set_num_threads(2)
    source_before = _source_hashes()
    cases = []
    for seed, (history, horizon, features) in enumerate(((24, 24, 1), (28, 7, 2)), 11):
        torch.manual_seed(seed)
        n = 5
        prior = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            prior[i, (i + 1) % n] = 1.0
            prior[i, (i - 1) % n] = 0.5
        model = DCRNNForecast(
            fixed_training_prior=prior, n_nodes=n, n_features=features,
            history=history, hmax=horizon, count_data=True,
            output_scale=torch.ones(features),
        )
        model.to(torch.device("cpu"))
        device_ok = all(s.device.type == "cpu" for m in model.modules()
                        if m.__class__.__name__ == "DCGRUCell" for s in m._supports)
        parameter_count = sum(p.numel() for p in model.parameters())
        x = torch.rand(1, history, n, features)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 2, 1, :] = False
        model.train()
        out = model(x, mask=mask)["predictions"]
        loss = out.mean()
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        grad_ok = bool(grads) and all(g is not None and torch.isfinite(g).all().item() for g in grads)
        prediction_ok = tuple(out.shape) == (1, horizon, n, features) and torch.isfinite(out).all().item()
        after_forward_count = sum(p.numel() for p in model.parameters())
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        restored = DCRNNForecast(
            fixed_training_prior=prior, n_nodes=n, n_features=features,
            history=history, hmax=horizon, count_data=True,
            output_scale=torch.ones(features),
        ).to("cpu")
        restored.load_state_dict(state)
        model.eval(); restored.eval()
        with torch.no_grad():
            pred_a = model(x, mask=mask)["predictions"]
            pred_b = restored(x, mask=mask)["predictions"]
        exact_reload = torch.allclose(pred_a, pred_b, rtol=0, atol=1e-7)
        cases.append({
            "N": n, "L": history, "H": horizon, "F": features,
            "prediction_shape": list(out.shape), "finite_prediction": bool(prediction_ok),
            "finite_gradients_all_parameters": grad_ok,
            "parameters_materialized": model.materialized_parameter_count,
            "parameters_after_forward": after_forward_count,
            "parameter_count_stable": parameter_count == after_forward_count,
            "supports_cpu_after_apply": bool(device_ok),
            "state_reload_prediction_equal_atol_1e-7": bool(exact_reload),
        })
    source_after = _source_hashes()
    return {
        "check_type": "CPU toy forward/backward/state smoke only; no optimizer or dataset",
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "source_hashes_unchanged": source_before == source_after,
        "cases": cases,
        "all_checks_pass": source_before == source_after and all(
            c["finite_prediction"] and c["finite_gradients_all_parameters"]
            and c["parameter_count_stable"] and c["supports_cpu_after_apply"]
            and c["state_reload_prediction_equal_atol_1e-7"] for c in cases
        ),
    }


if __name__ == "__main__":
    print(json.dumps(smoke(), indent=2, sort_keys=True))
