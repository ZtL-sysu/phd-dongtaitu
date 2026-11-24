"""Runtime adapter definitions copied verbatim from the audited staging source."""

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

_ROOT = Path(__file__).resolve().parents[3]

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
    """DCRNN-adapted: author architecture with sparse index/value correction."""

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
        """Build mathematically aligned dual random walks from the checkpoint graph.

        Upstream `_build_sparse_matrix` sorts COO coordinates without applying
        that permutation to values. This compatibility correction keeps each
        scipy COO coordinate paired with its value while leaving the author
        recurrent cells, diffusion recurrence, parameters, and graph formula intact.
        """
        prior = self.fixed_training_prior.detach().cpu().numpy()
        random_walk = self._author_cell_module.utils.calculate_random_walk_matrix
        supports = [random_walk(prior.copy()).T, random_walk(prior.T.copy()).T]
        device = self.fixed_training_prior.device
        for module in self.modules():
            if module.__class__.__name__ == "DCGRUCell" and hasattr(module, "_supports"):
                module._supports = [self._paired_coo_tensor(s, device) for s in supports]

    @staticmethod
    def _paired_coo_tensor(matrix, device):
        coo = matrix.tocoo(copy=True)
        indices = torch.as_tensor(
            np.vstack((coo.row, coo.col)), dtype=torch.long, device=device
        )
        values = torch.as_tensor(coo.data.copy(), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()

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


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _node_digest(node_ids):
    return hashlib.sha256(np.asarray(node_ids).tobytes(order="C")).hexdigest()


def _root_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def _dataset_sources_match(manifest, expected_path):
    """Check the current prior-manifest schema, which stores `sources` (not dataset_path)."""
    expected = _root_path(expected_path)
    declared = manifest.get("dataset_path")
    sources = manifest.get("sources", [])
    source_paths = [_root_path(row["path"]) for row in sources if row.get("path")]
    if declared:
        if _root_path(declared) != expected:
            raise ValueError("fixed-prior manifest dataset path differs from expected_dataset_path")
    elif expected.is_file():
        if source_paths != [expected]:
            raise ValueError("fixed-prior manifest source path does not match expected dataset bundle")
    elif expected.is_dir():
        expected_train = sorted(p.resolve() for p in expected.glob("train_seed*.npz") if p.is_file())
        if not expected_train or sorted(source_paths) != expected_train:
            raise ValueError("synthetic fixed-prior manifest sources do not exactly match the train collection")
        for row in sources:
            if row.get("partition") != "train":
                raise ValueError("synthetic fixed-prior manifest includes a non-train source")
    else:
        raise ValueError("expected_dataset_path does not exist")
    for row in sources:
        path = _root_path(row["path"])
        if not path.is_file() or _sha256_file(path) != row.get("sha256"):
            raise ValueError(f"fixed-prior source bundle hash mismatch: {path}")


def load_bound_training_prior(*, prior_path, prior_sha256, manifest_path,
                              manifest_sha256, expected_node_ids,
                              expected_dataset_path, expected_n_nodes):
    """Load only the frozen graph; reject any hash, dataset or node-order drift."""
    if not all((prior_path, prior_sha256, manifest_path, manifest_sha256,
                expected_node_ids is not None, expected_dataset_path)):
        raise ValueError("DCRNN requires bound prior paths/hashes, expected nodes, and dataset path")
    prior_path, manifest_path = _root_path(prior_path), _root_path(manifest_path)
    if _sha256_file(prior_path) != str(prior_sha256):
        raise ValueError("fixed-training-prior file SHA-256 mismatch")
    if _sha256_file(manifest_path) != str(manifest_sha256):
        raise ValueError("fixed-training-prior manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _dataset_sources_match(manifest, expected_dataset_path)
    with np.load(prior_path, allow_pickle=False) as z:
        if not {"fixed_training_prior", "node_ids"}.issubset(z.files):
            raise ValueError("fixed-prior NPZ lacks graph or node_ids")
        matrix = np.asarray(z["fixed_training_prior"], dtype=np.float32).copy()
        node_ids = np.asarray(z["node_ids"]).copy()
    expected = np.asarray(expected_node_ids)
    if matrix.shape != (int(expected_n_nodes), int(expected_n_nodes)):
        raise ValueError("fixed-prior graph shape does not match model node count")
    if node_ids.shape != expected.shape or not np.array_equal(node_ids, expected):
        raise ValueError("fixed-prior node ordering differs from PreparedData")
    if manifest.get("node_count") != int(expected_n_nodes) or manifest.get("node_ids_sha256") != _node_digest(node_ids):
        raise ValueError("fixed-prior node count/order digest differs from its manifest")
    array_hash = hashlib.sha256(np.asarray(matrix, dtype="<f4", order="C").tobytes(order="C")).hexdigest()
    if array_hash != manifest.get("fixed_training_prior_array_sha256"):
        raise ValueError("fixed-prior array SHA-256 differs from its manifest")
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(np.diag(matrix) != 0):
        raise ValueError("fixed-prior matrix must be finite, nonnegative and zero-diagonal")
    return matrix
