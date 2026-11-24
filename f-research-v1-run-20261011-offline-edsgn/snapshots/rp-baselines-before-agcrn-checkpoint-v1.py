"""Runtime adapter definitions copied verbatim from the audited staging source."""

from __future__ import annotations

import importlib.util

import importlib.metadata

import hashlib

import io

import sys

import time

import types

from pathlib import Path

from types import SimpleNamespace

import torch

from torch import nn

from torch.nn import functional as F

_RUN_ROOT = Path(__file__).resolve().parents[3]

_SOURCE_ROOT = _RUN_ROOT / "sources"

_SRC = _RUN_ROOT / "edsgn" / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from edsgn.baselines import _ForecastBaseline  # noqa: E402

def _load_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load staged author source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _load_gwnet():
    name = "_edsgn_rp_graph_wavenet_model"
    if name not in sys.modules:
        _load_file(name, _SOURCE_ROOT / "Graph-WaveNet" / "model.py")
    return sys.modules[name].gwnet

def _gwnet_gate_conv2d_compat(module, value):
    """Apply the source's declared ``(1,k)`` gate kernel to its 4-D tensor.

    The pinned file constructs ``nn.Conv1d(kernel_size=(1,k))`` and then calls
    it on ``[B,C,N,T]``. Current PyTorch rejects that rank/weight combination.
    The stored weight is already ``[out,in,1,k]``; using it unchanged in the
    semantically matching Conv2d operation preserves channels, kernel, stride,
    dilation, bias, and parameter/state-dict layout.
    """
    kernel = tuple(module.kernel_size)
    if len(kernel) != 2 or kernel[0] != 1 or module.weight.ndim != 4:
        raise RuntimeError("Pinned Graph WaveNet gate is not the expected 1×k temporal kernel")
    return F.conv2d(
        value,
        module.weight,
        module.bias,
        stride=(1, int(module.stride[-1])),
        padding=(0, int(module.padding[-1])),
        dilation=(1, int(module.dilation[-1])),
        groups=module.groups,
    )

def _load_agcrn():
    """Resolve upstream absolute ``model.*`` imports in an isolated namespace."""
    name = "_edsgn_rp_agcrn_model"
    if name in sys.modules:
        return sys.modules[name].AGCRN
    source_model = _SOURCE_ROOT / "AGCRN" / "model"
    previous_model = sys.modules.get("model")
    if previous_model is not None:
        paths = [Path(p).resolve() for p in getattr(previous_model, "__path__", [])]
        if source_model.resolve() not in paths:
            raise ImportError(
                "Cannot safely stage AGCRN: a different top-level 'model' module is already loaded"
            )
        namespace = previous_model
    else:
        namespace = types.ModuleType("model")
        namespace.__path__ = [str(source_model)]
        namespace.__package__ = "model"
        sys.modules["model"] = namespace
    try:
        module = _load_file(name, source_model / "AGCRN.py")
    finally:
        # Source modules retain class references; remove only temporary import aliases.
        if previous_model is None:
            sys.modules.pop("model", None)
            sys.modules.pop("model.AGCRNCell", None)
            sys.modules.pop("model.AGCN", None)
    return module.AGCRN

def _load_staeformer():
    """Load author model using the real declared torchinfo dependency."""
    name = "_edsgn_rp_staeformer_model"
    if name in sys.modules:
        return sys.modules[name].STAEformer
    __import__("torchinfo")
    module = _load_file(name, _SOURCE_ROOT / "STAEformer-pinned" / "model" / "STAEformer.py")
    return module.STAEformer

def _valid_input(x, mask, history, nodes, features):
    if x.ndim != 4 or tuple(x.shape[1:]) != (history, nodes, features):
        raise ValueError(
            f"Expected [B,{history},{nodes},{features}], received {tuple(x.shape)}"
        )
    if mask is None:
        return x
    visible = torch.as_tensor(mask, device=x.device).bool()
    if visible.ndim == 3:
        visible = visible.unsqueeze(-1).expand_as(x)
    if visible.shape != x.shape:
        raise ValueError("mask must be [B,L,N] or [B,L,N,F]")
    return x * visible.to(dtype=x.dtype)

def _setup_defaults(kwargs):
    """Extract the shared task/scale contract without changing its dimensions."""
    return dict(
        n_nodes=kwargs["n_nodes"],
        n_features=kwargs["n_features"],
        history=kwargs["history"],
        hmax=kwargs["hmax"],
        count_data=kwargs.get("count_data", True),
        norm_mean=kwargs.get("norm_mean"),
        norm_std=kwargs.get("norm_std"),
        output_scale=kwargs.get("output_scale"),
        output_mean=kwargs.get("output_mean"),
    )

def _init_agcrn_author(network):
    """Exact pinned Run.py:102-106 loop, executed before any training."""
    for parameter in network.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
        else:
            nn.init.uniform_(parameter)

class GraphWaveNetForecast(_ForecastBaseline):
    """Graph WaveNet author architecture, learned adjacency, direct H×F channels."""

    def __init__(self, **kwargs):
        super().__init__(**_setup_defaults(kwargs))
        official = _load_gwnet()
        self.network = official(
            device=torch.device("cpu"),
            num_nodes=self.n_nodes,
            dropout=0.3,
            supports=None,
            gcn_bool=True,
            addaptadj=True,
            aptinit=None,
            in_dim=self.n_features,
            out_dim=self.hmax * self.n_features,
        )
        # Upstream passes 4-D [B,C,N,T] activations to Conv1d modules whose
        # saved weights encode (1,k). Route these exact layers through the
        # mathematically matching 2-D convolution; keep module/parameter names
        # and tensor shapes intact for provenance and state-dict inspection.
        for module in self.network.modules():
            if isinstance(module, nn.Conv1d) and module.weight.ndim == 4:
                module.forward = types.MethodType(_gwnet_gate_conv2d_compat, module)

    def forward(self, x, mask=None, age=None, raw=None, policy=None, event_priors=None,
                return_trace=False):
        values = _valid_input(x, mask, self.history, self.n_nodes, self.n_features)
        # Official gwnet consumes [B,F,N,L] and returns [B,H*F,N,Tout].
        # Select only the last temporal convolution output, never a future target.
        native = self.network(values.permute(0, 3, 2, 1).contiguous())
        logits = native[..., -1].reshape(
            x.shape[0], self.hmax, self.n_features, self.n_nodes
        ).permute(0, 1, 3, 2)
        prediction = self._raw_forecast(logits)
        counters = {"adaptive_adjacency_builds": 1,
                    "graph_convolution_calls": self.network.blocks * self.network.layers,
                    "recurrent_cell_updates": 0,
                    "readout_calls": 2,
                    "gate_conv2d_compat_calls_per_forward": sum(1 for m in self.network.gate_convs if m.weight.ndim == 4),
                    "native_temporal_outputs": int(native.shape[-1])}
        return self._result(prediction, counters=counters,
                            trace=[] if not return_trace else [])

class AGCRNForecast(_ForecastBaseline):
    """AGCRN official encoder and direct horizon×feature convolution head."""

    def __init__(self, *, hidden_dim: int = 64, embed_dim: int = 10,
                 num_layers: int = 2, cheb_k: int = 2, **kwargs):
        super().__init__(**_setup_defaults(kwargs))
        official = _load_agcrn()
        # The PEMS04 author config supplies 64 hidden units, 10-D node embeddings,
        # two recurrent layers and Chebyshev order 2; only task shapes are changed.
        args = SimpleNamespace(
            num_nodes=self.n_nodes,
            input_dim=self.n_features,
            rnn_units=int(hidden_dim),
            output_dim=self.n_features,
            horizon=self.hmax,
            num_layers=int(num_layers),
            default_graph=False,
            embed_dim=int(embed_dim),
            cheb_k=int(cheb_k),
        )
        self.network = official(args)
        _init_agcrn_author(self.network)

    def forward(self, x, mask=None, age=None, raw=None, policy=None, event_priors=None,
                return_trace=False):
        values = _valid_input(x, mask, self.history, self.n_nodes, self.n_features)
        # Upstream signature requires a targets slot but its forward never reads it;
        # pass None and teacher_forcing_ratio=0.0, never future labels.
        logits = self.network(values, targets=None, teacher_forcing_ratio=0.0)
        prediction = self._raw_forecast(logits)
        cells = self.history * self.network.num_layers
        counters = {"adaptive_adjacency_builds": 2*cells,
                    "graph_convolution_calls": 2*cells,
                    "recurrent_cell_updates": cells,
                    "readout_calls": 1,
                    "horizon_decoder_steps": 0}
        return self._result(prediction, counters=counters,
                            trace=[] if not return_trace else [])

class STAEformerForecast(_ForecastBaseline):
    """STAEformer author model with official time-of-day/week embeddings disabled."""

    def __init__(self, **kwargs):
        super().__init__(**_setup_defaults(kwargs))
        official = _load_staeformer()
        # Retain published architecture defaults; adapt only L/H/F and disable
        # invalid calendar embeddings via the constructor's official dimensions.
        self.network = official(
            num_nodes=self.n_nodes,
            in_steps=self.history,
            out_steps=self.hmax,
            input_dim=self.n_features,
            output_dim=self.n_features,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
        )

    def forward(self, x, mask=None, age=None, raw=None, policy=None, event_priors=None,
                return_trace=False):
        values = _valid_input(x, mask, self.history, self.n_nodes, self.n_features)
        logits = self.network(values)
        prediction = self._raw_forecast(logits)
        counters = {"adaptive_adjacency_builds": 0,
                    "graph_convolution_calls": 0,
                    "recurrent_cell_updates": 0,
                    "attention_module_calls": 2 * self.network.num_layers,
                    "readout_calls": 1}
        return self._result(prediction, counters=counters,
                            trace=[] if not return_trace else [])
