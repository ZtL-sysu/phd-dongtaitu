"""Staged, unregistered adapters for three pinned RP forecasting models.

This file is deliberately outside ``edsgn.src`` and is not wired into the
package factory.  Upstream neural modules are loaded from their local source
snapshots without editing those snapshots.  All adapters accept
``[B,L,N,F]`` normalized history and return the shared baseline result dict.
For these architectures, unavailable feature values are zero-filled, while
mask and age are intentionally not passed as learned channels; that capability
gap must be reported when comparing against methods that consume them.
"""
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

_RUN_ROOT = Path(__file__).resolve().parents[2]
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


def _tensor_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _source_hashes():
    paths = [_SOURCE_ROOT/"Graph-WaveNet"/"model.py",
             *sorted((_SOURCE_ROOT/"AGCRN"/"model").glob("*.py")),
             _SOURCE_ROOT/"AGCRN"/"model"/"Run.py",
             _SOURCE_ROOT/"STAEformer-pinned"/"model"/"STAEformer.py"]
    return {str(p.relative_to(_RUN_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def _conv_compat_parity(module):
    """Compare adapter Conv2d with per-node flattened Conv1d, including gradients."""
    B, C, N, T = 2, module.weight.shape[1], 5, 13
    torch.manual_seed(41017)
    random_input = torch.randn(B, C, N, T)
    impulse_input = torch.zeros(B, C, N, T); impulse_input[1, min(1,C-1), 2, 5] = 1.0
    checks = {}
    for case, x0 in (("random", random_input), ("impulse", impulse_input)):
        x1=x0.clone().requires_grad_(); x2=x0.clone().requires_grad_()
        w1=module.weight.detach().clone().requires_grad_();w2=module.weight.detach().clone().requires_grad_()
        b1=module.bias.detach().clone().requires_grad_();b2=module.bias.detach().clone().requires_grad_()
        proxy=types.SimpleNamespace(kernel_size=module.kernel_size,weight=w1,bias=b1,
            stride=module.stride,padding=module.padding,dilation=module.dilation,groups=module.groups)
        got=_gwnet_gate_conv2d_compat(proxy,x1)
        flat=x2.permute(0,2,1,3).reshape(B*N,C,T)
        ref=F.conv1d(flat,w2.squeeze(2),b2,stride=int(module.stride[-1]),
            padding=int(module.padding[-1]),dilation=int(module.dilation[-1]),groups=module.groups)
        ref=ref.reshape(B,N,ref.shape[1],ref.shape[-1]).permute(0,2,1,3).contiguous()
        got.square().sum().backward();ref.square().sum().backward()
        checks[case]={"output_max_abs_diff":float((got-ref).abs().max().detach()),
            "input_grad_max_abs_diff":float((x1.grad-x2.grad).abs().max().detach()),
            "weight_grad_max_abs_diff":float((w1.grad-w2.grad).abs().max().detach()),
            "bias_grad_max_abs_diff":float((b1.grad-b2.grad).abs().max().detach())}
    if any(v > 1e-6 for case in checks.values() for v in case.values()):
        raise AssertionError(f"Graph WaveNet Conv compatibility parity failed: {checks}")
    return checks


def _smoke():
    """CPU-only task-shape/init/gradient/reload checks; no optimizer or data."""
    torch.set_num_threads(2)
    import torchinfo
    started = time.perf_counter(); source_before = _source_hashes()
    reports = []; parity = None; agcrn_repeatable = None
    constructors = (GraphWaveNetForecast, AGCRNForecast, STAEformerForecast)
    for cls in constructors:
        for features, count_data in ((1, True), (2, True), (1, False), (2, False)):
            torch.manual_seed(104729)
            kwargs = dict(n_nodes=5, n_features=features, history=28, hmax=7,
                          count_data=count_data, norm_mean=[0.1]*features,
                          norm_std=[1.2]*features, output_scale=[1.4+i for i in range(features)],
                          output_mean=[0.2+i for i in range(features)])
            model = cls(**kwargs).cpu()
            if cls is AGCRNForecast and agcrn_repeatable is None:
                state = {k:v.detach().clone() for k,v in model.state_dict().items()}
                torch.manual_seed(104729); again = cls(**kwargs).cpu()
                agcrn_repeatable = all(torch.equal(v, again.state_dict()[k]) for k,v in state.items())
                if not agcrn_repeatable: raise AssertionError("AGCRN author initialization not seed-repeatable")
            if cls is GraphWaveNetForecast and parity is None:
                parity = _conv_compat_parity(model.network.gate_convs[0])
            model.train(); x = torch.randn(2, 28, 5, features, requires_grad=True)
            mask = torch.ones_like(x, dtype=torch.bool); mask[:, 2, 1, :] = False
            age = torch.zeros_like(x); age[:, 2, 1, :] = 3
            before = _tensor_hash(model)
            result = model(x, mask=mask, age=age, raw=None, policy=None,
                           event_priors=None, return_trace=False)
            pred = result["predictions"]
            if tuple(pred.shape) != (2, 7, 5, features):
                raise AssertionError(f"{cls.__name__}: unexpected prediction shape {tuple(pred.shape)}")
            if not torch.isfinite(pred).all(): raise AssertionError(f"{cls.__name__}: nonfinite prediction")
            if count_data and (pred < 0).any(): raise AssertionError(f"{cls.__name__}: negative count forecast")
            pred.square().mean().backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            nonnull = [g for g in grads if g is not None]
            if not nonnull or any(not torch.isfinite(g).all() for g in nonnull):
                raise AssertionError(f"{cls.__name__}: missing/nonfinite parameter gradient")
            if _tensor_hash(model) != before: raise AssertionError(f"{cls.__name__}: parameters changed in forward/backward")
            model.eval(); test_x = x.detach()
            expected = model(test_x, mask=mask, age=age)["predictions"].detach().clone()
            checkpoint = io.BytesIO(); torch.save(model.state_dict(), checkpoint); checkpoint.seek(0)
            torch.manual_seed(104729); reloaded = cls(**kwargs).cpu()
            reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)); reloaded.eval()
            actual = reloaded(test_x, mask=mask, age=age)["predictions"].detach()
            if not torch.equal(expected, actual): raise AssertionError(f"{cls.__name__}: reload predictions differ")
            reports.append({"adapter":cls.__name__, "features":features, "count_data":count_data,
                "batch_size":2, "nodes":5, "history":28, "horizon":7,
                "prediction_shape":list(pred.shape), "prediction_finite":True,
                "count_nonnegative":True if count_data else None, "all_nonnull_gradients_finite":True,
                "parameter_hash_unchanged":True, "checkpoint_reload_predictions_exact":True,
                "parameter_count":sum(p.numel() for p in model.parameters()),
                "parameter_gradient_count":len(nonnull), "counters":result["counters"]})
    source_after = _source_hashes()
    if source_before != source_after: raise AssertionError("Pinned upstream source hashes changed during smoke")
    return {"schema_version":1, "scope":"resource/functionality diagnostic",
        "primary_scientific_timing":False, "status":"passed", "device":"cpu", "torch_threads":2,
        "torch_version":torch.__version__, "torchinfo_version":importlib.metadata.version("torchinfo"),
        "optimizer_steps":0, "dataset_accessed":False, "test_accessed":False,
        "future_targets_passed":False, "case_count":len(reports),
        "agcrn_author_init_seed_repeatable":agcrn_repeatable,
        "gwnet_conv2d_compatibility_parity":parity, "checks":reports,
        "upstream_source_sha256_before":source_before, "upstream_source_sha256_after":source_after,
        "upstream_sources_unchanged":True, "wall_seconds":time.perf_counter()-started}


if __name__ == "__main__":
    import json
    process_started = time.perf_counter()
    report = _smoke()
    report["smoke_wall_seconds"] = report["wall_seconds"]
    report["wall_seconds"] = time.perf_counter() - process_started
    print(json.dumps(report, indent=2))
