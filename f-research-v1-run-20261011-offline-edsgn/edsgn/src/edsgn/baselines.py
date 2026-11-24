"""Comparable local count-forecast baselines.

All predictors accept the shared offline interface and emit raw-scale forecasts.
They consume only historical normalized values, masks, and observation ages.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _feature_vector(value: Any, width: int, default: float) -> torch.Tensor:
    if value is None:
        return torch.full((width,), float(default), dtype=torch.float32)
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 1:
        tensor = tensor.expand(width).clone()
    if tensor.numel() != width:
        raise ValueError(f"Expected one or {width} feature statistics, got {tensor.numel()}")
    return tensor


class _ForecastBaseline(nn.Module):
    """Shared feature metadata and target-scale conversion."""

    def __init__(
        self,
        *,
        n_nodes: int,
        n_features: int,
        history: int,
        hmax: int,
        count_data: bool,
        norm_mean: Any = None,
        norm_std: Any = None,
        output_scale: Any = None,
        output_mean: Any = None,
    ) -> None:
        super().__init__()
        if min(n_nodes, n_features, history, hmax) < 1:
            raise ValueError("Baseline dimensions must be positive")
        self.n_nodes = int(n_nodes)
        self.n_features = int(n_features)
        self.history = int(history)
        self.hmax = int(hmax)
        self.count_data = bool(count_data)
        self.register_buffer("norm_mean", _feature_vector(norm_mean, n_features, 0.0))
        self.register_buffer("norm_std", _feature_vector(norm_std, n_features, 1.0).clamp_min(1e-6))
        self.register_buffer("output_scale", _feature_vector(output_scale, n_features, 1.0).clamp_min(1e-8))
        self.register_buffer("output_mean", _feature_vector(output_mean, n_features, 0.0))

    def _inputs(self, x, mask=None, age=None):
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, history, nodes, features]")
        b, t, n, f = x.shape
        if (t, n, f) != (self.history, self.n_nodes, self.n_features):
            raise ValueError(
                f"Expected [B,{self.history},{self.n_nodes},{self.n_features}], "
                f"received {tuple(x.shape)}"
            )
        if mask is None:
            valid = torch.ones_like(x, dtype=torch.bool)
        else:
            valid = torch.as_tensor(mask, device=x.device).bool()
            if valid.ndim == 3:
                valid = valid.unsqueeze(-1).expand_as(x)
            if valid.shape != x.shape:
                raise ValueError("mask must be [B,L,N] or [B,L,N,F]")
        if age is None:
            age_features = torch.zeros_like(x)
        else:
            age_features = torch.as_tensor(age, device=x.device, dtype=x.dtype)
            if age_features.ndim == 3:
                age_features = age_features.unsqueeze(-1)
            age_features = age_features.expand_as(x).clamp_min(0)
            age_features = torch.log1p(age_features) / math.log1p(max(self.history, 1))
        return torch.cat((x * valid, valid.to(x.dtype), age_features), dim=-1)

    def _raw_forecast(self, logits):
        scale = self.output_scale.view(1, 1, 1, -1)
        mean = self.output_mean.view(1, 1, 1, -1)
        if self.count_data:
            return F.softplus(logits) * scale
        return logits * scale + mean

    def _init_output_bias(self, layer: nn.Linear) -> None:
        with torch.no_grad():
            if self.count_data:
                ratio = (self.output_mean / self.output_scale).clamp_min(1e-3)
                bias = ratio + torch.log(-torch.expm1(-ratio))
            else:
                bias = torch.zeros_like(self.output_mean)
            layer.bias.copy_(bias.repeat(self.hmax))

    @staticmethod
    def _result(predictions, *, counters=None, trace=None, intermediate=None):
        zero = predictions.sum() * 0.0
        return {
            "predictions": predictions,
            "graph_loss": zero,
            "counters": counters or {},
            "trace": trace or [],
            "runtime": None,
            "intermediate": intermediate or [],
        }


class LinearPredictor(_ForecastBaseline):
    """Per-node linear multi-horizon baseline over the complete history."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.head = nn.Linear(self.history * 3 * self.n_features, self.hmax * self.n_features)
        self._init_output_bias(self.head)

    def forward(
        self, x, mask=None, age=None, raw=None, policy=None, event_priors=None, return_trace=False
    ):
        features = self._inputs(x, mask, age)
        b, _, n, _ = features.shape
        flattened = features.permute(0, 2, 1, 3).reshape(b, n, -1)
        logits = self.head(flattened).reshape(b, n, self.hmax, self.n_features)
        predictions = self._raw_forecast(logits.permute(0, 2, 1, 3))
        counters = {"readout_calls": 1, "graph_calls": 0, "state_calls": 0}
        return self._result(predictions, counters=counters, trace=[] if return_trace else [])


class CompactGRU(_ForecastBaseline):
    """Small shared univariate-per-node GRU control with a direct multi-step head."""

    def __init__(self, *, hidden_size: int = 64, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.gru = nn.GRU(3 * self.n_features, self.hidden_size, batch_first=True)
        self.head = nn.Linear(self.hidden_size, self.hmax * self.n_features)
        self._init_output_bias(self.head)

    def forward(
        self, x, mask=None, age=None, raw=None, policy=None, event_priors=None, return_trace=False
    ):
        features = self._inputs(x, mask, age)
        b, t, n, channels = features.shape
        sequence = features.permute(0, 2, 1, 3).reshape(b * n, t, channels)
        _, hidden = self.gru(sequence)
        logits = self.head(hidden[-1]).reshape(b, n, self.hmax, self.n_features)
        predictions = self._raw_forecast(logits.permute(0, 2, 1, 3))
        counters = {"readout_calls": 1, "graph_calls": 0, "state_calls": 0}
        return self._result(predictions, counters=counters, trace=[] if return_trace else [])


def build_baseline(
    name: str,
    *,
    n_nodes: int,
    n_features: int,
    history: int,
    hmax: int,
    count_data: bool = False,
    norm_mean=None,
    norm_std=None,
    output_scale=None,
    output_mean=None,
    **kwargs,
) -> nn.Module:
    """Construct a baseline using the same fitted, training-only scale metadata."""
    # Engine statistics include residual_scale for SNN confidence monitors;
    # the forecasting baselines have no monitor and must ignore that field.
    kwargs.pop("residual_scale", None)
    options = dict(
        n_nodes=n_nodes,
        n_features=n_features,
        history=history,
        hmax=hmax,
        count_data=count_data,
        norm_mean=norm_mean,
        norm_std=norm_std,
        output_scale=output_scale,
        output_mean=output_mean,
    )
    key = name.lower().replace("-", "_")
    if key in {"linear", "linear_predictor"}:
        return LinearPredictor(**options)
    if key in {"gru", "compactgru", "compact_gru"}:
        return CompactGRU(**options, **kwargs)
    if key in {"dcrnn", "dcrnn_adapted"}:
        from .dcrnn_baseline import DCRNNForecast, load_bound_training_prior

        prior = load_bound_training_prior(
            prior_path=kwargs.pop("fixed_training_prior_path", None),
            prior_sha256=kwargs.pop("fixed_training_prior_sha256", None),
            manifest_path=kwargs.pop("fixed_training_prior_manifest_path", None),
            manifest_sha256=kwargs.pop("fixed_training_prior_manifest_sha256", None),
            expected_node_ids=kwargs.pop("expected_node_ids", None),
            expected_dataset_path=kwargs.pop("expected_dataset_path", None),
            expected_n_nodes=n_nodes,
        )
        return DCRNNForecast(fixed_training_prior=prior, **options, **kwargs)
    if key in {"graphwavenet", "graph_wave_net", "gwnet"}:
        from .rp_baselines import GraphWaveNetForecast
        return GraphWaveNetForecast(**options, **kwargs)
    if key == "agcrn":
        from .rp_baselines import AGCRNForecast
        return AGCRNForecast(**options, **kwargs)
    if key in {"staeformer", "stae_former"}:
        from .rp_baselines import STAEformerForecast
        return STAEformerForecast(**options, **kwargs)
    if key in {"dgcrn", "dgcrn_adapted"}:
        from .external_baselines import DGCRNAdapted

        return DGCRNAdapted(**options, **kwargs)
    if key in {"spikestag", "spikystag", "spikestag_paper"}:
        from .external_baselines import SpikeSTAGPaper

        return SpikeSTAGPaper(**options, **kwargs)
    if key in {"seqsnn", "ispikformer", "seqsnn_ispikformer"}:
        from .external_baselines import SeqSNNForecast

        return SeqSNNForecast(**options, **kwargs)
    raise ValueError(f"Unknown baseline {name!r}")
