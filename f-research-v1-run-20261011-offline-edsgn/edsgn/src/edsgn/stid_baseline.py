"""STID forecast adapter loaded from the pinned official source snapshot.

This module deliberately remains independent of the central baseline factory;
the task owner can register it after inspecting `reviews/stid-baseline-code-audit.md`.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

from .baselines import _ForecastBaseline


def _load_official_stid():
    module_name = "_edsgn_pinned_stid.arch"
    if module_name in sys.modules:
        return sys.modules[module_name].STID
    source_root = Path(__file__).resolve().parents[3] / "sources" / "STID" / "stid"
    source_arch = source_root / "arch"
    root_package = types.ModuleType("_edsgn_pinned_stid")
    root_package.__path__ = [str(source_root)]
    sys.modules[root_package.__name__] = root_package
    spec = importlib.util.spec_from_file_location(
        module_name,
        source_arch / "__init__.py",
        submodule_search_locations=[str(source_arch)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pinned STID source at {source_arch}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.STID


class STIDForecast(_ForecastBaseline):
    """STID with official residual MLP/node identity and multi-feature head.

    Time-of-day and day-of-week identities are explicitly disabled because the
    local binned datasets have no valid calendar covariates. The original
    network accepts multiple input channels but predicts one target channel;
    only its output projection is adapted to H*F task outputs.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 32,
        node_dim: int = 32,
        num_layer: int = 3,
        **kwargs,
    ) -> None:
        kwargs.pop("residual_scale", None)
        super().__init__(**kwargs)
        official_stid = _load_official_stid()
        args = {
            "num_nodes": self.n_nodes,
            "node_dim": int(node_dim),
            "input_len": self.history,
            "input_dim": self.n_features,
            "embed_dim": int(embed_dim),
            "output_len": self.hmax * self.n_features,
            "num_layer": int(num_layer),
            "temp_dim_tid": 0,
            "temp_dim_diw": 0,
            "time_of_day_size": 1,
            "day_of_week_size": 1,
            "if_T_i_D": False,
            "if_D_i_W": False,
            "if_node": True,
        }
        self.network = official_stid(**args)

    def forward(
        self,
        x,
        mask=None,
        age=None,
        raw=None,
        policy=None,
        event_priors=None,
        return_trace=False,
    ):
        if x.ndim != 4 or tuple(x.shape[1:]) != (
            self.history,
            self.n_nodes,
            self.n_features,
        ):
            raise ValueError(
                f"Expected [B,{self.history},{self.n_nodes},{self.n_features}], got {tuple(x.shape)}"
            )
        if mask is None:
            values = x
        else:
            valid = torch.as_tensor(mask, device=x.device).bool()
            if valid.ndim == 3:
                valid = valid.unsqueeze(-1).expand_as(x)
            if valid.shape != x.shape:
                raise ValueError("mask must be [B,L,N] or [B,L,N,F]")
            values = x * valid.to(x.dtype)
        # Official STID returns [B, H*F, N, 1]; preserve output-channel order
        # (horizon-major, then feature) and convert to the shared [B,H,N,F].
        logits = self.network(values, None, 0, 0, False)
        b = x.shape[0]
        logits = logits.squeeze(-1).reshape(
            b, self.hmax, self.n_features, self.n_nodes
        ).permute(0, 1, 3, 2)
        prediction = self._raw_forecast(logits)
        counters = {
            "graph_calls": 0,
            "state_calls": 0,
            "readout_calls": 1,
            "node_identity_parameters": self.n_nodes * self.network.node_emb.shape[1],
        }
        return self._result(prediction, counters=counters, trace=[] if not return_trace else [])
