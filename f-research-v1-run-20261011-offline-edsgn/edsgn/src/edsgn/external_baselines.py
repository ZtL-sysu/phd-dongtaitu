"""Pinned external forecast baselines and explicitly labeled adaptations.

DGCRNAdapted re-expresses the author-linked DGCRN graph generator and DGCRM
layers at the shared count-forecast interface. Its architectural equations
follow sources/DGCRN/methods/DGCRN/{net.py,layer.py} at commit
b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228. The MIT notice is preserved in
that source snapshot. See reviews/external-baseline-code-audit.md for the
task-head, graph-prior, and teacher-forcing changes.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .baselines import _ForecastBaseline


def _load_official_seqsnn_ispikformer():
    """Load the pinned official iSpikformer modules without SeqSNN's full stack.

    The upstream package initializer imports snntorch and its forecasting
    framework even though this model only needs SpikingJelly. Namespace
    packages plus a dummy registry isolate the unchanged upstream source files.
    """
    module_name = "SeqSNN.network.snn.ispikformer"
    if module_name in sys.modules:
        return sys.modules[module_name].iSpikformer
    source = Path(__file__).resolve().parents[3] / "sources" / "SeqSNN" / "SeqSNN"

    def namespace(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    namespace("SeqSNN", source)
    namespace("SeqSNN.module", source / "module")
    namespace("SeqSNN.module.encoding", source / "module" / "encoding")
    snn_stub = types.ModuleType("SeqSNN.module.encoding.snntorch")
    snn_stub.encoder = types.SimpleNamespace(RepeatEncoder=None, ConvEncoder=None, DeltaEncoder=None)
    sys.modules["SeqSNN.module.encoding.snntorch"] = snn_stub
    namespace("SeqSNN.network", source / "network")
    namespace("SeqSNN.network.snn", source / "network" / "snn")
    base = types.ModuleType("SeqSNN.network.base")

    class Registry:
        def register_module(self, *_args, **_kwargs):
            return lambda cls: cls

    base.NETWORKS = Registry()
    sys.modules["SeqSNN.network.base"] = base
    return importlib.import_module(module_name).iSpikformer


class SeqSNNForecast(_ForecastBaseline):
    """Official Microsoft SeqSNN iSpikformer with the upstream direct decoder.

    The SNN backbone is imported unchanged from the pinned source snapshot.
    The forecasting wrapper follows SeqSNN's `SNNVariateSpecTS` head: a shared
    linear projection from each variate embedding to `out_size` future steps.
    """

    def __init__(
        self,
        *,
        dim: int = 256,
        d_ff: int | None = 1024,
        depths: int = 2,
        num_steps: int = 4,
        heads: int = 8,
        encoder_type: str = "delta",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        ispikformer = _load_official_seqsnn_ispikformer()
        self.backbone = ispikformer(
            dim=dim,
            d_ff=d_ff,
            depths=depths,
            num_steps=num_steps,
            heads=heads,
            max_length=self.history,
            encoder_type=encoder_type,
        )
        self.head = nn.Linear(dim, self.hmax)
        with torch.no_grad():
            if self.count_data:
                ratio = (self.output_mean / self.output_scale).clamp_min(1e-3).mean()
                self.head.bias.fill_(float(ratio + torch.log(-torch.expm1(-ratio))))
            else:
                self.head.bias.zero_()

    def forward(self, x, mask=None, age=None, raw=None, policy=None, event_priors=None,
                return_trace=False):
        # SeqSNN's forecasting task assumes fully observed synchronous variates.
        # This project's binned count arrays are complete; masked entries are
        # zero-filled, but mask/age are not additional learned channels.
        if mask is None:
            values = x
        else:
            valid = torch.as_tensor(mask, device=x.device).to(x.dtype)
            if valid.ndim == 3:
                valid = valid.unsqueeze(-1).expand_as(x)
            values = x * valid
        b, l, n, f = values.shape
        flattened = values.reshape(b, l, n * f)
        _sequence, embedding = self.backbone(flattened)
        logits = self.head(embedding).transpose(1, 2).reshape(b, self.hmax, n, f)
        pred = self._raw_forecast(logits)
        counters = {
            "spike_time_steps": self.backbone.T,
            "graph_calls": 0,
            "state_calls": self.backbone.T * (1 + 4 * self.backbone.depths),
            "readout_calls": 1,
            "attention_token_count": n * f,
            "dense_attention_elements_per_step": (n * f) ** 2,
        }
        return self._result(pred, counters=counters, trace=[] if not return_trace else [])


class _DGCRNGraphConv(nn.Module):
    """DGCRN's diffusion stack, translated from the pinned official layer.py."""

    def __init__(
        self,
        dims: list[int],
        depth: int,
        *,
        mode: str,
        alpha: float = 0.05,
        beta: float = 0.95,
        gamma: float = 0.95,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.mode = mode
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        if mode == "hyper":
            self.mlp = nn.Sequential(
                nn.Linear((depth + 1) * dims[0], dims[1]),
                nn.Sigmoid(),
                nn.Linear(dims[1], dims[2]),
                nn.Sigmoid(),
                nn.Linear(dims[2], dims[3]),
            )
        elif mode == "rnn":
            self.mlp = nn.Linear((depth + 1) * dims[0], dims[1])
        else:
            raise ValueError(f"Unknown DGCRN graph-convolution mode {mode!r}")

    def forward(self, x, supports):
        h = x
        outputs = [h]
        if self.mode == "hyper":
            (support,) = supports
            for _ in range(self.depth):
                h = self.alpha * x + self.gamma * torch.einsum("bvc,vw->bwc", h, support)
                outputs.append(h)
        else:
            dynamic, predefined = supports
            for _ in range(self.depth):
                h = (
                    self.alpha * x
                    + self.beta * torch.einsum("bvc,bvw->bwc", h, dynamic)
                    + self.gamma * torch.einsum("bvc,vw->bwc", h, predefined)
                )
                outputs.append(h)
        return self.mlp(torch.cat(outputs, dim=-1))


class _DGCRNHyperPair(nn.Module):
    """Two source hyper-GCN branches that generate one dynamic filter."""

    def __init__(self, input_width, hyper_width, middle_width, node_width, depth):
        super().__init__()
        dims = [input_width, hyper_width, middle_width, node_width]
        self.first = _DGCRNGraphConv(dims, depth, mode="hyper")
        self.second = _DGCRNGraphConv(dims, depth, mode="hyper")

    def forward(self, hyper_input, static_support):
        return self.first(hyper_input, (static_support,)) + self.second(
            hyper_input, (static_support,)
        )


class _DGCRNDirectionalGRU(nn.Module):
    """DGCRM update with the two dynamic directions and static support."""

    def __init__(self, input_width, hidden_width, depth):
        super().__init__()
        dims = [input_width + hidden_width, hidden_width]
        self.z_forward = _DGCRNGraphConv(dims, depth, mode="rnn")
        self.z_reverse = _DGCRNGraphConv(dims, depth, mode="rnn")
        self.r_forward = _DGCRNGraphConv(dims, depth, mode="rnn")
        self.r_reverse = _DGCRNGraphConv(dims, depth, mode="rnn")
        self.c_forward = _DGCRNGraphConv(dims, depth, mode="rnn")
        self.c_reverse = _DGCRNGraphConv(dims, depth, mode="rnn")

    def forward(self, x, hidden, dynamic, dynamic_transpose, static_support):
        combined = torch.cat((x, hidden), dim=-1)
        forward_supports = (dynamic, static_support)
        reverse_supports = (dynamic_transpose, static_support.T.contiguous())
        z = torch.sigmoid(
            self.z_forward(combined, forward_supports)
            + self.z_reverse(combined, reverse_supports)
        )
        r = torch.sigmoid(
            self.r_forward(combined, forward_supports)
            + self.r_reverse(combined, reverse_supports)
        )
        gated = torch.cat((x, r * hidden), dim=-1)
        candidate = torch.tanh(
            self.c_forward(gated, forward_supports)
            + self.c_reverse(gated, reverse_supports)
        )
        return z * hidden + (1.0 - z) * candidate


class _DGCRNCell(nn.Module):
    """Encoder or decoder cell with a generated skew-symmetric adjacency."""

    def __init__(self, input_width, hidden_width, node_width, *, depth):
        super().__init__()
        hyper_width = 16
        middle_width = 2
        hyper_input = input_width + hidden_width
        self.filter_one = _DGCRNHyperPair(
            hyper_input, hyper_width, middle_width, node_width, depth
        )
        self.filter_two = _DGCRNHyperPair(
            hyper_input, hyper_width, middle_width, node_width, depth
        )
        self.gru = _DGCRNDirectionalGRU(input_width, hidden_width, depth)

    def forward(self, x, hidden, embedding_one, embedding_two, static_support, alpha):
        hyper_input = torch.cat((x, hidden), dim=-1)
        filter_one = self.filter_one(hyper_input, static_support)
        filter_two = self.filter_two(hyper_input, static_support)
        node_one = torch.tanh(alpha * embedding_one.unsqueeze(0) * filter_one)
        node_two = torch.tanh(alpha * embedding_two.unsqueeze(0) * filter_two)
        skew = node_one @ node_two.transpose(1, 2) - node_two @ node_one.transpose(1, 2)
        adjacency = F.relu(torch.tanh(alpha * skew))
        identity = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype)
        dynamic = adjacency + identity
        dynamic = dynamic / dynamic.sum(-1, keepdim=True).clamp_min(1e-12)
        transpose = adjacency.transpose(1, 2) + identity
        dynamic_transpose = transpose / transpose.sum(-1, keepdim=True).clamp_min(1e-12)
        hidden = self.gru(x, hidden, dynamic, dynamic_transpose, static_support)
        return hidden, adjacency


class DGCRNAdapted(_ForecastBaseline):
    """DGCRN adapted to fixed-cohort multi-feature activity forecasting.

    Preserved: learned node embeddings, hypernetwork-generated skew dynamic
    graph, separately parameterized encoder/decoder graph generators, directed
    adjacency and transposed-adjacency diffusion, and recurrent DGCRM gates.

    Task adaptations: identity is the only static support because these data
    have no legitimate fixed road graph; historical value/mask/age channels are
    supplied at encoder steps; the decoder is autoregressive over all F count
    features and never receives future labels or fabricated calendar inputs.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        node_dim: int = 40,
        gcn_depth: int = 2,
        tanh_alpha: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.node_dim = int(node_dim)
        self.gcn_depth = int(gcn_depth)
        self.tanh_alpha = float(tanh_alpha)
        channels = 3 * self.n_features
        self.embedding_one = nn.Parameter(torch.empty(self.n_nodes, self.node_dim))
        self.embedding_two = nn.Parameter(torch.empty(self.n_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.embedding_one)
        nn.init.xavier_uniform_(self.embedding_two)
        self.encoder = _DGCRNCell(channels, hidden_size, node_dim, depth=gcn_depth)
        self.decoder = _DGCRNCell(channels, hidden_size, node_dim, depth=gcn_depth)
        self.head = nn.Linear(hidden_size, self.n_features)
        self._init_single_output_bias(self.head)
        self.register_buffer(
            "static_support",
            torch.eye(self.n_nodes, dtype=torch.float32),
        )

    def _init_single_output_bias(self, layer):
        with torch.no_grad():
            if self.count_data:
                ratio = (self.output_mean / self.output_scale).clamp_min(1e-3)
                layer.bias.copy_(ratio + torch.log(-torch.expm1(-ratio)))
            else:
                layer.bias.copy_(torch.zeros_like(self.output_mean))

    def _future_input(self, prediction):
        if self.count_data:
            value = (torch.log1p(prediction.clamp_min(0)) - self.norm_mean) / self.norm_std
        else:
            value = (prediction - self.norm_mean) / self.norm_std
        valid = torch.ones_like(value)
        age = torch.zeros_like(value)
        return torch.cat((value, valid, age), dim=-1)

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
        features = self._inputs(x, mask, age)
        b, _, n, _ = features.shape
        hidden = features.new_zeros((b, n, self.hidden_size))
        adjacency_trace = []
        graph_calls = 0
        for step in range(self.history):
            hidden, adjacency = self.encoder(
                features[:, step],
                hidden,
                self.embedding_one,
                self.embedding_two,
                self.static_support,
                self.tanh_alpha,
            )
            graph_calls += 1
            if return_trace:
                adjacency_trace.append(adjacency.detach())

        decoder_input = features.new_zeros((b, n, features.shape[-1]))
        predictions = []
        for _ in range(self.hmax):
            hidden, adjacency = self.decoder(
                decoder_input,
                hidden,
                self.embedding_one,
                self.embedding_two,
                self.static_support,
                self.tanh_alpha,
            )
            graph_calls += 1
            if return_trace:
                adjacency_trace.append(adjacency.detach())
            logits = self.head(hidden)
            forecast = self._raw_forecast(logits.unsqueeze(1)).squeeze(1)
            predictions.append(forecast)
            decoder_input = self._future_input(forecast)
        pred = torch.stack(predictions, dim=1)
        counters = {
            "graph_calls": graph_calls,
            "dynamic_dense_adjacency_elements": graph_calls * b * n * n,
            "state_calls": graph_calls,
            "readout_calls": self.hmax,
        }
        return self._result(pred, counters=counters, trace=adjacency_trace)


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return (value >= 1.0).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        # Arctangent-family surrogate, matching the paper's stated surrogate class.
        grad = 1.0 / (1.0 + (torch.pi * (value - 1.0)).square())
        return grad_output * grad


class _SpikeSTAGLIF(nn.Module):
    def __init__(self, width: int, decay: float = 0.8):
        super().__init__()
        self.decay = float(decay)
        self.projection = nn.Linear(width, width)

    def forward(self, current, membrane):
        potential = self.decay * membrane + self.projection(current)
        spikes = _SurrogateSpike.apply(potential)
        return spikes, potential * (1.0 - spikes)


class SpikeSTAGPaper(_ForecastBaseline):
    """Runnable paper-derived SpikeSTAG adaptation; not author-code reproduction.

    Implements the stated adaptive adjacency, threshold/top-k multi-scale
    neighborhood aggregation, LIF spike aggregation, LSTM + spiking temporal
    attention fusion. The paper does not provide official code and omits key
    OBS/SSA and sampling details; see the audit for deviations and faithfulness.
    """

    def __init__(self, *, hidden_size=32, node_dim=16, top_k=8, lif_steps=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.node_dim = int(node_dim)
        self.top_k = min(max(int(top_k), 1), max(self.n_nodes - 1, 1))
        self.lif_steps = int(lif_steps)
        width = 3 * self.n_features
        self.node_embed = nn.Parameter(torch.empty(self.n_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_embed)
        self.node_projection = nn.Linear(width, self.hidden_size)
        self.value_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.local_lif = _SpikeSTAGLIF(self.hidden_size)
        self.global_lif = _SpikeSTAGLIF(self.hidden_size)
        self.temporal_lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.attn_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_lif = _SpikeSTAGLIF(self.hidden_size)
        self.fusion_gate = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.head = nn.Linear(self.hidden_size, self.hmax * self.n_features)
        self._init_output_bias(self.head)

    def forward(self, x, mask=None, age=None, raw=None, policy=None, event_priors=None,
                return_trace=False):
        features = self._inputs(x, mask, age)
        b, t, n, _ = features.shape
        adjacency = torch.sigmoid(self.node_embed @ self.node_embed.T)
        adjacency = adjacency + torch.eye(n, device=x.device, dtype=x.dtype)
        threshold = adjacency.mean(-1, keepdim=True)
        candidates = adjacency.masked_fill(adjacency <= threshold, 0.0)
        k = min(self.top_k, n)
        values, indices = candidates.topk(k, dim=-1)
        weights = values / values.sum(-1, keepdim=True).clamp_min(1e-8)
        sequence = self.node_projection(features)
        local_mem = sequence.new_zeros((b, n, self.hidden_size))
        global_mem = torch.zeros_like(local_mem)
        spatial = []
        for step in range(t):
            current = self.value_projection(sequence[:, step])
            gathered = current[:, indices, :]
            neighbor = (gathered * weights[None, :, :, None]).sum(dim=2)
            spike, local_mem = self.local_lif(neighbor, local_mem)
            # A two-hop sampled aggregation approximates MSSA's local/semi-global scales.
            second = (spike[:, indices, :] * weights[None, :, :, None]).sum(dim=2)
            multi = (spike + second) / 2.0
            for _ in range(self.lif_steps - 1):
                multi, global_mem = self.global_lif(multi, global_mem)
            spatial.append(multi)
        encoded = torch.stack(spatial, dim=1)
        flat = encoded.permute(0, 2, 1, 3).reshape(b * n, t, self.hidden_size)
        temporal, _ = self.temporal_lstm(flat)
        q, kvec, v = self.attn_q(temporal), self.attn_k(temporal), self.attn_v(temporal)
        scores = q @ kvec.transpose(-1, -2) / (self.hidden_size ** 0.5)
        membrane = torch.zeros_like(v)
        attn_out = v
        for _ in range(self.lif_steps):
            attn_out, membrane = self.attn_lif(torch.softmax(scores, dim=-1) @ v, membrane)
        temporal_spikes = attn_out[:, -1]
        smooth = temporal[:, -1]
        gate = torch.sigmoid(self.fusion_gate(torch.cat((smooth, temporal_spikes), dim=-1)))
        fused = gate * smooth + (1.0 - gate) * temporal_spikes
        logits = self.head(fused).reshape(b, n, self.hmax, self.n_features).permute(0, 2, 1, 3)
        pred = self._raw_forecast(logits)
        counters = {"graph_calls": t * self.lif_steps, "state_calls": t * self.lif_steps,
                    "readout_calls": 1, "adaptive_dense_adjacency_elements": n * n,
                    "sampled_edges_per_row": k}
        trace = [{"adjacency": adjacency.detach()}] if return_trace else []
        return self._result(pred, counters=counters, trace=trace)
