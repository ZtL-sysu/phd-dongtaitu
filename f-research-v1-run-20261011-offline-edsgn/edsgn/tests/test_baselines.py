import torch
from types import SimpleNamespace

from edsgn.baselines import build_baseline
from edsgn.engine import build_model


def _make(name, **kwargs):
    return build_baseline(
        name,
        n_nodes=4,
        n_features=2,
        history=5,
        hmax=3,
        count_data=True,
        norm_mean=[0.7, 0.9],
        norm_std=[0.4, 0.6],
        output_scale=[2.0, 3.0],
        output_mean=[1.5, 2.5],
        **kwargs,
    )


def test_simple_baselines_forward_backward_count_scale():
    torch.manual_seed(12)
    x = torch.randn(2, 5, 4, 2)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, 1, 2, 0] = False
    age = torch.zeros_like(x)
    age[:, 1, 2, 0] = 2
    for name in ("linear", "compact_gru"):
        model = _make(name)
        result = model(x, mask=mask, age=age, return_trace=True)
        assert result["predictions"].shape == (2, 3, 4, 2)
        assert torch.isfinite(result["predictions"]).all()
        assert (result["predictions"] >= 0).all()
        (result["predictions"].mean() + result["graph_loss"]).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_dgcrn_and_spikestag_adapted_cpu_gradient_and_h24_autoregression():
    torch.manual_seed(31)
    # Covers the email/MOOC longest horizon shape without passing labels to forward.
    for name, options in (
        ("dgcrn", {"hidden_size": 8, "node_dim": 4, "gcn_depth": 1}),
        ("seqsnn", {"dim": 8, "d_ff": 16, "depths": 1, "num_steps": 2,
                    "heads": 2, "encoder_type": "conv"}),
    ):
        model = build_baseline(
            name, n_nodes=3, n_features=1, history=24, hmax=24, count_data=True,
            norm_mean=[0.8], norm_std=[0.5], output_scale=[2.0], output_mean=[1.0], **options
        )
        x = torch.randn(1, 24, 3, 1)
        mask = torch.ones_like(x, dtype=torch.bool)
        result = model(x, mask=mask, return_trace=True)
        assert result["predictions"].shape == (1, 24, 3, 1)
        assert torch.isfinite(result["predictions"]).all()
        assert (result["predictions"] >= 0).all()
        if name == "dgcrn":
            assert result["counters"]["graph_calls"] > 0
            assert result["counters"]["dynamic_dense_adjacency_elements"] > 0
        if name == "seqsnn":
            assert result["counters"]["spike_time_steps"] == 2
        (result["predictions"].mean() + result["graph_loss"]).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_engine_build_model_accepts_full_training_statistics():
    ds = SimpleNamespace(
        values=torch.zeros(1, 3, 1), history=24, hmax=24, count_data=True
    )
    stats = {
        "norm_mean": [0.8],
        "norm_std": [0.5],
        "residual_scale": [[1.0], [2.0], [1.5]],
        "output_scale": [2.0],
        "output_mean": [1.0],
    }
    for variant, options in (
        ("GRU", {"hidden_size": 8}),
        ("DGCRN", {"hidden_size": 8, "node_dim": 4, "gcn_depth": 1}),
        ("SeqSNN", {"dim": 8, "d_ff": 16, "depths": 1, "num_steps": 2,
                    "heads": 2, "encoder_type": "conv"}),
    ):
        model = build_model({"variant": variant, "baseline": options}, ds, stats)
        result = model(torch.randn(1, 24, 3, 1))
        assert result["predictions"].shape == (1, 24, 3, 1)
