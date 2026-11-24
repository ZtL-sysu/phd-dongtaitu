import torch

from edsgn.stid_baseline import STIDForecast


def _model(*, history, hmax, n_nodes, n_features, **kwargs):
    return STIDForecast(
        n_nodes=n_nodes,
        n_features=n_features,
        history=history,
        hmax=hmax,
        count_data=True,
        norm_mean=[0.7] * n_features,
        norm_std=[0.5] * n_features,
        output_scale=[2.0] * n_features,
        output_mean=[1.0] * n_features,
        residual_scale=[[1.0] * n_features for _ in range(n_nodes)],
        **kwargs,
    )


def test_stid_email_multifeature_and_mooc_direct_h24_gradients():
    torch.manual_seed(123)
    for history, hmax, n_nodes, n_features in ((28, 7, 9, 2), (24, 24, 7, 1)):
        model = _model(history=history, hmax=hmax, n_nodes=n_nodes, n_features=n_features)
        x = torch.randn(2, history, n_nodes, n_features)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 2, 1, 0] = False
        result = model(x, mask=mask, age=torch.zeros_like(x), return_trace=True)
        assert result["predictions"].shape == (2, hmax, n_nodes, n_features)
        assert torch.isfinite(result["predictions"]).all()
        assert (result["predictions"] >= 0).all()
        assert not any("time_in_day_emb" in n or "day_in_week_emb" in n
                       for n, _ in model.named_parameters())
        result["predictions"].mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.network.node_emb.grad is not None


def test_stid_batch_one_986_nodes_cpu_forward():
    model = _model(history=28, hmax=7, n_nodes=986, n_features=2).eval()
    x = torch.zeros(1, 28, 986, 2)
    with torch.no_grad():
        result = model(x)
    assert result["predictions"].shape == (1, 7, 986, 2)
    assert torch.isfinite(result["predictions"]).all()
