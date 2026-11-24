"""Factory-level CPU tests for the integrated AGCRN checkpoint opt-in."""
import copy
import io

import pytest
import torch

from edsgn.baselines import build_baseline
import edsgn.agcrn_checkpoint as checkpoint_impl


torch.set_num_threads(2)


def _build(features, count_data, checkpoint_cells):
    return build_baseline(
        "AGCRN",
        n_nodes=5,
        n_features=features,
        history=6,
        hmax=3,
        count_data=count_data,
        norm_mean=[0.1] * features,
        norm_std=[1.2] * features,
        output_scale=[0.8] * features,
        output_mean=[0.6] * features,
        hidden_dim=8,
        embed_dim=3,
        num_layers=2,
        cheb_k=2,
        training_checkpoint_cells=checkpoint_cells,
    )


@pytest.mark.parametrize("features,count_data", [(1, False), (2, False), (1, True), (2, True)])
def test_factory_checkpoint_optin_matches_predictions_inputs_and_all_grads(features, count_data):
    torch.manual_seed(1901 + features + int(count_data))
    reference = _build(features, count_data, False).train()
    checked = _build(features, count_data, True).train()
    checked.load_state_dict(copy.deepcopy(reference.state_dict()))
    keys = tuple(reference.state_dict().keys())
    x_ref = torch.randn(2, 6, 5, features, requires_grad=True)
    x_checked = x_ref.detach().clone().requires_grad_(True)

    expected = reference(x_ref)
    actual = checked(x_checked)
    assert torch.equal(actual["predictions"], expected["predictions"])
    assert torch.equal(actual["graph_loss"], expected["graph_loss"])
    assert actual["counters"] == expected["counters"]
    assert tuple(checked.state_dict().keys()) == keys
    expected["predictions"].square().mean().backward()
    actual["predictions"].square().mean().backward()
    assert torch.equal(x_checked.grad, x_ref.grad)
    ref_params = dict(reference.named_parameters())
    checked_params = dict(checked.named_parameters())
    assert ref_params.keys() == checked_params.keys()
    for name, param in ref_params.items():
        ref_grad = param.grad
        checked_grad = checked_params[name].grad
        assert (ref_grad is None) == (checked_grad is None), name
        if ref_grad is not None:
            assert torch.isfinite(ref_grad).all(), name
            assert torch.isfinite(checked_grad).all(), name
            assert torch.allclose(ref_grad, checked_grad, rtol=1e-6, atol=1e-7), name

    # The wrapper is ephemeral and adds no checkpoint keys; its trained state
    # reloads through the ordinary factory and produces the same eval output.
    payload = io.BytesIO()
    torch.save(checked.state_dict(), payload)
    payload.seek(0)
    restored = _build(features, count_data, True).eval()
    restored.load_state_dict(torch.load(payload, map_location="cpu", weights_only=True))
    reference.eval()
    with torch.no_grad():
        assert torch.equal(restored(x_ref.detach())["predictions"],
                           reference(x_ref.detach())["predictions"])
    assert tuple(restored.state_dict().keys()) == keys


def test_default_off_and_eval_no_grad_bypass_checkpoint(monkeypatch):
    calls = []
    actual_checkpoint = checkpoint_impl.checkpoint

    def observed(function, *args, **kwargs):
        calls.append(dict(kwargs))
        return actual_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(checkpoint_impl, "checkpoint", observed)
    default = build_baseline(
        "AGCRN", n_nodes=5, n_features=1, history=6, hmax=3,
        hidden_dim=8, embed_dim=3, num_layers=2, cheb_k=2,
    )
    assert default.training_checkpoint_cells is False
    enabled = _build(1, False, True)
    x = torch.randn(2, 6, 5, 1)
    default.train()(x)
    assert calls == []

    enabled.eval()(x)
    assert calls == []
    enabled.train()
    with torch.no_grad():
        enabled(x)
    assert calls == []

    enabled(x)
    assert len(calls) == 12  # L=6 times x two official recurrent cells.
    assert all(c["use_reentrant"] is False and c["preserve_rng_state"] is True for c in calls)

