"""CPU-only parity for the prospective external AGCRN cell checkpoint wrapper."""
import copy
import io
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "staging"))
sys.path.insert(0, str(ROOT / "src"))

import agcrn_checkpoint_adapter_v1 as checkpoint_adapter
from rp_forecast_adapters_v2 import AGCRNForecast


torch.set_num_threads(2)


def _model(features, count_data):
    return AGCRNForecast(
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
    ).cpu()


@pytest.mark.parametrize("features,count_data", [(1, False), (2, False), (1, True), (2, True)])
def test_checkpointed_official_cells_match_forward_input_and_all_parameter_grads(features, count_data):
    torch.manual_seed(841 + features + int(count_data))
    reference = _model(features, count_data).train()
    checkpointed = _model(features, count_data).train()
    checkpointed.load_state_dict(copy.deepcopy(reference.state_dict()))
    x_ref = torch.randn(2, 6, 5, features, requires_grad=True)
    x_ckpt = x_ref.detach().clone().requires_grad_(True)

    expected = reference(x_ref)
    state_keys = tuple(checkpointed.state_dict().keys())
    before_state = {k: v.detach().clone() for k, v in checkpointed.state_dict().items()}
    with checkpoint_adapter.agcrn_cell_checkpoint(checkpointed, enabled=True) as receipt:
        assert receipt.requested and receipt.cell_count == 2 and receipt.training_grad_guarded
        actual = checkpointed(x_ckpt)
        assert tuple(checkpointed.state_dict().keys()) == state_keys
        assert all(torch.equal(before_state[k], v) for k, v in checkpointed.state_dict().items())

    assert torch.equal(actual["predictions"], expected["predictions"])
    assert torch.equal(actual["graph_loss"], expected["graph_loss"])
    expected["predictions"].square().mean().backward()
    actual["predictions"].square().mean().backward()
    assert torch.equal(x_ref.grad, x_ckpt.grad)
    ref_params = dict(reference.named_parameters())
    ckpt_params = dict(checkpointed.named_parameters())
    assert ref_params.keys() == ckpt_params.keys()
    for name, ref_param in ref_params.items():
        grad_ref, grad_ckpt = ref_param.grad, ckpt_params[name].grad
        assert (grad_ref is None) == (grad_ckpt is None), name
        if grad_ref is not None:
            assert torch.isfinite(grad_ref).all(), name
            assert torch.isfinite(grad_ckpt).all(), name
            assert torch.allclose(grad_ref, grad_ckpt, rtol=1e-6, atol=1e-7), name
    assert tuple(checkpointed.state_dict().keys()) == state_keys
    restored = _model(features, count_data).cpu()
    restored.load_state_dict(copy.deepcopy(checkpointed.state_dict()))
    reference.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.equal(restored(x_ref.detach())["predictions"],
                           reference(x_ref.detach())["predictions"])


def test_checkpoint_guard_skips_eval_and_no_grad(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint should be bypassed")

    monkeypatch.setattr(checkpoint_adapter, "checkpoint", forbidden)
    model = _model(2, False)
    x = torch.randn(2, 6, 5, 2)
    with checkpoint_adapter.agcrn_cell_checkpoint(model, enabled=True):
        model.eval()
        # Eval bypasses checkpoint even when autograd is otherwise enabled.
        model(x)
        model.train()
        # Training mode alone is insufficient when gradients are disabled.
        with torch.no_grad():
            model(x)


def test_default_disabled_and_state_dict_reload_preserve_predictions():
    torch.manual_seed(129)
    original = _model(1, True).eval()
    keys = tuple(original.state_dict().keys())
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    reloaded = _model(1, True).eval()
    reloaded.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    assert tuple(reloaded.state_dict().keys()) == keys
    assert all(torch.equal(a, b) for a, b in zip(original.state_dict().values(), reloaded.state_dict().values()))
    x = torch.randn(2, 6, 5, 1)
    with torch.no_grad():
        expected = original(x)["predictions"]
        with checkpoint_adapter.agcrn_cell_checkpoint(reloaded):
            actual = reloaded(x)["predictions"]
    assert torch.equal(actual, expected)


def test_wrapper_restores_original_forward_after_exception():
    model = _model(1, False).train()
    cell = model.network.encoder.dcrnn_cells[0]
    original = cell.forward
    with pytest.raises(RuntimeError, match="sentinel"):
        with checkpoint_adapter.agcrn_cell_checkpoint(model, enabled=True):
            raise RuntimeError("sentinel")
    assert "forward" not in cell.__dict__
    assert cell.forward.__func__ is original.__func__
