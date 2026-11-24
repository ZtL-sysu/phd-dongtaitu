import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn" / "staging"))
sys.path.insert(0, str(ROOT / "edsgn" / "src"))

import h2_matched_graph_retention_v1 as h2
from edsgn.model import ForecastModel, ModelConfig
from edsgn.scheduling import Policy


def make_model(seed):
    torch.manual_seed(seed)
    model = ForecastModel(ModelConfig(n_nodes=5, n_features=2, history=6, hmax=3,
                                      count_data=False, variant="V4", d_total=16,
                                      encoder_width=8, readout_hidden=8, k=2,
                                      candidate_cap=4, retention_mode="informative"))
    model.eval()
    return model


def make_inputs(seed):
    generator = torch.Generator().manual_seed(seed + 1000)
    x = torch.randn(2, 6, 5, 2, generator=generator) * .2
    mask = torch.ones_like(x, dtype=torch.bool)
    age = torch.zeros_like(x)
    raw = x.clone()
    return {"x": x, "mask": mask, "age": age, "raw": raw}


@pytest.mark.parametrize("seed", [3, 19, 41])
def test_matched_graph_arms_keep_relations_and_full_actions_and_informative_replays(seed):
    model = make_model(seed)
    inputs = make_inputs(seed)
    before = h2.parameter_sha256(model)
    original_refresh = model.refresh_graph
    with torch.no_grad():
        result = h2.run_matched_retention_probe(model, **inputs, policy=Policy(kind="full"))

    assert result["parameter_sha256_before"] == result["parameter_sha256_after"] == before
    assert result["informative_teacher_parity"] is True
    assert result["teacher_candidate_work_is_diagnostic_only"]
    assert "energy" not in result["timing_claim"] and "ordinary" not in result["timing_claim"]
    assert model.refresh_graph.__func__ is original_refresh.__func__

    teacher_rows = result["teacher_rows"]
    assert len(teacher_rows) == 6
    for arm_name in h2.ARMS:
        arm = result["arms"][arm_name]
        diagnostics = arm["candidate_diagnostics"]
        assert len(diagnostics) == len(teacher_rows)
        assert arm["candidate_graph_loss_is_discarded_scorer_diagnostic"]
        for t, (teacher, replay) in enumerate(zip(teacher_rows, diagnostics)):
            assert torch.equal(replay["row_ids"], teacher["row_ids"])
            assert torch.equal(replay["state_ids"], teacher["state_ids"])
            assert torch.equal(replay["matched_supports"], teacher["supports"])
            assert torch.equal(replay["matched_weights"], teacher["weights"])
            assert torch.equal(replay["matched_valid"], teacher["valid"])
            assert torch.equal(replay["teacher_c"], teacher["c"])
            # Full graph and full state are selected at every history step.
            assert replay["row_ids"].tolist() == list(range(10))
            assert replay["state_ids"].tolist() == list(range(10))
            trace = arm["result"]["trace"][t]
            assert torch.as_tensor(trace["pending_log_g_after_state"]).count_nonzero().item() == 0
    # Shuffled/constant/reset alter only the retention signal; each is nontrivial
    # for these fixed-seed, nonzero toy inputs.
    teacher_pred = result["teacher"]["predictions"]
    for arm_name in ("shuffled", "constant", "reset"):
        assert not torch.allclose(result["arms"][arm_name]["result"]["predictions"], teacher_pred,
                                  atol=1e-7, rtol=1e-7)


def test_first_refresh_uses_g_one_and_each_pending_signal_is_consumed_once():
    model = make_model(7)
    inputs = make_inputs(7)
    with torch.no_grad():
        result = h2.run_matched_retention_probe(model, **inputs, policy=Policy(kind="full"))
    for arm_name, arm in result["arms"].items():
        rows = arm["candidate_diagnostics"]
        assert not rows[0]["initialized_before"].any()
        assert torch.equal(rows[0]["prescribed_log_g"], torch.zeros_like(rows[0]["prescribed_log_g"]))
        assert rows[1]["initialized_before"].all()
        if arm_name == "constant":
            assert torch.allclose(rows[1]["prescribed_log_g"],
                                  torch.full_like(rows[1]["prescribed_log_g"], np.log(.9)))
        elif arm_name == "reset":
            assert torch.equal(rows[1]["prescribed_log_g"], torch.full_like(rows[1]["prescribed_log_g"], -80.))
        elif arm_name == "informative":
            expected = -model.beta * rows[1]["teacher_c"]
            assert torch.allclose(rows[1]["prescribed_log_g"], expected)
        elif arm_name == "shuffled":
            source = torch.roll(rows[1]["teacher_c"][:5], 1)
            expected = torch.cat([-model.beta * source, -model.beta * torch.roll(rows[1]["teacher_c"][5:], 1)])
            assert torch.allclose(rows[1]["prescribed_log_g"], expected)
        for step, trace in enumerate(arm["result"]["trace"]):
            before_update = torch.as_tensor(trace["pending_log_g_before_state"])
            after_update = torch.as_tensor(trace["pending_log_g_after_state"])
            assert before_update.shape == after_update.shape
            assert torch.count_nonzero(after_update).item() == 0


def test_rejects_wrong_teacher_descriptors_model_mode_gradient_and_partial_policy():
    model = make_model(13)
    inputs = make_inputs(13)
    with torch.no_grad():
        with pytest.raises(ValueError, match="descriptor"):
            h2.run_matched_retention_probe(model, **inputs, policy=Policy(kind="full"),
                descriptor=replace(h2.EXPECTED_DESCRIPTOR, oracle_accessed=True))
        with pytest.raises(ValueError, match="descriptor"):
            h2.run_matched_retention_probe(model, **inputs, policy=Policy(kind="full"),
                descriptor=replace(h2.EXPECTED_DESCRIPTOR, target_accessed=True))
        with pytest.raises(ValueError, match="full reference"):
            h2.run_matched_retention_probe(model, **inputs, policy=Policy(kind="periodic"))
    with pytest.raises(RuntimeError, match="disabled gradients"):
        h2.run_matched_retention_probe(model, **inputs, policy=Policy(kind="full"))
    with torch.no_grad():
        model.train()
        with pytest.raises(RuntimeError, match="eval mode"):
            h2.run_matched_retention_probe(model, **inputs, policy=Policy(kind="full"))
        model.eval()


def test_rejects_non_v4_nonspiking_and_nonretention_configs():
    inputs = make_inputs(23)
    for variant, retention_mode in (("V2", "informative"), ("V3", "informative"), ("V4", "none")):
        torch.manual_seed(23)
        model = ForecastModel(ModelConfig(n_nodes=5, n_features=2, history=6, hmax=3,
                                          count_data=False, variant=variant, d_total=16,
                                          encoder_width=8, readout_hidden=8, k=2,
                                          candidate_cap=4, retention_mode=retention_mode))
        model.eval()
        with torch.no_grad(), pytest.raises(ValueError, match="V4"):
            h2.run_matched_retention_probe(model, **inputs, policy=Policy(kind="full"))

