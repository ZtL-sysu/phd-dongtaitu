"""CPU-only toy tests for the staged frozen graph/state diagnostic helpers."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "staging"))
sys.path.insert(0, str(ROOT / "src"))

import frozen_graph_state_diagnostic_v1 as diagnostic
from edsgn.model import ForecastModel, ModelConfig
from edsgn.scheduling import Policy


def test_row_tv_and_alignment_are_raw_set_diagnostics():
    assert diagnostic.row_total_variation(None, None, [2, 3], [.5, .5], 4) is None
    tv = diagnostic.row_total_variation([1, 2], [.75, .25], [2, 3], [.5, .5], 4)
    assert tv == pytest.approx(.75)
    assert diagnostic.alignment_counts([1, 2], np.array([0., .3, .7, 0.])) == {
        "candidate_edge_count": 2, "oracle_edge_count": 2,
        "intersection_count": 2, "symmetric_difference_count": 0,
    }


def test_oracle_transition_timeline_covers_abrupt_and_interpolated_changes():
    before = np.array([[0., 1.], [1., 0.]])
    after = np.array([[0., 1.], [0., 1.]])
    abrupt, alpha = diagnostic.expected_transition({"case": "identity", "switch_index": 4}, before, after, 4)
    assert alpha == 1 and np.array_equal(abrupt, after)
    gradual, alpha = diagnostic.expected_transition(
        {"case": "gradual", "switch_index": 4, "gradual_length": 4}, before, after, 5)
    assert alpha == pytest.approx(.5)
    assert np.allclose(gradual, .5 * before + .5 * after)


def test_gate_requires_p3a_p4_and_clean_resource_budget():
    diagnostic.check_stage_gate({"stages": [{"id": "P3A", "status": "passed"},
                                             {"id": "P4", "status": "running"}]})
    with pytest.raises(RuntimeError, match="P3A passed"):
        diagnostic.check_stage_gate({"stages": [{"id": "P3A", "status": "pending"},
                                                 {"id": "P4", "status": "running"}]})
    diagnostic.check_resource_gate(locks_clear=True, owned_jobs=[], spent_seconds=1, budget_seconds=2)
    with pytest.raises(RuntimeError, match="lease"):
        diagnostic.check_resource_gate(locks_clear=False, owned_jobs=[], spent_seconds=0, budget_seconds=2)
    with pytest.raises(RuntimeError, match="process"):
        diagnostic.check_resource_gate(locks_clear=True, owned_jobs=[{"pid": 9}], spent_seconds=0, budget_seconds=2)
    with pytest.raises(RuntimeError, match="exhausted"):
        diagnostic.check_resource_gate(locks_clear=True, owned_jobs=[], spent_seconds=2, budget_seconds=2)


def test_collection_fingerprint_binds_test_bytes_and_oracle_bytes_without_npz_decode(tmp_path, monkeypatch):
    data = tmp_path / "toy"
    (data / "oracle").mkdir(parents=True)
    (data / "test_seed200.npz").write_bytes(b"toy-trajectory-bytes")
    (data / "oracle" / "test_seed200.oracle.npz").write_bytes(b"toy-oracle-bytes")
    (data / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    first = diagnostic.collection_fingerprint(data)
    assert set(first) == {"test_seed200.npz", "oracle/test_seed200.oracle.npz", "manifest.json"}
    (data / "oracle" / "test_seed200.oracle.npz").write_bytes(b"modified-oracle-bytes")
    assert diagnostic.collection_fingerprint(data) != first


def test_passive_capture_records_lif_g_and_ages_without_changing_forward():
    torch.manual_seed(113)
    model = ForecastModel(ModelConfig(n_nodes=4, n_features=1, history=3, hmax=2,
                                      count_data=False, variant="V4", d_total=16,
                                      encoder_width=8, readout_hidden=8, k=2,
                                      candidate_cap=4))
    model.eval()
    x = torch.tensor([[[[.2], [.1], [-.3], [.4]],
                       [[.3], [.15], [-.1], [.2]],
                       [[.1], [.3], [.2], [-.2]]]], dtype=torch.float32)
    mask = torch.ones_like(x, dtype=torch.bool)
    age = torch.zeros_like(x)
    raw = x.clone()
    originals = {name: getattr(model, name) for name in ("advance_clock", "refresh_graph", "update_states")}
    with diagnostic.TraceCapture(model) as capture:
        result = model(x, mask=mask, age=age, raw=raw, policy=Policy(), return_trace=True)
    plain = model(x, mask=mask, age=age, raw=raw, policy=Policy(), return_trace=True)
    assert len(capture.rows) == len(result["trace"]) == 3
    assert result["predictions"].shape == (1, 2, 4, 1)
    assert capture.rows[0]["u_before_clock"] == capture.rows[0]["u_after_clock"]
    assert capture.rows[-1]["state_pending_log_g_after_update"] == [0.0] * len(capture.rows[-1]["state_ids"])
    assert capture.rows[0]["row_tv"] is not None and all(v is None for v in capture.rows[0]["row_tv"])
    for row in capture.rows:
        assert np.allclose(row["state_g_used"], np.exp(row["state_pending_log_g_before_update"]))
        assert np.allclose(row["state_pending_log_g_after_update"], 0.0)
    assert all(getattr(model, name).__func__ is method.__func__ for name, method in originals.items())
    assert torch.equal(result["predictions"], plain["predictions"])
    assert torch.equal(result["graph_loss"], plain["graph_loss"])
    assert torch.equal(result["runtime"].u, plain["runtime"].u)
    assert "c" in result["trace"][1] and "pending_log_g_before_state" in result["trace"][1]
