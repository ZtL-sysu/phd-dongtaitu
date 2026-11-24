"""CPU-only toy tests for the staged frozen graph/state diagnostic helpers."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn" / "staging"))
sys.path.insert(0, str(ROOT / "edsgn" / "src"))

import frozen_graph_state_diagnostic_v3 as diagnostic
from edsgn.model import ForecastModel, ModelConfig
from edsgn.scheduling import Policy


def test_row_tv_and_alignment_are_raw_set_diagnostics():
    assert diagnostic.row_total_variation(None, None, [2, 3], [.5, .5], 4) is None
    tv = diagnostic.row_total_variation([1, 2], [.75, .25], [2, 3], [.5, .5], 4)
    assert tv == pytest.approx(.75)
    with pytest.raises(ValueError, match="in-range"):
        diagnostic.row_total_variation([1, -1], [.5, .5], [2, 3], [.5, .5], 4)
    with pytest.raises(ValueError, match="duplicate"):
        diagnostic.row_total_variation([1, 1], [.5, .5], [2, 3], [.5, .5], 4)
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
    diagnostic.check_stage_gate({"stages": [{"id": "P2", "status": "passed", "gate_verdict":"pass"},
                                             {"id": "P3", "status": "passed", "gate_verdict":"pass"},
                                             {"id": "P3A", "status": "passed", "gate_verdict":"pass", "evidence":[{"path":"inputs/implementation-plan.md","sha256":diagnostic.sha256(diagnostic.ROOT / "inputs/implementation-plan.md")} ]},
                                             {"id": "P4", "status": "running"}]})
    with pytest.raises(RuntimeError, match="P3A passed"):
        diagnostic.check_stage_gate({"stages": [{"id": "P2", "status": "passed", "gate_verdict":"pass"},
                                                 {"id": "P3", "status": "passed", "gate_verdict":"pass"},
                                                 {"id": "P3A", "status": "pending", "gate_verdict":"pass", "evidence":[{"path":"proof","sha256":"x"}]},
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
    with diagnostic.TraceCapture(model, detailed=True, detailed_node_ids=range(4)) as capture:
        result = model(x, mask=mask, age=age, raw=raw, policy=Policy(), return_trace=True)
    plain = model(x, mask=mask, age=age, raw=raw, policy=Policy(), return_trace=True)
    assert len(capture.rows) == len(result["trace"]) == 3
    assert result["predictions"].shape == (1, 2, 4, 1)
    assert capture.rows[0]["u_before_clock"] == capture.rows[0]["u_after_clock"]
    assert capture.rows[-1]["state_detailed"]["pending_log_g_after_update"] == [0.0] * len(capture.rows[-1]["state_detailed"]["node_ids"])
    assert capture.rows[0]["graph_rows"] and all(v["row_tv"] is None for v in capture.rows[0]["graph_rows"])
    for row in capture.rows:
        assert np.allclose([r["g_used"] for r in row["state_rows"]],
                           np.exp([r["consumed_log_g"] for r in row["state_rows"]]))
        assert np.allclose(row["state_detailed"]["pending_log_g_after_update"], 0.0)
    assert all(getattr(model, name).__func__ is method.__func__ for name, method in originals.items())
    assert torch.equal(result["predictions"], plain["predictions"])
    assert torch.equal(result["graph_loss"], plain["graph_loss"])
    assert torch.equal(result["runtime"].u, plain["runtime"].u)
    assert "c" in result["trace"][1] and "pending_log_g_before_state" in result["trace"][1]


def test_fixed_detailed_selection_is_uniform_deterministic_and_freezable():
    origins = list(range(100, 200))
    chosen = diagnostic.uniform_detailed_origins(origins)
    assert len(chosen) == 8 and chosen[0] == 100 and chosen[-1] == 199
    assert chosen == diagnostic.uniform_detailed_origins(origins)
    selection = diagnostic.expected_detailed_selection([origins, [7, 9, 12]], [64, 4])
    assert selection == [
        {"dataset_index": 0, "detailed_origins": chosen, "detailed_node_ids": list(range(8))},
        {"dataset_index": 1, "detailed_origins": [7, 9, 12], "detailed_node_ids": [0, 1, 2, 3]},
    ]
    assert diagnostic.trajectory_seed_ids_for_origins([12, 29], [(1, 9), (0, 10), (1, 12)]).tolist() == [29, 12, 29]


def test_artifact_preflight_is_conservative_and_fails_closed():
    estimate = diagnostic.estimate_artifact_bytes(20, 6, 64, 8, 20 * 6 * 64)
    assert estimate["estimated_bytes"] > estimate["compact_trace_bytes"]
    with pytest.raises(RuntimeError, match="exceeds cap"):
        diagnostic.estimate_artifact_bytes(1000, 28, 64, 32, 1000000, max_artifact_gib=.01)
    with pytest.raises(RuntimeError, match="Insufficient free disk"):
        diagnostic.check_disk_capacity(10, {"estimated_bytes": 100})
    assert diagnostic.check_disk_capacity(200, {"estimated_bytes": 100}) == 125
    with pytest.raises(ValueError, match="safety factor"):
        diagnostic.check_disk_capacity(200, {"estimated_bytes": 100}, safety_factor=.9)
    with pytest.raises(ValueError, match="Invalid artifact"):
        diagnostic.estimate_artifact_bytes(-1, 2, 4, 1, 2)
    with pytest.raises(ValueError, match="Invalid artifact"):
        diagnostic.estimate_artifact_bytes(1, 2, 4, 1, 2, max_artifact_gib=float("inf"))


def test_compact_trace_keeps_full_origin_c_tv_g_without_full_membrane_arrays():
    torch.manual_seed(17)
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
    with diagnostic.TraceCapture(model, detailed=False) as compact:
        model(x, mask=mask, age=age, raw=raw, policy=Policy(), return_trace=True)
    row = compact.rows[0]
    assert len(row["graph_rows"]) > 0
    assert all({"c", "row_tv", "node_id", "row_id"} <= set(item) for item in row["graph_rows"])
    assert "u_before_clock" not in row and "spike_pending_before_clock" not in row
    assert "old_support_ids" not in row["graph_rows"][0]
    with diagnostic.TraceCapture(model, detailed=True, detailed_node_ids=[0, 2]) as detailed:
        model(x, mask=mask, age=age, raw=raw, policy=Policy(), return_trace=True)
    drow = detailed.rows[0]
    assert drow["detailed_node_ids"] == [0, 2]
    assert len(drow["u_before_clock"]) == 2
    assert all(item.get("node_id") not in {1, 3} or "new_support_ids" not in item
               for item in drow["graph_rows"])


def test_final_collection_fingerprint_and_resource_helper_hashes_detect_drift(tmp_path, monkeypatch):
    diagnostic.assert_fingerprint_unchanged({"a": "1"}, {"a": "1"})
    with pytest.raises(RuntimeError, match="changed during diagnostic"):
        diagnostic.assert_fingerprint_unchanged({"a": "1"}, {"a": "2"})
    run_root = tmp_path / "run"
    helper = run_root / "edsgn" / "scripts" / "guard.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"guard-v1")
    monkeypatch.setattr(diagnostic, "ROOT", run_root)
    monkeypatch.setattr(diagnostic, "RESOURCE_HELPERS", (helper,))
    bound = diagnostic.resource_helper_hashes()
    diagnostic.verify_resource_helper_hashes(bound)
    helper.write_bytes(b"guard-v2")
    with pytest.raises(RuntimeError, match="helper implementation changed"):
        diagnostic.verify_resource_helper_hashes(bound)


def test_freeze_extras_require_trace_selection_and_bind_helper_hashes(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    data = run_root / "toy"
    (data / "oracle").mkdir(parents=True)
    (data / "test_seed4.npz").write_bytes(b"test")
    (data / "oracle" / "test_seed4.oracle.npz").write_bytes(b"oracle")
    helper = run_root / "edsgn" / "scripts" / "guard.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"guard")
    (run_root / "edsgn" / "scripts" / "frozen_synthetic_stress.py").write_bytes(b"normalizer")
    (run_root / "edsgn" / "src" / "edsgn").mkdir(parents=True)
    (run_root / "edsgn" / "src" / "edsgn" / "model.py").write_bytes(b"model")
    monkeypatch.setattr(diagnostic, "ROOT", run_root)
    monkeypatch.setattr(diagnostic, "RESOURCE_HELPERS", (helper,))
    with pytest.raises(ValueError, match="selection"):
        diagnostic.frozen_diagnostic_extras([data], None)
    selection = [{"dataset_index": 0, "detailed_origins": [10], "detailed_node_ids": [0]}]
    extras = diagnostic.frozen_diagnostic_extras([data], selection)
    assert extras["frozen_graph_state_trace_selections"] == selection
    assert list(extras["frozen_graph_state_resource_helper_sha256"]) == ["edsgn/scripts/guard.py"]
    assert extras["frozen_graph_state_normalization_helper_sha256"] == diagnostic.sha256(run_root / "edsgn/scripts/frozen_synthetic_stress.py")


def test_scalar_model_seed_and_trajectory_seed_roundtrip_statistics_loader(tmp_path):
    from edsgn.statistics import _load_prediction
    path = tmp_path / "pred.npz"
    np.savez(path, predictions=np.zeros((2, 7, 3, 2)), targets=np.zeros((2, 7, 3, 2)),
             target_mask=np.ones((2, 7, 3, 2), dtype=bool),
             origins=np.array([[0, 200], [1, 201]], dtype=np.int64),
             seed_id=np.asarray(diagnostic.model_seed_id({"seed": 3}), dtype=np.int64),
             trajectory_seed_id=diagnostic.trajectory_seed_ids_for_origins([200, 203], [(0,200),(1,201)]))
    loaded = _load_prediction(path)
    assert loaded["seed_id"] == 3
    with np.load(path, allow_pickle=False) as archive:
        assert archive["trajectory_seed_id"].tolist() == [200, 203]


def test_spike_pending_clock_capture_keeps_all_node_layer_channel_dimensions():
    torch.manual_seed(911)
    model = ForecastModel(ModelConfig(n_nodes=4, n_features=1, history=3, hmax=2,
                                      count_data=False, variant="V4", d_total=16,
                                      encoder_width=8, readout_hidden=8, k=2, candidate_cap=4))
    model.eval()
    x = torch.zeros((1, 3, 4, 1)); mask = torch.ones_like(x, dtype=torch.bool)
    age = torch.zeros_like(x)
    with diagnostic.TraceCapture(model, detailed=True, detailed_node_ids=[0, 2]) as capture:
        out = model(x, mask=mask, age=age, raw=x, policy=Policy(), return_trace=True)
    before = np.asarray(capture.rows[0]["spike_pending_before_clock"])
    after = np.asarray(capture.rows[0]["spike_pending_after_clock"])
    assert before.shape == after.shape == (2, 2, 2, 4)
    # Per-node, both fast/slow layers and every channel are represented. The
    # recorded first pre-clock state must align with the full runtime layout.
    expected_shape = (2, 2, 2, 4)
    assert before.shape == expected_shape
    assert np.asarray(capture.rows[-1]["spike_pending_after_clock"]).shape == expected_shape


def test_compact_trace_fields_null_c_and_artifact_hmax_accounting():
    step = {"quality": np.array([.2,.4]), "residual": np.array([.5,1.]),
            "graph_ids": [], "state_ids": [0], "mandatory_graph": [False, True],
            "mandatory_state": [True, False], "graph_quota_overrun": 1,
            "state_quota_overrun": 0, "estimated_cost": 3.,
            "estimated_mandatory_cost": 2., "runtime_budget_overrun_ms": 0.,
            "component_host_ms": {"monitor_ms": 1.}, "c": None}
    required = {"quality", "residual", "mandatory_graph", "mandatory_state", "graph_quota_overrun",
                "state_quota_overrun", "estimated_cost", "estimated_mandatory_cost",
                "runtime_budget_overrun_ms", "component_host_ms", "c"}
    assert required <= set(step)
    assert step["c"] is None  # no graph refresh is a null signal, never fabricated zero
    small = diagnostic.estimate_artifact_bytes(2, 3, 4, 1, 2*9*4*2, d_total=16)
    assert small["array_bytes"] >= 2*9*4*2*40
    assert small["compact_trace_bytes"] >= 2*3*10*1024


def test_gate_requires_canonical_gate_verdict_and_hash_bound_evidence(tmp_path, monkeypatch):
    evidence = tmp_path / "proof.md"; evidence.write_text("proof", encoding="utf-8")
    import hashlib
    row = {"id":"P3A", "status":"passed", "gate_verdict":"pass",
           "evidence":[{"path":"proof.md", "sha256":hashlib.sha256(evidence.read_bytes()).hexdigest()}]}
    state = {"stages":[{"id":"P2","status":"passed","gate_verdict":"pass"},
             {"id":"P3","status":"passed","gate_verdict":"pass"}, row,
             {"id":"P4","status":"running"}]}
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    diagnostic.check_stage_gate(state)
    bad = {"stages":[{"id":"P2","status":"passed","gate_verdict":"pass"},
           {"id":"P3","status":"passed","gate_verdict":"pass"},
           {"id":"P3A","status":"passed","gate_verdict":"fail","evidence":row["evidence"]},
           {"id":"P4","status":"running"}]}
    with pytest.raises(RuntimeError, match="canonical P3A"):
        diagnostic.check_stage_gate(bad)
    row["evidence"][0]["sha256"] = "0"*64
    with pytest.raises(RuntimeError, match="hash mismatch"):
        diagnostic.check_stage_gate(state)



def test_normalization_helper_hash_is_fail_closed(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    helper = run_root / "edsgn" / "scripts" / "frozen_synthetic_stress.py"
    helper.parent.mkdir(parents=True); helper.write_bytes(b"normalizer-v1")
    monkeypatch.setattr(diagnostic, "ROOT", run_root)
    bound = diagnostic.normalization_helper_hash()
    diagnostic.verify_normalization_helper_hash(bound)
    helper.write_bytes(b"normalizer-v2")
    with pytest.raises(RuntimeError, match="normalization helper implementation changed"):
        diagnostic.verify_normalization_helper_hash(bound)
    with pytest.raises(RuntimeError, match="normalization helper implementation changed"):
        diagnostic.verify_normalization_helper_hash(None)
