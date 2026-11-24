import hashlib
import json
from pathlib import Path

import pytest

import prospective_training_candidate_binding_v1 as cb


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(root, path):
    return {"path": Path(path).resolve().relative_to(root.resolve()).as_posix(), "sha256": digest(path)}


def putjson(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


@pytest.fixture
def prospective(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    # Invalid archive bytes prove the prospective path hashes sources without decoding arrays.
    data = root / "datasets" / "toy.npz"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"metadata-only opaque archive bytes")
    code = putjson(root / "fake-runtime.py", {"code": "toy"})
    monkeypatch.setattr(cb._binding, "ROOT", root)
    monkeypatch.setattr(cb._binding, "_code_paths", lambda variant: [code])

    candidate_rows = []
    for seed in range(5):
        cfg = putjson(root / f"configs/candidate-a-seed{seed}.json", {
            "variant": "V4", "seed": seed, "dataset_path": str(data),
            "output_dir": str(root / f"runs/candidate-a/seed{seed}")})
        config_ref = ref(root, cfg)
        blueprint_ref = cb.write_input_blueprint(root, config_ref, comparison_id="cmp-1",
                                                   candidate_id="g00-m00", seed_id=seed,
                                                   dataset_id="synthetic",
                                                   output_path=f"protocols/blueprints/seed{seed}.json")
        candidate_rows.append({"seed_id": seed, "config": config_ref,
                               "input_blueprint": blueprint_ref,
                               "training_binding_path": f"runs/candidate-a/seed{seed}/input-binding-at-launch.json",
                               "selection_receipt_path": f"runs/evidence/selection-seed{seed}.json"})
    selector = root / "protocols/selector.py"
    selector.parent.mkdir(parents=True, exist_ok=True)
    selector.write_bytes(Path(cb.__file__).read_bytes())
    pool = {"schema_version": 1, "scope": "prospective_training_candidate_pool", "comparison_id": "cmp-1",
            "dataset_id": "synthetic", "variant": "V4",
            "selector": {"rule_id": "normalized-syn-mooc-v4-seed0-v1", "source": ref(root, selector)},
            "candidates": [{"candidate_id": "g00-m00", "graph_candidate_id": "g00",
                            "memory_candidate_id": "m00", "seeds": candidate_rows}]}
    pool_path = putjson(root / "protocols/candidate-pool.json", pool)
    prep = {"schema_version": 4, "candidate_pool": ref(root, pool_path), "comparison_id": "cmp-1"}
    return root, prep, pool, candidate_rows


def test_preparation_binds_complete_static_inputs_without_archive_decode(prospective):
    root, prep, _, _ = prospective
    consumed = cb.validate_preparation(root, prep, 0)
    assert "datasets/toy.npz" in consumed
    assert "fake-runtime.py" in consumed
    assert "configs/candidate-a-seed0.json" in consumed


def test_candidate_pool_config_variant_is_distinct_from_v4_rank_reference(prospective):
    root, prep, pool, rows = prospective
    pool["variant"] = "V2"
    for seed, row in enumerate(rows):
        cfgpath = root / row["config"]["path"]
        cfgdoc = json.loads(cfgpath.read_text())
        cfgdoc["variant"] = "V2"
        putjson(cfgpath, cfgdoc)
        row["config"] = ref(root, cfgpath)
        row["input_blueprint"] = cb.write_input_blueprint(
            root, row["config"], comparison_id="cmp-1", candidate_id="g00-m00", seed_id=seed,
            dataset_id="synthetic", output_path=f"protocols/v2-blueprints/seed{seed}.json")
    pool_path = putjson(root / "protocols/v2-candidate-pool.json", pool)
    prep = {**prep, "candidate_pool": ref(root, pool_path)}
    assert "protocols/v2-blueprints/seed0.json" in cb.validate_preparation(root, prep, 0)


def test_pool_tamper_or_config_drift_rejected(prospective):
    root, prep, _, rows = prospective
    config = root / rows[0]["config"]["path"]
    config.write_text(config.read_text() + " ", encoding="utf-8")
    with pytest.raises(cb.CandidateBindingError, match="hash-mismatched|differs from current static"):
        cb.validate_preparation(root, prep, 0)


def test_binding_path_must_match_config_output_dir(prospective):
    root, prep, pool, _ = prospective
    pool["candidates"][0]["seeds"][0]["training_binding_path"] = "runs/wrong/input-binding-at-launch.json"
    p = putjson(root / "protocols/pool-wrong-binding-path.json", pool)
    bad = {**prep, "candidate_pool": ref(root, p)}
    with pytest.raises(cb.CandidateBindingError, match="config.output_dir"):
        cb.validate_preparation(root, bad, 0)


def test_candidate_bound_is_12_and_joint_pairs_unique(prospective):
    root, prep, pool, _ = prospective
    pool["candidates"] = [pool["candidates"][0]] * 13
    p = putjson(root / "protocols/too-many.json", pool)
    bad = {**prep, "candidate_pool": ref(root, p)}
    with pytest.raises(cb.CandidateBindingError, match="1..12"):
        cb.validate_preparation(root, bad, 0)


def test_selector_rule_and_source_are_frozen(prospective):
    root, prep, pool, _ = prospective
    pool["selector"]["rule_id"] = "arbitrary"
    p = putjson(root / "protocols/pool-wrong-selector.json", pool)
    bad = {**prep, "candidate_pool": ref(root, p)}
    with pytest.raises(cb.CandidateBindingError, match="frozen normalized"):
        cb.validate_preparation(root, bad, 0)


def test_validation_rank_uses_normalized_equal_mean_then_wall_then_id():
    pool = {"candidates": [{"candidate_id": "b"}, {"candidate_id": "a"}, {"candidate_id": "failed"}]}
    refs = {"synthetic": 2.0, "mooc": 4.0}
    rows = [
        {"candidate_id": "b", "status": "completed", "validation": {
            "synthetic": {"required_horizon_mae": 1.0, "wall_seconds": 4.0},
            "mooc": {"required_horizon_mae": 2.0, "wall_seconds": 5.0}}},
        {"candidate_id": "a", "status": "completed", "validation": {
            "synthetic": {"required_horizon_mae": 1.0, "wall_seconds": 3.0},
            "mooc": {"required_horizon_mae": 2.0, "wall_seconds": 4.0}}},
        {"candidate_id": "failed", "status": "failed", "validation": None},
    ]
    assert cb.select_validation_candidate(pool, rows, refs)["selected_candidate_id"] == "a"


def test_selection_rejects_incomplete_ledger_or_invalid_gru_reference():
    pool = {"candidates": [{"candidate_id": "a"}, {"candidate_id": "b"}]}
    one = [{"candidate_id": cid, "status": "completed", "validation": {
        "synthetic": {"required_horizon_mae": 1.0, "wall_seconds": 1.0},
        "mooc": {"required_horizon_mae": 1.0, "wall_seconds": 1.0}}} for cid in ("a", "b")]
    with pytest.raises(cb.CandidateBindingError, match="every candidate"):
        cb.select_validation_candidate(pool, one[:1], {"synthetic": 1.0, "mooc": 1.0})
    with pytest.raises(cb.CandidateBindingError, match="positive GRU"):
        cb.select_validation_candidate(pool, one, {"synthetic": 0.0, "mooc": 1.0})


def test_actual_at_launch_binding_matches_blueprint_and_selection_receipt(prospective, monkeypatch):
    root, prep, pool, rows = prospective
    seed = 0
    spec = rows[seed]
    config_path = root / spec["config"]["path"]
    outdir = root / "runs/candidate-a/seed0"
    actual_binding = cb._binding.prepare_binding(config_path, output_dir=outdir)
    def evidence(prefix):
        roles = ("pilot_config", "input_binding", "phase_receipt", "validation_archive",
                 "independent_numeric_review", "root_acceptance")
        refs = {}
        for role in roles:
            path = putjson(root / f"evidence/{prefix}-{role}.json", {"role": role})
            refs[role] = ref(root, path)
        return refs
    ev = evidence("candidate")
    def regime(mae, wall, horizons):
        return {"variant": "V4", "model_seed": 0, "split": "validation", "required_horizons": horizons,
                "required_horizon_mae": mae, "wall_seconds": wall, "evidence": ev}
    candidates = [{"candidate_id": "g00-m00", "status": "completed", "validation": {
        "synthetic": regime(0.5, 12.0, [1, 3, 7]), "mooc": regime(0.4, 14.0, [1, 6, 24])},
        "email_descriptive_status": "all-zero validation; descriptive only"}]
    gru_syn_num = putjson(root / "evidence/gru-syn-numeric.json", {"result": "accepted numeric"})
    gru_syn_root = putjson(root / "evidence/gru-syn-root.json", {"accepted": True})
    gru_mooc_num = putjson(root / "evidence/gru-mooc-numeric.json", {"result": "accepted numeric"})
    gru_mooc_root = putjson(root / "evidence/gru-mooc-root.json", {"accepted": True})
    receipt = {"schema_version": 1, "scope": "root_validation_only_candidate_selection", "status": "accepted",
              "accepted_by": "root", "test_feedback_used": False, "comparison_id": "cmp-1",
              "pool": prep["candidate_pool"], "selector": pool["selector"], "candidates": candidates,
              "selected_candidate_id": "g00-m00", "gru_reference": {
                  "synthetic": {"mae": 1.0, "required_horizons": [1, 3, 7], "evidence": {"numeric_review": ref(root, gru_syn_num), "root_acceptance": ref(root, gru_syn_root)}},
                  "mooc": {"mae": 1.0, "required_horizons": [1, 6, 24], "evidence": {"numeric_review": ref(root, gru_mooc_num), "root_acceptance": ref(root, gru_mooc_root)}}}}
    receipt_path = putjson(root / spec["selection_receipt_path"], receipt)
    runrow = {"candidate_selection": ref(root, receipt_path), "config": spec["config"],
              "training_binding": ref(root, actual_binding)}
    consumed = cb.validate_selected_training(root, prep, seed, runrow)
    assert "runs/candidate-a/seed0/input-binding-at-launch.json" in consumed
    assert "runs/evidence/selection-seed0.json" in consumed
    assert "evidence/candidate-independent_numeric_review.json" in consumed
    assert "evidence/gru-syn-root.json" in consumed
    valid_receipt = json.loads(receipt_path.read_text())
    wrong_domain = json.loads(json.dumps(valid_receipt))
    wrong_domain["candidates"][0]["validation"]["mooc"]["required_horizons"] = [1, 3, 7]
    putjson(receipt_path, wrong_domain)
    runrow["candidate_selection"] = ref(root, receipt_path)
    with pytest.raises(cb.CandidateBindingError, match="mooc required_horizons"):
        cb.validate_selected_training(root, prep, seed, runrow)
    putjson(receipt_path, valid_receipt)
    runrow["candidate_selection"] = ref(root, receipt_path)
    with pytest.raises(cb.CandidateBindingError, match="prospective output already exists"):
        cb.validate_preparation(root, prep, seed)
    assert cb.validate_preparation(root, prep, seed, allow_postrun_outputs=True)
    binddoc = json.loads(actual_binding.read_text())
    binddoc["recorded_utc"] = "not-a-timestamp"
    putjson(actual_binding, binddoc)
    runrow["training_binding"] = ref(root, actual_binding)
    with pytest.raises(cb.CandidateBindingError, match="ISO-8601"):
        cb.validate_selected_training(root, prep, seed, runrow)


def test_selection_receipt_cannot_claim_email_rank_or_wrong_selected_candidate(prospective):
    root, prep, pool, rows = prospective
    spec = rows[0]
    # No actual binding is needed: malformed receipt must fail first.
    receipt = {"schema_version": 1, "scope": "root_validation_only_candidate_selection", "status": "accepted",
              "accepted_by": "root", "test_feedback_used": False, "comparison_id": "cmp-1",
              "pool": prep["candidate_pool"], "selector": pool["selector"],
              "candidates": [{"candidate_id": "g00-m00", "status": "failed", "validation": None,
                              "email_descriptive_status": "descriptive only", "reason": "failed pilot",
                              "evidence": [ref(root, putjson(root / "evidence/failure.json", {"reason": "fail"}))]}],
              "selected_candidate_id": "g00-m00", "gru_reference": {"synthetic": {}, "mooc": {}}}
    p = putjson(root / spec["selection_receipt_path"], receipt)
    row = {"candidate_selection": ref(root, p), "config": spec["config"],
           "training_binding": {"path": spec["training_binding_path"], "sha256": "0" * 64}}
    with pytest.raises(cb.CandidateBindingError, match="completed status"):
        cb.validate_selected_training(root, prep, 0, row)


def test_run_row_requires_exact_projection(prospective):
    root, prep, _, _ = prospective
    with pytest.raises(cb.CandidateBindingError, match="exactly candidate_selection"):
        cb.validate_selected_training(root, prep, 0, {"candidate_selection": {}, "config": {}, "training_binding": {}, "extra": 1})
