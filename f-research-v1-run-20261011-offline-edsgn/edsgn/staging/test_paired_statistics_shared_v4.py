import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "edsgn" / "staging" / "paired_statistics_shared_v4.py"
spec = importlib.util.spec_from_file_location("paired_statistics_shared_v4", MODULE_PATH)
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def _write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pipeline(root, *, p3a="running", p4="pending"):
    evidence = root / "p3a-evidence.md"
    evidence.write_text("accepted test fixture\n", encoding="utf-8")
    stages = []
    for key, status in (("P2", "passed"), ("P3", "passed"), ("P3A", p3a), ("P4", p4)):
        row = {"id": key, "status": status}
        if key in {"P2", "P3"}:
            row["gate_verdict"] = "pass"
        if key == "P3A" and p3a == "passed":
            row.update(gate_verdict="pass", evidence=[{"path": evidence.name, "sha256": _sha(evidence)}])
        stages.append(row)
    return _write_json(root / "pipeline-state.json", {"schema_version": 1, "stages": stages})


def _bind_loader(monkeypatch):
    def loader(prediction_path, spec, base):
        path = paired._resolve(base, prediction_path)
        item = paired.st._load_prediction(path)
        binding = paired._resolve(base, spec["input_binding_path"])
        src = {"prediction_path": str(path), "prediction_sha256": _sha(path),
               "binding_path": str(binding), "binding_sha256": _sha(binding),
               "binding_verification": {"status": "verified"}}
        paired._ARCHIVE_CACHE[(str(path), src["prediction_sha256"])] = (item, src)
        return item, src
    monkeypatch.setattr(paired, "_load_bound", loader)
    monkeypatch.setattr(paired, "_reverify_sources", lambda provenance: None)
    monkeypatch.setattr(paired, "_verify_at_launch_binding", lambda path: {"status": "verified"})
    def source_arrays(dataset_root):
        with np.load(dataset_root, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata_json"].item()))
            source_targets = np.asarray(z["targets"])
            source_masks = np.asarray(z["target_mask"], dtype=bool)
        span = metadata["splits"]["test"]
        first = max(int(metadata["history"]) - 1, int(span[0]) - 1)
        last = int(span[1]) - int(metadata["hmax"]) - 1
        pairs = np.asarray([[0, origin] for origin in range(first, last + 1)], dtype=np.int64)
        targets = np.asarray([source_targets[o+1:o+int(metadata["hmax"])+1] for _, o in pairs])
        masks = np.asarray([source_masks[o+1:o+int(metadata["hmax"])+1] for _, o in pairs])
        return pairs, targets, masks
    monkeypatch.setattr(paired, "_test_source_arrays", source_arrays)


def _fixture(root, *, split="validation", n_primary=2, include_secondary=True,
             gap=False, binding=True):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "dataset.npz"
    metadata = {"history": 2, "hmax": 2, "partition": "chronological",
                "splits": {"train": [0, 3], "validation": [3, 6], "test": [5, 8]}}
    source_targets = np.zeros((8, 2, 1), dtype=np.float64)
    source_masks = np.ones_like(source_targets, dtype=bool)
    source_masks[5, 1, 0] = False
    np.savez(source, metadata_json=np.asarray(json.dumps(metadata)), targets=source_targets,
             target_mask=source_masks)
    acf = _write_json(root / "acf.json", {"dependence_windows": 1, "split": "validation"})
    pipeline = _pipeline(root, p3a="running" if split == "validation" else "passed",
                         p4="pending" if split == "validation" else "running")
    data_root = source
    contrasts = []
    all_paths = []
    count = n_primary + int(include_secondary)
    target_pairs = np.array([[0, 4], [0, 5]]) if split == "test" else np.array([[0, 2], [0, 3]])
    if gap:
        target_pairs[1, 1] += 1
    targets = np.zeros((len(target_pairs), 2, 2, 1), dtype=np.float64)
    mask = np.ones_like(targets, dtype=bool)
    mask[0, 0, 1, 0] = False
    from edsgn.engine import dependency_manifest, source_manifest, dataset_fingerprint
    from edsgn.scheduling import Policy
    policy = dataclasses.asdict(Policy())
    policy_digest = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for ci in range(count):
        row = {"name": f"contrast-{ci}", "claim": "H1" if ci < n_primary else "H2",
               "family": "H1" if ci < n_primary else "secondary", "primary_composite_mae": ci < n_primary,
               "left": [], "right": []}
        for side in ("left", "right"):
            for seed in range(5):
                stem = f"c{ci}-{side}-s{seed}"
                outdir = root / "test-runs" / stem if split == "test" else root / "predictions"
                outdir.mkdir(parents=True, exist_ok=True)
                predpath = outdir / ("predictions.npz" if split == "test" else f"{stem}.npz")
                pred = np.full_like(targets, (seed + 1) * (1 if side == "left" else 2))
                np.savez(predpath, predictions=pred, targets=targets, target_mask=mask,
                         origins=target_pairs, seed_id=np.asarray(seed))
                bindpath = root / "bindings" / f"{stem}.json"
                _write_json(bindpath, {"binding_type": "at-launch", "dataset_root": str(data_root),
                                       "config_source": {"config": {"variant": "GRU"}}})
                entry = {"prediction_path": str(predpath.relative_to(root)),
                         "input_binding_path": str(bindpath.relative_to(root))}
                if split == "test":
                    checkpoint = outdir / "checkpoint.pt"
                    checkpoint.write_bytes(f"checkpoint-{stem}".encode())
                    freeze = outdir / "test-freeze.json"
                    validation_ev = _write_json(outdir / "validation-evidence.json", {"best_epoch": 1})
                    runtime_keys = {"baselines.py", "external_baselines.py", "data.py", "engine.py"}
                    current_sources = source_manifest()
                    _write_json(freeze, {"checkpoint_sha256": _sha(checkpoint), "parameters_sha256": "p",
                                         "test_accessed_before_freeze": False,
                                         "allowed_policy_digests": {"full": policy_digest},
                                         "dataset_fingerprint": dataset_fingerprint(source),
                                         "prediction_source_sha256": {k: current_sources[k] for k in runtime_keys},
                                         "dependency_source_sha256": dependency_manifest("GRU"),
                                         "validation_evidence_sha256": {str(validation_ev.resolve()): _sha(validation_ev)}})
                    eval_manifest = outdir / "evaluation_manifest.json"
                    _write_json(eval_manifest, {"split": "test", "scope": "full_protocol",
                                                "checkpoint_sha256": _sha(checkpoint), "parameters_before": "p",
                                                "parameters_after": "p", "policy": policy,
                                                "n_windows": len(target_pairs), "source_sha256": source_manifest()})
                    phase = outdir / "phase.json"
                    prediction_sha = _sha(predpath)
                    _write_json(phase, {"scope": "offline-experiment-phase", "phase": "evaluation",
                                        "status": "completed", "output_dir": str(outdir),
                                        "command": ["python", "-m", "edsgn.evaluate", "--checkpoint", str(checkpoint), "--split", "test"],
                                        "outputs": [{"path": str(eval_manifest), "sha256": _sha(eval_manifest)},
                                                    {"path": str(predpath), "sha256": prediction_sha}]})
                    entry.update(evaluation_manifest_path=str(eval_manifest.relative_to(root)),
                                 phase_receipt_path=str(phase.relative_to(root)),
                                 test_freeze_path=str(freeze.relative_to(root)),
                                 checkpoint_path=str(checkpoint.relative_to(root)))
                row[side].append(entry)
                all_paths.extend([root / value for key, value in entry.items() if key.endswith("_path")])
        contrasts.append(row)
    plan = {"schema_version": 2, "split": split, "primary_claim": "H1", "primary_family": "H1",
            "history_length": 2, "max_horizon": 2, "validation_acf_dependence_windows": 1,
            "block_length_windows": 2, "bootstrap_reps": 2000, "random_seed": 1234,
            "requested_horizons": [1, 2], "seed_ids": [0, 1, 2, 3, 4],
            "analysis_stratum": "toy-email-main", "acf_evidence_path": str(acf.relative_to(root)), "acf_evidence_sha256": _sha(acf), "contrasts": contrasts}
    planpath = _write_json(root / "protocols" / "analysis-plan.json", plan)
    plan_sha = _sha(planpath)
    state = json.loads(pipeline.read_text())
    p3a_row = next(x for x in state["stages"] if x["id"] == "P3A")
    p3a_row.setdefault("evidence", []).append({"path": str(planpath.relative_to(root)), "sha256": plan_sha})
    _write_json(pipeline, state)
    gate = root / "gate-receipt.json"
    auth = "p3a-running-validation" if split == "validation" else "p4-running-test"
    _write_json(gate, {"authorization": auth, "analysis_plan_sha256": plan_sha,
                       "pipeline_state_sha256": _sha(pipeline), "split": split})
    all_paths.extend([acf, gate, pipeline])
    if not binding:
        all_paths.remove(next(p for p in all_paths if p.name.startswith("c0-left-s0") and p.parent.name == "bindings"))
    unique = {str(p.resolve().relative_to(root.resolve())): _sha(p) for p in all_paths}
    inputpath = _write_json(root / "logs" / "input-manifest.json", {"schema_version": 1,
        "analysis_plan_sha256": plan_sha, "split": split,
        "artifacts": [{"path": p, "sha256": h} for p, h in unique.items()]})
    return planpath, inputpath, gate, pipeline, root / "logs" / "report.json", plan, root


def test_primary_holm_is_only_h1_composite_not_secondary(monkeypatch, tmp_path):
    _bind_loader(monkeypatch)
    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path, n_primary=2, include_secondary=True)
    report = paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)
    assert report["stratum_primary_contrast_count"] == 2
    assert set(report["stratum_primary_holm_adjusted_p"]) == {"contrast-0", "contrast-1"}
    assert report["contrasts"]["contrast-2"]["requested_horizons_equal_weight_pooled_mae"]["status"].startswith("secondary")
    assert "holm_adjusted_p" not in report["contrasts"]["contrast-2"]["by_horizon"]["1"]["mae"]
    assert report["contrasts"]["contrast-0"]["by_horizon"]["1"]["mae"]["inferential_role"].startswith("secondary")
    draws, digest = paired._common_draws(np.array([[0, 2], [0, 3]]), 2, 2000, 1234)
    assert report["shared_draw_schedule_sha256"] == digest
    assert draws.shape == (2000, 2)
    pooled = report["contrasts"]["contrast-0"]["requested_horizons_equal_weight_pooled_mae"]
    assert pooled["observed_mean_difference_left_minus_right"] == pytest.approx(-3.0)


def test_canonical_p3a_p4_gates_cannot_be_overridden_by_receipt(tmp_path):
    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path, split="test", n_primary=1, include_secondary=False)
    state = json.loads(pipeline.read_text()); next(x for x in state["stages"] if x["id"] == "P3A")["status"] = "pending"
    _write_json(pipeline, state)
    receipt = json.loads(gate.read_text()); receipt["pipeline_state_sha256"] = _sha(pipeline)
    _write_json(gate, receipt)
    # Refresh only the input manifest pipeline/gate hashes; the plan stays fixed.
    inp = json.loads(inputs.read_text())
    for art in inp["artifacts"]:
        if Path(art["path"]).resolve() in {gate.resolve(), pipeline.resolve()}:
            art["sha256"] = _sha(Path(art["path"]))
    _write_json(inputs, inp)
    with pytest.raises(ValueError, match="canonical P3A=passed/pass and P4=running"):
        paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)


def test_plan_tamper_or_missing_binding_hash_fails_before_read(tmp_path):
    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path, n_primary=1, include_secondary=False)
    changed = json.loads(plan.read_text()); changed["block_length_windows"] = 1
    _write_json(plan, changed)
    with pytest.raises(ValueError, match="canonical P3A ledger does not hash-bind"):
        paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)


def test_missing_binding_manifest_artifact_and_gapped_origins_rejected(monkeypatch, tmp_path):
    _bind_loader(monkeypatch)
    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path / "missing", n_primary=1, include_secondary=False)
    inp = json.loads(inputs.read_text())
    bind_path = next(a["path"] for a in inp["artifacts"] if "bindings" in a["path"])
    inp["artifacts"] = [a for a in inp["artifacts"] if a["path"] != bind_path]
    _write_json(inputs, inp)
    with pytest.raises(ValueError, match="artifact set differs"):
        paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)

    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path / "gap", n_primary=1, include_secondary=False, gap=True)
    with pytest.raises(ValueError, match="origins must be consecutive|archive origins are incomplete"):
        paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)


def test_test_archive_requires_full_bound_test_provenance(monkeypatch, tmp_path):
    _bind_loader(monkeypatch)
    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path, split="test", n_primary=1, include_secondary=False)
    result = paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)
    assert result["split"] == "test" and result["canonical_stage_statuses"]["P4"] == "running"
    assert result["shared_draw_schedule_sha256"]
    assert result["test_target_source_alignment_verified"] is True
    # A plan marked validation cannot point into an otherwise test-bound archive set.
    newplan = json.loads(plan.read_text()); newplan["split"] = "validation"
    _write_json(plan, newplan)
    with pytest.raises(ValueError, match="canonical P3A ledger does not hash-bind"):
        paired.analyze_plan(plan, inputs, gate, pipeline, tmp_path / "other-report.json", _test_root=fixture_root)


def test_test_plan_rejects_validation_evaluation_manifest(monkeypatch, tmp_path):
    _bind_loader(monkeypatch)
    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path, split="test", n_primary=1, include_secondary=False)
    inp = json.loads(inputs.read_text())
    eval_path = next(paired._resolve(fixture_root, a["path"]) for a in inp["artifacts"] if a["path"].endswith("evaluation_manifest.json"))
    evaluation = json.loads(eval_path.read_text()); evaluation["split"] = "validation"
    _write_json(eval_path, evaluation)
    for art in inp["artifacts"]:
        if paired._resolve(fixture_root, art["path"]) == eval_path:
            art["sha256"] = _sha(eval_path)
    _write_json(inputs, inp)
    with pytest.raises(ValueError, match="full-protocol test evaluation manifest"):
        paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)


def test_test_archive_with_rehashed_wrong_target_is_rejected(monkeypatch, tmp_path):
    _bind_loader(monkeypatch)
    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path, split="test", n_primary=1, include_secondary=False)
    changed = set()
    for side in ("left", "right"):
        for entry in doc["contrasts"][0][side]:
            predpath = paired._resolve(fixture_root, entry["prediction_path"])
            with np.load(predpath, allow_pickle=False) as z:
                arrays = {k: np.asarray(z[k]) for k in z.files}
            arrays["targets"] = arrays["targets"].copy(); arrays["targets"][0, 0, 0, 0] = 17
            with predpath.open("wb") as stream:
                np.savez(stream, **arrays)
            phasepath = paired._resolve(fixture_root, entry["phase_receipt_path"])
            phase = json.loads(phasepath.read_text())
            for row in phase["outputs"]:
                if paired._resolve(fixture_root, row["path"]) == predpath:
                    row["sha256"] = _sha(predpath)
            _write_json(phasepath, phase)
            changed.update((predpath, phasepath))
    inp = json.loads(inputs.read_text())
    for row in inp["artifacts"]:
        if paired._resolve(fixture_root, row["path"]) in changed:
            row["sha256"] = _sha(paired._resolve(fixture_root, row["path"]))
    _write_json(inputs, inp)
    with pytest.raises(ValueError, match="targets/masks/origins differ from bound source"):
        paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)


def test_test_freeze_rejects_missing_params_source_or_fingerprint(monkeypatch, tmp_path):
    _bind_loader(monkeypatch)
    for field in ("parameters_sha256", "prediction_source_sha256", "dataset_fingerprint"):
        plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path / field, split="test", n_primary=1, include_secondary=False)
        freeze_path = paired._resolve(fixture_root, doc["contrasts"][0]["left"][0]["test_freeze_path"])
        freeze = json.loads(freeze_path.read_text())
        freeze.pop(field)
        _write_json(freeze_path, freeze)
        for row in json.loads(inputs.read_text())["artifacts"]:
            if paired._resolve(fixture_root, row["path"]) == freeze_path:
                row["sha256"] = _sha(freeze_path)
        manifest = json.loads(inputs.read_text())
        for row in manifest["artifacts"]:
            if paired._resolve(fixture_root, row["path"]) == freeze_path:
                row["sha256"] = _sha(freeze_path)
        _write_json(inputs, manifest)
        with pytest.raises(ValueError):
            paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)


def test_final_toctou_rechecks_plan_gate_manifest_and_helpers(monkeypatch, tmp_path):
    _bind_loader(monkeypatch)
    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path, n_primary=1, include_secondary=False)
    original = paired._load_bound
    changed = {"done": False}
    def mutate_after_load(*args, **kwargs):
        value = original(*args, **kwargs)
        if not changed["done"]:
            changed["done"] = True
            acf = paired._resolve(fixture_root, doc["acf_evidence_path"])
            acf.write_text("mutated after reconstruction", encoding="utf-8")
        return value
    monkeypatch.setattr(paired, "_load_bound", mutate_after_load)
    with pytest.raises(ValueError, match="input manifest artifact changed"):
        paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)
    assert not output.exists()


def test_seed_ids_reject_bool_float_and_reordered(monkeypatch, tmp_path):
    _bind_loader(monkeypatch)
    for bad in ([False, 1, 2, 3, 4], [0.0, 1, 2, 3, 4], [1, 0, 2, 3, 4]):
        plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path / str(bad), n_primary=1, include_secondary=False)
        changed = json.loads(plan.read_text()); changed["seed_ids"] = bad
        _write_json(plan, changed)
        with pytest.raises(ValueError):
            paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)


def test_evaluation_rejects_missing_or_equal_none_parameter_hashes(monkeypatch, tmp_path):
    _bind_loader(monkeypatch)
    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path, split="test", n_primary=1, include_secondary=False)
    eval_path = paired._resolve(fixture_root, doc["contrasts"][0]["left"][0]["evaluation_manifest_path"])
    evaluation = json.loads(eval_path.read_text())
    evaluation["parameters_before"] = None
    evaluation["parameters_after"] = None
    _write_json(eval_path, evaluation)
    manifest = json.loads(inputs.read_text())
    for row in manifest["artifacts"]:
        if paired._resolve(fixture_root, row["path"]) == eval_path:
            row["sha256"] = _sha(eval_path)
    _write_json(inputs, manifest)
    with pytest.raises(ValueError, match="weight immutability"):
        paired.analyze_plan(plan, inputs, gate, pipeline, output, _test_root=fixture_root)


def test_noncanonical_pipeline_path_rejected(tmp_path):
    plan, inputs, gate, pipeline, output, _doc, fixture_root = _fixture(tmp_path, n_primary=1, include_secondary=False)
    alternate = fixture_root / "other-pipeline.json"
    alternate.write_bytes(pipeline.read_bytes())
    with pytest.raises(ValueError, match="canonical RUN/pipeline-state.json"):
        paired.analyze_plan(plan, inputs, gate, alternate, output, _test_root=fixture_root)


@pytest.mark.parametrize("mode", ["failed-pass-verdict", "stale-passed-evidence", "stale-running-evidence"])
def test_validation_gate_rejects_failed_acceptance_and_stale_evidence(tmp_path, mode):
    plan, inputs, gate, pipeline, output, doc, fixture_root = _fixture(tmp_path, n_primary=1, include_secondary=False)
    state = json.loads(pipeline.read_text())
    p3a = next(x for x in state["stages"] if x["id"] == "P3A")
    if mode != "stale-running-evidence":
        p3a.update(status="passed", gate_verdict="fail" if mode == "failed-pass-verdict" else "pass")
        next(x for x in state["stages"] if x["id"] == "P4")["status"] = "running"
    if mode.startswith("stale"):
        evidence = fixture_root / "extra-evidence.md"
        evidence.write_text("current immutable bytes")
        p3a.setdefault("evidence", []).append({"path": evidence.name, "sha256": "0" * 64})
    _write_json(pipeline, state)
    receipt = json.loads(gate.read_text())
    receipt["pipeline_state_sha256"] = _sha(pipeline)
    receipt["authorization"] = "p3a-running-validation" if mode == "stale-running-evidence" else "p4-running-validation"
    _write_json(gate, receipt)
    expected = "gate_verdict=pass" if mode == "failed-pass-verdict" else "evidence hash mismatch"
    with pytest.raises(ValueError, match=expected):
        paired._canonical_gate(fixture_root, pipeline, gate, "validation", _sha(plan), plan)
