from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edsgn.data import PreparedData
from frozen_validation_calibration_v1 import (
    _calibration_callbacks, dependence_block, metadata_test_support,
    run_with_validation_dependence, validate_preflight, validate_task_payload,
)


def prepared(*, zero=True):
    T, N, F = 48, 2, 1
    values = np.zeros((T, N, F), np.float32)
    targets = np.zeros_like(values) if zero else np.arange(T*N*F, dtype=np.float32).reshape(T,N,F)
    mask = np.ones_like(values, bool)
    return PreparedData(values, targets, mask, mask.copy(), values.copy(), np.arange(T),
                        np.arange(N), {"train": (0, 30), "val": (30, 38), "test": (38, 48)},
                        np.zeros(F), np.ones(F), np.ones((N,F)), True, (1, 3), 2, 3)


def validation_result(zero=True):
    pairs = np.asarray([(0, i) for i in range(4)], dtype=np.int64)
    targets = np.zeros((4, 3, 2, 1), np.float32) if zero else np.ones((4,3,2,1), np.float32)
    return {"pairs": pairs, "predictions": np.zeros_like(targets), "targets": targets,
            "target_mask": np.ones_like(targets, bool)}


def payload():
    return {"task": {"history": 2, "hmax": 3, "n_nodes": 2, "n_features": 1},
            "config": {"seed": 0}}


def test_task_shape_and_test_support_are_metadata_only():
    ds = prepared()
    assert validate_task_payload(payload(), [ds])["shape_TNF"] == [48, 2, 1]
    assert metadata_test_support([ds]) == {0: 8}
    bad = payload(); bad["task"]["history"] = 28
    with pytest.raises(ValueError, match="history"):
        validate_task_payload(bad, [ds])


def test_all_zero_external_history_recorded_as_nonidentifiable():
    ds = prepared()
    length, report = dependence_block(validation_result(), (1,3), task=payload(),
                                      datasets=[ds], test_support={0: 8})
    assert length == 3
    assert report["dependence_status"] == "unidentifiable-all-zero-targets-external-history-preset"
    assert report["primary_quality_identifiable"] is False


def test_unavailable_acf_fails_closed_and_callbacks_restore_on_error():
    ds = prepared(zero=False)
    result = validation_result(zero=False)
    with pytest.raises(RuntimeError, match="unavailable"):
        dependence_block(result, (1,3), task=payload(), datasets=[ds], test_support={0:8})
    module = SimpleNamespace(validation_block_length=lambda *_: (99, []),
                             evaluate_batches=lambda *a, **k: {})
    original_block, original_eval = module.validation_block_length, module.evaluate_batches
    captured = {}
    with pytest.raises(RuntimeError, match="toy failure"):
        with _calibration_callbacks(module, payload=payload(), datasets=[prepared()],
                                    test_support={0:8}, captured=captured):
            raise RuntimeError("toy failure")
    assert module.validation_block_length is original_block
    assert module.evaluate_batches is original_eval


def test_full_validation_capture_and_npz_seed(tmp_path):
    ds = prepared()
    result = validation_result()
    module = SimpleNamespace()
    module.validation_block_length = lambda *_: (99, [])
    def evaluator(*args, **kwargs):
        return result
    module.evaluate_batches = evaluator
    def fake_calibrate(checkpoint, out, queue, **kwargs):
        val = module.evaluate_batches(None, [ds], "val", "cpu", 1, None, None)
        module.validation_block_length(val, (1,3))
        return {"chosen_id": "s00", "selection_status": "unidentifiable-all-zero-validation-targets"}
    module.calibrate_run = fake_calibrate
    out = tmp_path / "run"; out.mkdir()
    cp = tmp_path / "ckpt"; cp.write_bytes(b"toy")
    queue = tmp_path / "queue"; queue.write_bytes(b"toy")
    ret = run_with_validation_dependence(module, cp, out, queue, payload=payload(),
                                         validation_datasets=[ds], test_support={0:8})
    archive = np.load(out / "validation-predictions.npz")
    assert archive["seed_id"].shape == () and int(archive["seed_id"]) == 0
    assert archive["origins"].shape == (4,2)
    assert ret["validation_dependence"]["block_length"] == 3
    assert (out / "calibration-validation-dependence-v1.json").is_file()


def test_preflight_requires_all_canonical_gates_and_exact_queue(tmp_path):
    state = {"stages": [
        {"id":"P2","status":"passed","gate_verdict":"pass"},
        {"id":"P3","status":"passed","gate_verdict":"pass"},
        {"id":"P3A","status":"passed","gate_verdict":"pass"},
        {"id":"P4","status":"running"},
    ]}
    files=[]
    for name in ("cp","config","eps"):
        p=tmp_path/name;p.write_bytes(name.encode());files.append(p)
    good = validate_preflight(root=tmp_path,pipeline_state=state,checkpoint=files[0],config=files[1],
        queue=[{}]*12,epsilon_provenance=files[2],accepted_provenance={"accepted":True},
        locks_clear=True,owned_processes=[],spent_seconds=1)
    assert good["calibration_executed"] is False
    state["stages"][2]["gate_verdict"]="fail"
    with pytest.raises(RuntimeError,match="P3A"):
        validate_preflight(root=tmp_path,pipeline_state=state,checkpoint=files[0],config=files[1],
            queue=[{}]*12,epsilon_provenance=files[2],accepted_provenance={"accepted":True},
            locks_clear=True,owned_processes=[],spent_seconds=1)
