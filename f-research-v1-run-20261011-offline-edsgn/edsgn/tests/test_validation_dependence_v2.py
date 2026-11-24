import numpy as np
import pytest

from validation_dependence_v2 import validation_dependence


def fixture(series_by_traj, *, horizons=3, nodes=2, masks=None, targets=None):
    pairs = []
    vals = []
    for tid, series in series_by_traj:
        for origin, e in enumerate(series):
            pairs.append((tid, origin))
            vals.append(float(e))
    n = len(pairs)
    pred = np.zeros((n, horizons, nodes, 1), dtype=float)
    target = np.ones_like(pred) if targets is None else np.asarray(targets, dtype=float)
    mask = np.ones_like(pred, dtype=bool) if masks is None else np.asarray(masks, dtype=bool)
    for i, e in enumerate(vals):
        # Every required horizon has the same raw MAE e in the base fixture.
        pred[i, :, :, 0] = e + 1.0
    return {"pairs": np.asarray(pairs, dtype=np.int64), "predictions": pred,
            "targets": target, "target_mask": mask}


def test_trajectory_centering_prevents_between_trajectory_cross_products():
    result = fixture([(0, [0, 2, 0, 0]), (1, [100, 102, 100, 100])])
    out = validation_dependence(result, [1, 3], 2, 3, test_support_by_trajectory={0: 8, 1: 8})
    # Per-trajectory ACF is isolated; different residual levels cannot pool.
    assert out["per_trajectory_dependence"][0]["acf"][0]["acf"] == pytest.approx(-1.25/3)
    assert out["per_trajectory_dependence"][0]["acf"][1]["acf"] == pytest.approx(-1/6)
    assert out["selection_status"] == "available-for-prospective-use"
    assert out["block_length"] == max(2, 3, out["dependence_length"])


def test_equal_horizon_weighting_and_heterogeneous_masks():
    r = fixture([(0, [0, 2, 0, 2, 0, 2])], horizons=3, nodes=2)
    # Horizon 1 has two valid cells with abs errors 1 and 3 (MAE 2).
    # Horizon 3 has one valid cell with abs error 10 (MAE 10); h2 is 0.
    r["predictions"][:] = 0
    r["targets"][:] = 1
    r["predictions"][:, 0, :, 0] = np.array([[3, 3]] * 6)
    r["predictions"][:, 2, 0, 0] = 11
    r["target_mask"][:, 2, 1, 0] = False
    out = validation_dependence(r, [1, 3], 1, 1, test_support_by_trajectory={0: 10})
    # Each origin residual is (2 + 10) / 2 = 6, and therefore has zero variance.
    assert out["per_trajectory"][0]["residual_mean"] == pytest.approx(6.0)
    assert out["per_trajectory"][0]["zero_variance_unidentifiable"]
    assert out["dependence_status"] == "unavailable-trajectory-zero-variance"
    assert out["block_length"] is None


def test_no_valid_origin_and_missing_horizon_fail_closed():
    r = fixture([(0, [1, 2, 3])])
    r["target_mask"][:] = False
    with pytest.raises(ValueError, match="zero valid targets"):
        validation_dependence(r, [1], 1, 1, test_support_by_trajectory={0: 5})
    r = fixture([(0, [1, 2, 3])])
    r["target_mask"][1, 0] = False
    with pytest.raises(ValueError, match="zero valid targets"):
        validation_dependence(r, [1], 1, 1, test_support_by_trajectory={0: 5})


@pytest.mark.parametrize("pairs", [
    [(0, 0), (0, 2), (0, 3)],
    [(0, 0), (1, 0), (0, 1)],
    [(0, 0), (0, 1), (0, 1)],
    [(0, 1), (0, 0), (0, 2)],
])
def test_reject_gaps_reordering_duplicates_and_interleaving(pairs):
    r = fixture([(0, [1, 2, 3])])
    r["pairs"] = np.asarray(pairs, dtype=np.int64)
    with pytest.raises(ValueError, match="canonical|gap|duplicate|reordering"):
        validation_dependence(r, [1], 1, 1, test_support_by_trajectory={0: 5, 1: 5})


def test_history_and_hmax_are_hard_lower_bounds_and_support_never_caps():
    r = fixture([(0, [0, 1, 2, 1, 0, 1, 2, 1])])
    out = validation_dependence(r, [1], 9, 7, test_support_by_trajectory={0: 20})
    assert out["block_length"] == 9
    assert out["selection_status"] == "unavailable"
    assert "block-exceeds-shortest-validation-trajectory" in out["unavailable_reasons"]
    short_test = validation_dependence(r, [1], 2, 2, test_support_by_trajectory={0: 1})
    assert short_test["block_length"] == max(2, 2, short_test["dependence_length"])
    assert short_test["selection_status"] == "unavailable"
    assert "block-exceeds-shortest-declared-test-support" in short_test["unavailable_reasons"]
    missing = validation_dependence(r, [1], 2, 2)
    assert missing["selection_status"] == "unavailable"
    assert "unavailable-test-declared-support-missing" in missing["unavailable_reasons"]


def test_right_censored_records_every_supported_lag_without_n_over_ten_shortcut():
    # Residual errors grow linearly, yielding positive high correlations through lag 64.
    r = fixture([(0, [0, 2] * 40)])
    out = validation_dependence(r, [1], 1, 1, test_support_by_trajectory={0: 100})
    assert len(out["per_trajectory_dependence"][0]["acf"]) == 64
    assert out["max_supported_lag"] == 64
    assert out["dependence_status"] == "unavailable-right-censored-trajectory"
    assert out["per_trajectory_dependence"][0]["status"] == "right-censored-no-crossing-through-max-supported-lag"
    assert out["dependence_length"] is None
    assert out["block_length"] is None
    assert out["selection_status"] == "unavailable"


def test_all_zero_targets_are_unidentifiable_and_external_email_history_is_explicit():
    r = fixture([(0, [1, 2, 1, 2])])
    r["targets"][:] = 0
    base = validation_dependence(r, [1], 2, 3, test_support_by_trajectory={0: 10})
    assert base["all_valid_targets_zero"]
    assert base["block_length"] is None
    assert base["positive_quality_claim_supported"] is False
    assert base["selection_status"] == "unavailable"
    email = validation_dependence(r, [1], 2, 3, test_support_by_trajectory={0: 40},
                                  all_zero_target_external_history=28)
    assert email["block_length"] == 28
    assert email["dependence_length"] is None
    assert email["positive_quality_claim_supported"] is False


def test_maximum_trajectory_dependence_not_pooled_early_crossing():
    # Trajectory 0 alternates and crosses at lag 1. Trajectory 1 is a half/half
    # step signal and crosses at lag 5. Scale trajectory 0 so pooled lag-1
    # numerator would cancel, which would incorrectly make pooled ACF cross at 1.
    base0 = np.asarray([0, 1, 3, 2, 1, 4, 0, 2], dtype=float)
    centered0 = base0 - base0.mean()
    scale0 = np.sqrt(3.25 / -float(np.dot(centered0[1:], centered0[:-1])))
    series0 = (base0 * scale0 + 5.0).tolist()
    series1 = [0.5] * 8 + [1.5] * 8
    r = fixture([(0, series0), (1, series1)])
    out = validation_dependence(r, [1], 1, 1, test_support_by_trajectory={200: 20, 201: 20})
    deps = out["per_trajectory_dependence"]
    # A naively pooled ACF would cancel the lag-1 numerators and cross early.
    centered1 = np.asarray(series1) - np.mean(series1)
    pooled_lag1 = (float(np.dot((np.asarray(series0)-np.mean(series0))[1:], (np.asarray(series0)-np.mean(series0))[:-1]))
                   + float(np.dot(centered1[1:], centered1[:-1]))) / (
                   float(np.dot(np.asarray(series0)-np.mean(series0), np.asarray(series0)-np.mean(series0)))
                   + float(np.dot(centered1, centered1)))
    assert abs(pooled_lag1) < 0.1
    assert deps[0]["dependence_length"] == 2
    assert deps[1]["dependence_length"] == 5
    assert out["dependence_length"] == 5
    assert out["block_length"] == 5
    assert out["test_support_status"] == "declared"
    assert out["test_trajectory_count"] == 2


def test_test_support_ids_are_independent_and_every_test_trajectory_limits_support():
    r = fixture([(0, [0, 2] * 8), (1, [1, 3] * 8)])
    # Test trajectories need not share validation IDs and may contain more
    # independent trajectories than the validation side.
    out = validation_dependence(r, [1], 1, 1,
        test_support_by_trajectory={200: 20, 201: 18, 202: 17, 203: 4})
    assert out["test_trajectory_count"] == 4
    assert out["test_support_by_trajectory"][-1] == {"trajectory_id": 203, "n_origins": 4}
    assert out["selection_status"] == "unavailable"
    assert "block-exceeds-shortest-declared-test-support" in out["unavailable_reasons"]
    missing = validation_dependence(r, [1], 1, 1)
    assert missing["selection_status"] == "unavailable"
    with pytest.raises(ValueError, match="at least one"):
        validation_dependence(r, [1], 1, 1, test_support_by_trajectory={})


@pytest.mark.parametrize("kwargs", [
    {"required_horizons": [1.0]},
    {"required_horizons": [True]},
    {"history_length": 1.5},
    {"hmax": False},
])
def test_integer_protocol_fields_reject_float_and_bool(kwargs):
    r = fixture([(0, [1, 2, 1, 2])])
    options = {"required_horizons": [1], "history_length": 1, "hmax": 1}
    options.update(kwargs)
    with pytest.raises(ValueError):
        validation_dependence(r, options["required_horizons"], options["history_length"], options["hmax"],
                              test_support_by_trajectory={200: 10})

