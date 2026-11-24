import numpy as np
import pytest

from validation_dependence_v1 import validation_dependence


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
    result = fixture([(0, [1, 2, 1, 2]), (1, [101, 102, 101, 102])])
    out = validation_dependence(result, [1, 3], 2, 3, test_support_by_trajectory={0: 8, 1: 8})
    # The biased denominator covers all four samples: lag-1 numerator −.75.
    assert out["acf"][0]["acf"] == pytest.approx(-0.75)
    assert out["acf"][1]["acf"] == pytest.approx(0.5)
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
    assert out["dependence_status"] == "unidentifiable-zero-variance-or-all-zero-targets"
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
    r = fixture([(0, [1, 2, 1, 2, 1, 2, 1, 2])])
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
    assert len(out["acf"]) == 64
    assert out["max_supported_lag"] == 64
    assert out["dependence_status"] == "right_censored-no-crossing-through-max-supported-lag"
    assert out["dependence_length"] == 65
    assert out["block_length"] == 65


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

