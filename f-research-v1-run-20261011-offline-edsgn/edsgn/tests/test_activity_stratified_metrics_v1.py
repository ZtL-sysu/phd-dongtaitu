import numpy as np
import pytest

from activity_stratified_metrics_v1 import fit_training_strata, stratified_metrics


def test_training_ties_zero_and_unobserved_remain_distinct():
    x = np.array([[[0.], [1.], [1.], [9.], [99.]], [[0.], [1.], [1.], [9.], [99.]]])
    m = np.ones_like(x, dtype=bool); m[:, 4] = False
    s = fit_training_strata(x, m, np.arange(5))
    assert s["groups"] == ["zero_train", "low_positive", "low_positive", "high_positive", "unobserved_train"]
    assert s["valid_training_cell_count"] == [2, 2, 2, 2, 0]
    assert s["training_scores"][4] == 0.


def test_raw_masked_cell_pool_and_equal_horizon_seed_summary():
    train = np.array([[[1.], [2.], [3.]]])
    s = fit_training_strata(train, np.ones_like(train, dtype=bool), np.arange(3))
    target = np.zeros((2, 3, 3, 1)); mask = np.ones_like(target, dtype=bool)
    mask[1, 0, 0, 0] = False
    p = np.zeros((2, 2, 3, 3, 1))
    p[0, :, 0, 0] = [[2.], [1000.]]; p[0, :, 2, 0] = 4.
    p[1, :, 0, 0] = [[6.], [1000.]]; p[1, :, 2, 0] = 8.
    rows = stratified_metrics(p, target, mask, [1, 3], s)["rows"]
    summary = next(r for r in rows if r["stratum"] == "low_positive" and r.get("scope"))
    assert summary["per_seed_raw_mae"] == [3., 7.]
    assert summary["mean_raw_mae"] == 5.
    assert summary["seed_sd_raw_mae"] == pytest.approx(np.sqrt(8.))
    assert next(r for r in rows if r["stratum"] == "zero_train" and r.get("scope"))["identifiable"] is False


def test_signed_strata_and_strict_invalid_axes():
    x = np.array([[[-2.], [1.]]]); m = np.ones_like(x, dtype=bool)
    assert fit_training_strata(x, m, np.arange(2), signed=True)["training_scores"] == [2., 1.]
    with pytest.raises(ValueError): fit_training_strata(x, m, np.arange(2))
    with pytest.raises(ValueError): fit_training_strata(x, m.astype(int), np.arange(2), signed=True)
