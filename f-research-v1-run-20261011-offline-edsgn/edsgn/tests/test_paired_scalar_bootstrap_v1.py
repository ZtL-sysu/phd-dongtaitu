import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "edsgn/staging/paired_scalar_bootstrap_v1.py"
spec = importlib.util.spec_from_file_location("paired_scalar_bootstrap_v1", MODULE)
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


def _inputs(n=12, seeds=5, trajectories=1):
    pairs = np.concatenate([np.column_stack((np.full(n, t), np.arange(n)))
                            for t in range(trajectories)]).astype(np.int64)
    shape = (seeds, len(pairs))
    counts = np.ones(shape, dtype=float)
    left = np.full(shape, 2.0)
    right = np.full(shape, 3.0)
    return left, counts, right, counts.copy(), pairs


def test_exact_shared_schedule_and_per_seed_pooled_fractional_count_weighting():
    pairs = np.array([[0, i] for i in range(8)] + [[1, i] for i in range(8)], dtype=np.int64)
    count = np.tile(np.array([0.5, 1.5, 2.0, 0.75, 1.25, 3.0, 0.25, 2.5] * 2), (5, 1))
    left_values = np.tile(np.array([1, 10, 2, 8, 3, 9, 4, 7] * 2, dtype=float), (5, 1))
    right_values = left_values + 2.0
    left_sum = left_values * count
    right_sum = right_values * count
    result = kernel.paired_scalar_bootstrap(left_sum, count, right_sum, count.copy(), pairs,
                                            block_length=3)
    expected = np.mean(left_sum.sum(axis=1) / count.sum(axis=1)
                       - right_sum.sum(axis=1) / count.sum(axis=1))
    draws, digest = kernel._shared._common_draws(pairs, 3, 2000, 1234)
    assert result["estimate"] == pytest.approx(expected)
    assert result["estimate"] == pytest.approx(-2.0)
    assert result["draw_schedule_sha256"] == digest
    assert result["bootstrap_reps"] == 2000 and result["random_seed"] == 1234
    assert result["percentile_95_interval"] == pytest.approx([-2.0, -2.0])
    assert result["upper_95"] == pytest.approx(-2.0)
    assert result["centered_two_sided_p"] == pytest.approx(1 / 2001)


def test_nonzero_margin_lower_tail_direction_and_perfect_shift():
    left, count, right, _, pairs = _inputs()
    right += 0.5
    result = kernel.paired_scalar_bootstrap(left, count, right, count, pairs,
                                            block_length=3, margin=-0.25)
    assert result["estimate"] == pytest.approx(-1.5)
    assert result["upper_95"] < -0.25
    assert result["centered_lower_tail_margin_p"] == pytest.approx(1 / 2001)
    # If left is worse by the same amount, a negative margin is not passed.
    worse = kernel.paired_scalar_bootstrap(right, count, left, count, pairs,
                                           block_length=3, margin=-0.25)
    assert worse["estimate"] == pytest.approx(1.5)
    assert worse["centered_lower_tail_margin_p"] == pytest.approx(1.0)


@pytest.mark.parametrize("change,match", [
    ("gap", "gapped"), ("block", "block length"), ("nan", "finite"),
    ("seedshape", "shape"), ("support", "support counts"), ("pairorder", "ordered"),
])
def test_rejects_unsupported_or_mismatched_inputs(change, match):
    left, lc, right, rc, pairs = _inputs()
    if change == "gap":
        pairs[5:, 1] += 1
    elif change == "block":
        with pytest.raises(ValueError, match=match):
            kernel.paired_scalar_bootstrap(left, lc, right, rc, pairs, block_length=13)
        return
    elif change == "nan":
        left[0, 0] = np.nan
    elif change == "seedshape":
        left, lc, right, rc = left[:4], lc[:4], right[:4], rc[:4]
    elif change == "support":
        rc[0, 0] = 2
    elif change == "pairorder":
        pairs[[0, 1]] = pairs[[1, 0]]
    with pytest.raises(ValueError, match=match):
        kernel.paired_scalar_bootstrap(left, lc, right, rc, pairs, block_length=3)


def test_seedwise_pooled_then_equal_five_not_window_or_seed_weighted():
    _, _, _, _, pairs = _inputs(n=6)
    counts = np.ones((5, 6), dtype=float)
    counts[0] = 100.0
    left_sum = np.vstack([np.full(6, value) for value in [100, 0, 0, 0, 0]])
    right_sum = np.zeros_like(left_sum)
    out = kernel.paired_scalar_bootstrap(left_sum, counts, right_sum, counts.copy(), pairs,
                                         block_length=2)
    # Seed 0 contributes its own pooled mean (1), then receives equal 1/5 weight.
    assert out["seed_estimates"] == pytest.approx([1, 0, 0, 0, 0])
    assert out["estimate"] == pytest.approx(0.2)
