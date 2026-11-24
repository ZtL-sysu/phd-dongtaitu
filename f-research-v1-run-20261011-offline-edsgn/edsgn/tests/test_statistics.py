import numpy as np
import pytest

from edsgn.statistics import (
    block_bootstrap_difference,
    block_bootstrap_pooled_mae_difference,
    circular_block_indices,
    compare_prediction_sets,
    holm_adjust,
    run_config,
)


def _write_predictions(path, prediction_value, *, seed, targets=None, pairs=None, mask=None):
    if targets is None:
        targets = np.zeros((8, 3, 2, 1), dtype=np.float32)
    if pairs is None:
        pairs = np.array([[0, t] for t in range(4)] + [[1, t] for t in range(4)], dtype=np.int64)
    if mask is None:
        mask = np.ones_like(targets, dtype=bool)
    predictions = np.full_like(targets, prediction_value)
    np.savez_compressed(path, predictions=predictions, targets=targets, mask=mask,
                        pairs=pairs, seed_id=np.asarray(seed))


def test_known_constant_difference_and_seed_spread(tmp_path):
    left, right = [], []
    for seed in range(5):
        lp, rp = tmp_path / f"left{seed}.npz", tmp_path / f"right{seed}.npz"
        _write_predictions(lp, 1.0, seed=seed)
        _write_predictions(rp, 2.0, seed=seed)
        left.append(lp)
        right.append(rp)
    result = compare_prediction_sets(left, right, horizon_steps=[1, 3], block_length=2,
                                    bootstrap_reps=99, random_seed=1234)
    primary = result["primary"]
    assert primary["observed_mean_difference_left_minus_right"] == pytest.approx(-1.0)
    assert primary["ci95_percentile"] == pytest.approx([-1.0, -1.0])
    assert primary["left_better_if_negative"] is True
    assert primary["seed_mean_differences"] == pytest.approx([-1.0] * 5)
    assert primary["cross_seed_sample_std"] == pytest.approx(0.0)
    assert primary["centered_bootstrap_two_sided_p"] == pytest.approx(0.01)
    assert result["secondary_by_horizon"]["rmse_horizon_3"]["observed_mean_difference_left_minus_right"] == pytest.approx(-1.0)


def test_engine_canonical_target_mask_origins_seed_schema_runs_full_compare(tmp_path):
    pairs = np.array([[0, 27], [0, 28], [0, 29], [1, 27], [1, 28], [1, 29]], dtype=np.int64)
    targets = np.zeros((6, 2, 3, 2), dtype=np.float32)
    mask = np.ones_like(targets, dtype=bool)
    left, right = [], []
    for seed in range(5):
        lp, rp = tmp_path / f"engine-left-{seed}.npz", tmp_path / f"engine-right-{seed}.npz"
        np.savez_compressed(lp, predictions=np.ones_like(targets), targets=targets,
                            target_mask=mask, origins=pairs, seed_id=np.asarray(seed))
        np.savez_compressed(rp, predictions=np.full_like(targets, 2.0), targets=targets,
                            target_mask=mask, origins=pairs, seed_id=np.asarray(seed))
        left.append(lp)
        right.append(rp)
    result = compare_prediction_sets(left, right, horizon_steps=[1, 2], block_length=2,
                                    bootstrap_reps=49, random_seed=77)
    assert result["primary"]["observed_mean_difference_left_minus_right"] == pytest.approx(-1.0)
    assert result["primary"]["n_windows"] == 6
    assert result["primary"]["trajectory_window_counts"] == {"0": 3, "1": 3}
    assert result["primary"]["seed_mean_differences"] == pytest.approx([-1.0] * 5)


def test_heterogeneous_mask_primary_mae_pools_valid_sae_before_dividing(tmp_path):
    pairs = np.array([[0, 10], [0, 11]], dtype=np.int64)
    targets = np.zeros((2, 1, 2, 1), dtype=np.float32)
    mask = np.array([[[[True], [True]]], [[[True], [False]]]], dtype=bool)
    left, right = [], []
    for seed in range(5):
        lp, rp = tmp_path / f"l{seed}.npz", tmp_path / f"r{seed}.npz"
        left_pred = np.zeros_like(targets)
        left_pred[1, 0, 0, 0] = 10.0
        right_pred = np.zeros_like(targets)
        for path, prediction in ((lp, left_pred), (rp, right_pred)):
            np.savez_compressed(path, predictions=prediction, targets=targets,
                                target_mask=mask, origins=pairs, seed_id=np.asarray(seed))
        left.append(lp); right.append(rp)
    result = compare_prediction_sets(left, right, horizon_steps=[1], block_length=1,
                                     bootstrap_reps=30, random_seed=31)
    assert result["primary"]["observed_mean_difference_left_minus_right"] == pytest.approx(10 / 3)
    rmse = result["secondary_by_horizon"]["rmse_horizon_1"]
    assert rmse["observed_mean_difference_left_minus_right"] == pytest.approx(np.sqrt(100 / 3))
    assert rmse["metric_aggregation"].startswith("per seed: sqrt(sum sampled valid squared errors")


def test_constant_valid_counts_make_pooled_primary_equal_window_macro(tmp_path):
    pairs = np.array([[0, 3], [0, 4]], dtype=np.int64)
    targets = np.zeros((2, 1, 2, 1), dtype=np.float32)
    mask = np.ones_like(targets, dtype=bool)
    prediction = np.array([[[[1.0], [3.0]]], [[[5.0], [7.0]]]], dtype=np.float32)
    left, right = [], []
    for seed in range(5):
        lp, rp = tmp_path / f"constant-l{seed}.npz", tmp_path / f"constant-r{seed}.npz"
        np.savez_compressed(lp, predictions=prediction, targets=targets, target_mask=mask,
                            origins=pairs, seed_id=np.asarray(seed))
        np.savez_compressed(rp, predictions=np.zeros_like(prediction), targets=targets,
                            target_mask=mask, origins=pairs, seed_id=np.asarray(seed))
        left.append(lp); right.append(rp)
    result = compare_prediction_sets(left, right, horizon_steps=[1], block_length=1,
                                     bootstrap_reps=20, random_seed=15)
    # Window MAEs are 2 and 6, whose equal-window mean and pooled mean are both 4.
    assert result["primary"]["observed_mean_difference_left_minus_right"] == pytest.approx(4.0)


def test_pooled_primary_bootstrap_reproduces_common_temporal_index_draws():
    pairs = np.array([[0, t] for t in range(5)], dtype=np.int64)
    left_sae = np.stack([np.arange(1, 6, dtype=np.float64) * (seed + 1) for seed in range(5)])[:, None, :]
    counts = np.ones_like(left_sae)
    zeros = np.zeros_like(left_sae)
    reps, seed, block = 80, 702, 2
    report = block_bootstrap_pooled_mae_difference(
        left_sae, counts, zeros, counts, pairs, block_length=block,
        bootstrap_reps=reps, random_seed=seed,
    )
    rng = np.random.default_rng(seed)
    manual = []
    for _ in range(reps):
        chosen = circular_block_indices(pairs, block, rng)
        per_seed = left_sae[:, :, chosen].sum(axis=2) / counts[:, :, chosen].sum(axis=2)
        manual.append(float(per_seed.mean()))
    expected_ci = np.quantile(manual, [0.025, 0.975])
    assert report["ci95_percentile"] == pytest.approx(expected_ci)
    assert report["shared_temporal_indices_across_all_seeds_horizons_methods"] is True


def test_gapped_origins_are_rejected_before_time_block_resampling():
    pairs = np.array([[0, 3], [0, 4], [0, 9], [0, 10]], dtype=np.int64)
    with pytest.raises(ValueError, match="must be consecutive"):
        circular_block_indices(pairs, block_length=2, rng=np.random.default_rng(1))


def test_same_method_difference_is_zero_with_no_false_significance():
    pairs = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    report = block_bootstrap_difference(np.zeros((5, 6)), pairs, block_length=2,
                                        bootstrap_reps=50, random_seed=4)
    assert report["observed_mean_difference_left_minus_right"] == 0
    assert report["ci95_percentile"] == [0.0, 0.0]
    assert report["centered_bootstrap_two_sided_p"] == 1


def test_identical_seed_copies_keep_single_series_temporal_ci_width():
    pairs = np.array([[0, t] for t in range(4)] + [[1, t] for t in range(4)])
    one_series = np.array([0.0, 0.2, 0.8, -0.1, 0.3, -0.5, 0.7, 0.1])
    repeated = np.tile(one_series, (5, 1))
    reps, seed, block = 300, 91, 2
    result = block_bootstrap_difference(repeated, pairs, block_length=block,
                                        bootstrap_reps=reps, random_seed=seed)
    rng = np.random.default_rng(seed)
    single_series_boot = np.asarray([
        one_series[circular_block_indices(pairs, block, rng)].mean()
        for _ in range(reps)
    ])
    expected_ci = np.quantile(single_series_boot, [0.025, 0.975])
    assert result["ci95_percentile"] == pytest.approx(expected_ci)
    assert result["shared_temporal_indices_across_all_seeds"] is True


def test_circular_blocks_preserve_trajectory_boundaries_and_wrap_inside_group():
    pairs = np.array([[3, t] for t in range(4)] + [[9, t] for t in range(4)])
    sampled = circular_block_indices(pairs, block_length=2, rng=np.random.default_rng(8))
    assert len(sampled) == len(pairs)
    assert set(sampled[:4]).issubset(set(range(4)))
    assert set(sampled[4:]).issubset(set(range(4, 8)))
    for group_slice, offset in ((sampled[:4], 0), (sampled[4:], 4)):
        for start in (0, 2):
            first, second = group_slice[start:start + 2] - offset
            assert (pairs[second, 1] - pairs[first, 1]) % 4 == 1


@pytest.mark.parametrize("field", ["targets", "mask", "pairs"])
def test_rejects_mismatched_targets_masks_or_pairs(tmp_path, field):
    left, right = [], []
    for seed in range(5):
        lp, rp = tmp_path / f"l{seed}.npz", tmp_path / f"r{seed}.npz"
        _write_predictions(lp, 0.0, seed=seed)
        targets = np.zeros((8, 3, 2, 1), dtype=np.float32)
        mask = np.ones_like(targets, dtype=bool)
        pairs = np.array([[0, t] for t in range(4)] + [[1, t] for t in range(4)], dtype=np.int64)
        if seed == 2:
            if field == "targets":
                targets[0, 0, 0, 0] = 1
            elif field == "mask":
                mask[0, 0, 0, 0] = False
            else:
                pairs[:, 1] += 10
        _write_predictions(rp, 0.0, seed=seed, targets=targets, mask=mask, pairs=pairs)
        left.append(lp)
        right.append(rp)
    with pytest.raises(ValueError, match=f"identical {field}"):
        compare_prediction_sets(left, right, horizon_steps=[1], block_length=2, bootstrap_reps=10)


def test_holm_correction_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust({"h1": 0.01, "h2": 0.04, "h3": 0.03})
    assert adjusted == pytest.approx({"h1": 0.03, "h2": 0.06, "h3": 0.06})
    assert holm_adjust({}) == {}


def test_config_applies_holm_across_only_named_primary_contrasts(tmp_path):
    left_one, right_one, left_two, right_two = [], [], [], []
    for seed in range(5):
        files = [tmp_path / f"{label}{seed}.npz" for label in ("l1", "r1", "l2", "r2")]
        for path, value in zip(files, (1.0, 2.0, 2.0, 1.0)):
            _write_predictions(path, value, seed=seed)
        left_one.append(str(files[0]))
        right_one.append(str(files[1]))
        left_two.append(str(files[2]))
        right_two.append(str(files[3]))
    report = run_config({
        "horizon_steps": [1], "block_length": 2, "bootstrap_reps": 49,
        "random_seed": 1234,
        "contrasts": [
            {"name": "primary-a", "left_npz": left_one, "right_npz": right_one},
            {"name": "primary-b", "left_npz": left_two, "right_npz": right_two},
        ],
    }, base_dir=tmp_path)
    assert report["primary_family_contrast_count"] == 2
    assert report["primary_contrasts"]["primary-a"]["primary"]["holm_adjusted_p_primary_family"] == pytest.approx(0.04)
    assert report["primary_contrasts"]["primary-b"]["primary"]["holm_adjusted_p_primary_family"] == pytest.approx(0.04)

