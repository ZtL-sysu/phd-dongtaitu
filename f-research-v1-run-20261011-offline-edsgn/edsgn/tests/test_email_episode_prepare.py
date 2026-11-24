import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_email_episode.py"
SPEC = importlib.util.spec_from_file_location("prepare_email_episode", SCRIPT)
episode = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(episode)


def test_episode_selection_chooses_first_feasible_not_most_active():
    episodes = [(0, 140), (175, 375)]
    index, selected, splits, origins = episode.choose_earliest_feasible(episodes)
    assert index == 0
    assert selected == episodes[0]
    assert all(origins[name] > 0 for name in ("train", "validation", "test"))
    # The later candidate is longer, but chronology is the only selection key.
    assert episodes[1][1] - episodes[1][0] > episodes[0][1] - episodes[0][0]


def test_long_empty_run_cut_is_inclusive_at_exact_threshold_and_keeps_short_gaps():
    daily = np.zeros(150, dtype=np.int64)
    daily[[0, 10, 20, 56, 65, 149]] = 1
    episodes, gaps = episode.find_episodes(daily, min_gap=35)
    assert episodes == [(0, 21), (56, 66), (149, 150)]
    assert gaps == [(21, 56), (66, 149)]
    # Exactly 35 empty bins trigger a cut; short gaps remain inside an episode.
    assert gaps[0][1] - gaps[0][0] == 35


def test_training_cohort_counts_noncore_incident_events_but_relation_requires_core_both_sides():
    # Four nodes appear in training days 0..19; future node 99 is not in cohort.
    rows = np.array([
        [10, 20, 1 * 86400], [10, 20, 1 * 86400], [30, 40, 2 * 86400],
        [10, 99, 21 * 86400], [99, 20, 22 * 86400], [99, 100, 23 * 86400],
    ], dtype=np.float64)
    raw, nodes, events, counts = episode.build_episode_counts(rows, (0, 50), (0, 20))
    assert nodes.tolist() == [10, 20, 30, 40]
    assert raw[21, 0, 1] == 1  # outgoing event to noncore node remains counted
    assert raw[22, 1, 0] == 1  # incoming event from noncore node remains counted
    assert raw[23].sum() == 0  # event with no core endpoint contributes no node count
    assert len(events) == 3  # duplicate training rows remain distinct relation events
    assert counts["source_rows_within_episode"] == 6
    assert counts["source_rows_contributing_to_core_counts"] == 5
    assert counts["source_rows_with_at_least_one_noncore_endpoint"] == 3


def test_normalizer_and_scale_use_train_prefix_only():
    raw = np.array([0, 1, 2, 3, 4], dtype=np.float32).reshape(5, 1, 1)
    mask = np.ones_like(raw, dtype=bool)
    train_bounds = (0, 3)
    expected = episode.fit_train_statistics(raw, mask, train_bounds, episode_start=0)
    changed = raw.copy()
    changed[3:] = 1e8
    actual = episode.fit_train_statistics(changed, mask, train_bounds, episode_start=0)
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual))


def test_prepare_refuses_existing_output_before_opening_source(tmp_path):
    out = tmp_path / "already-there"
    out.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        episode.prepare(tmp_path / "missing-source.gz", out)

