import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_dcrnn_training_prior as prior


def _real_npz(path: Path, kind: str, node_ids, splits, events):
    meta = {"event_kind": kind, "partition": "train", "splits": splits}
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(meta)),
        node_ids=np.asarray(node_ids),
        events=np.asarray(events, dtype=np.int64).reshape((-1, 3)),
        # Deliberately present but must not be accessed by the builder.
        targets=np.asarray(["not-needed"]),
    )


def test_email_preserves_duplicate_direction_and_ignores_outside_train(tmp_path):
    path = tmp_path / "email.npz"
    _real_npz(path, "email", [10, 11, 12], {"train": [0, 3], "validation": [3, 4], "test": [4, 5]},
              [[0, 0, 1], [0, 0, 1], [1, 1, 0], [2, 1, 2], [4, 2, 0]])
    a, ids, detail = prior.build_real_prior(path, "email")
    assert ids.tolist() == [10, 11, 12]
    assert a[0, 1] == 2 and a[1, 0] == 1 and a[1, 2] == 1
    assert a[2, 0] == 0 and np.all(np.diag(a) == 0)
    assert detail["training_split_bounds"] == [0, 3]


def test_mooc_deduplicates_clicks_per_user_and_projects_shared_users(tmp_path):
    path = tmp_path / "mooc.npz"
    _real_npz(path, "mooc", [100, 101, 102], {"train": [0, 3], "validation": [3, 4], "test": [4, 5]},
              [[0, 50001, 0], [0, 50001, 0], [1, 50001, 1],
               [2, 50002, 0], [2, 50002, 1], [4, 50001, 2]])
    a, _, detail = prior.build_real_prior(path, "mooc")
    assert a[0, 1] == 2 and a[1, 0] == 2
    assert not a[0, 0] and not a[1, 1] and not a[0, 2]
    assert detail["distinct_training_users"] == 2
    assert detail["repeat_clicks_deduplicated_within_user_activity"]


def _synthetic_file(path: Path, node_ids, values, mask):
    meta = {"partition": "train", "metadata": {"split": "train", "seed": int(path.stem[-1])},
            "splits": {"train": [0, len(values)], "validation": [len(values), len(values)], "test": [len(values), len(values)]}}
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(meta)), node_ids=np.asarray(node_ids),
                        values=np.asarray(values, dtype=np.float32).reshape((-1, len(node_ids), 1)),
                        input_mask=np.asarray(mask, dtype=bool).reshape((-1, len(node_ids), 1)),
                        targets=np.asarray(["not-needed"]))


def test_synthetic_pairs_do_not_cross_trajectory_and_mask_both_ends(tmp_path):
    _synthetic_file(tmp_path / "train_seed0.npz", [0, 1, 2],
                    [[0, 0, 0], [10, 1, 0]], [[1, 1, 1], [1, 1, 1]])
    # If files were concatenated, the boundary would add a spurious pair 10 -> 10.
    _synthetic_file(tmp_path / "train_seed1.npz", [0, 1, 2],
                    [[0, 10, 0], [0, 10, 0]], [[1, 1, 1], [1, 0, 1]])
    a, _, stats, sources = prior.build_synthetic_prior(tmp_path)
    assert len(sources) == 2
    assert stats["training_lag_pairs_per_trajectory_sum"] == 2
    # Only one observed pair remains after the second trajectory's destination mask;
    # n<2 means zero correlation. Constant node 2 also has zero variance.
    assert a[0, 1] == 0 and a[2, 1] == 0
    assert stats["lag_pair_count_max"] == 2


def test_synthetic_rejects_node_order_mismatch(tmp_path):
    _synthetic_file(tmp_path / "train_seed0.npz", [0, 1], [[0, 0], [1, 1]], [[1, 1], [1, 1]])
    _synthetic_file(tmp_path / "train_seed1.npz", [1, 0], [[0, 0], [1, 1]], [[1, 1], [1, 1]])
    with pytest.raises(ValueError, match="inconsistent node ordering"):
        prior.build_synthetic_prior(tmp_path)


def test_zero_variance_and_single_valid_pair_produce_zero_edges():
    sums = {key: np.zeros((2, 2), dtype=np.float64) for key in ("count", "sx", "sy", "sxx", "syy", "sxy")}
    prior._add_lag_pair_sums(sums, np.asarray([[1.0, 4.0]]), np.asarray([[2.0, 4.0]]),
                             np.ones((1, 2), bool), np.ones((1, 2), bool))
    a, _ = prior.correlations_from_sums(sums)
    assert np.array_equal(a, np.zeros((2, 2), dtype=np.float32))


def test_artifact_writer_refuses_existing_destination(tmp_path):
    destination = tmp_path / "email.npz"
    destination.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prior.build_one("email", tmp_path / "not-read.npz", tmp_path)
    assert destination.read_bytes() == b"preserve"
