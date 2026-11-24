import numpy as np

from edsgn.data import PreparedData, collate_windows, load_collection, load_dataset


def _toy_data():
    raw = np.arange(12, dtype=np.float32).reshape(6, 2, 1)
    mask = np.ones_like(raw, dtype=bool)
    mask[2, 0, 0] = False
    values = np.where(mask, raw / 10.0, 0.0)
    return PreparedData(
        values=values, targets=raw, input_mask=mask, target_mask=np.ones_like(mask),
        obs_age=np.where(mask, 0.0, 1.0), interval_end=np.arange(1, 7),
        node_ids=np.array([10, 11]), splits={"train": (0, 3), "validation": (3, 5), "test": (5, 6)},
        mean=np.array([0.0]), std=np.array([1.0]), scale=np.ones((2, 1)),
        count_data=True, horizons=(1,), history=2, hmax=1,
        events=np.array([[0, 0, 1], [2, 1, 0], [3, 0, 1], [5, 1, 0]]), event_kind="email",
        metadata={"raw_values": raw},
    )


def test_split_origins_keep_full_hmax_targets_inside_partition():
    data = _toy_data()
    assert data.origins("train").tolist() == [1]
    assert data.origins("validation").tolist() == [2, 3]
    assert data.origins("test").tolist() == [4]
    for split in ("train", "validation", "test"):
        start, stop = data.splits[split]
        for origin in data.origins(split):
            assert start <= origin + 1
            assert origin + data.hmax < stop


def test_window_is_causal_hides_targets_and_event_stream_is_window_local():
    data = _toy_data()
    window = data.window(3)
    assert window["origin"] == 3
    assert window["time_indices"].tolist() == [2, 3]
    assert window["events"].tolist() == [[2, 1, 0], [3, 0, 1]]
    assert "targets" not in window
    assert data.target(3)["target_indices"].tolist() == [4]
    batch = collate_windows([window])
    assert batch["events"][0]["kind"] == "email"
    assert batch["events"][0]["records"].shape == (2, 3)
    assert batch["events"][0]["time_indices"].tolist() == [2, 3]


def test_mask_override_zero_fills_raw_and_normalized_and_recomputes_age():
    data = _toy_data()
    override = np.ones((2, 2, 1), dtype=bool)
    override[0, 1, 0] = False
    window = data.window(3, mask_override=override)
    assert window["input_mask"][0, 1, 0] == 0
    assert window["values"][0, 1, 0] == 0
    assert window["raw_observations"][0, 1, 0] == 0
    assert window["obs_age"][0, 1, 0] == 1
    assert window["obs_age"][1, 1, 0] == 0


def test_real_prepared_bundles_load_with_per_node_scales_and_valid_origins():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for rel, split_counts in (("data/processed/email/email.npz", (528, 74, 155)),
                              ("data/processed/mooc/mooc.npz", (452, 49, 120))):
        data = load_dataset(root / rel)
        assert data.scale.shape == data.values.shape[1:]
        for split, expected in zip(("train", "validation", "test"), split_counts):
            assert len(data.origins(split)) == expected
        assert len(data.origins("val")) == split_counts[1]


def test_real_zero_counts_remain_observed_and_masked_inputs_stay_distinct():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    email = load_dataset(root / "data/processed/email/email.npz")
    assert email.input_mask.all()
    assert np.any(email.raw_values == 0)
    assert np.all(email.input_mask[email.raw_values == 0])

    synthetic = load_dataset(root / "data/processed/synthetic-n64-identity-change0.3-miss0/train_seed0.npz")
    override = np.ones((synthetic.history, *synthetic.values.shape[1:]), dtype=bool)
    override[0, 0, 0] = False
    window = synthetic.window(synthetic.history - 1, mask_override=override)
    assert window["input_mask"][0, 0, 0] == 0
    assert window["values"][0, 0, 0] == 0
    assert window["raw_observations"][0, 0, 0] == 0


def test_official_training_cohorts_and_future_events_are_isolated():
    import csv
    import gzip
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    email = load_dataset(root / "data/processed/email/email.npz")
    email_train_stop = email.splits["train"][1]
    train_email_nodes = set()
    with gzip.open(root / "data/raw/downloads/email-Eu-core-temporal.txt.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            src, dst, timestamp = map(int, line.split())
            if timestamp // 86400 < email_train_stop:
                train_email_nodes.update((src, dst))
    assert email.node_ids.tolist() == sorted(train_email_nodes)
    origin = int(email.origins("val")[0])
    ew = email.window(origin)
    assert np.all(ew["events"][:, 0] >= origin - email.history + 1)
    assert np.all(ew["events"][:, 0] <= origin)
    assert np.any(email.events[:, 0] > origin)

    mooc = load_dataset(root / "data/processed/mooc/mooc.npz")
    mooc_train_stop = mooc.splits["train"][1]
    train_targets = set()
    with (root / "data/raw/mooc_actions.tsv").open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            if int(float(row["TIMESTAMP"]) // 3600) < mooc_train_stop:
                train_targets.add(int(row["TARGETID"]))
    assert mooc.node_ids.tolist() == sorted(train_targets)
    origin = int(mooc.origins("val")[0])
    mw = mooc.window(origin)
    assert np.all(mw["events"][:, 0] >= origin - mooc.history + 1)
    assert np.all(mw["events"][:, 0] <= origin)
    assert np.any(mooc.events[:, 0] > origin)
    assert mooc.metadata["dropout_labels_loaded"] is False


def test_default_synthetic_collection_is_partitioned_and_oracle_separate():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "data/processed/synthetic-n64-identity-change0.3-miss0"
    expected = {"train": 8, "validation": 2, "test": 4}
    for split, count in expected.items():
        items = load_collection(root, "val" if split == "validation" else split)
        assert len(items) == count
        assert all(item.values.shape == (2048, 64, 1) for item in items)
        assert all(item.scale.shape == (64, 1) for item in items)
    assert not list(root.glob("*.oracle.npz"))
    assert len(list((root / "oracle").glob("*.oracle.npz"))) == 14
