"""Prepare the accepted, separate source-episode email supplement bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn" / "src"))

from edsgn.prepare import DAY_SECONDS, fit_normalizer, raw_record_stream, write_bundle, utc_now


RAW_DEFAULT = ROOT / "data/raw/downloads/email-Eu-core-temporal.txt.gz"
OUT_DEFAULT = ROOT / "data/processed/email-episode-v1"
HISTORY, HMAX, MIN_GAP = 28, 7, 35
SPLIT_FRACTIONS = (0.70, 0.10)
EXPECTED_EPISODE = (0, 526)
EXPECTED_GAP = (526, 797)
EXPECTED_SPLITS = {"train": (0, 368), "validation": (368, 420), "test": (420, 526)}
EXPECTED_ORIGINS = {"train": 334, "validation": 46, "test": 100}
EXPECTED_TRAIN_NODES = 893
PROTOCOL_PATH = ROOT / "protocols/email-episode-supplement-v1.md"
ACCEPTANCE_PATH = ROOT / "protocols/email-episode-supplement-acceptance-v2.md"
REVIEW_PATH = ROOT / "reviews/email-episode-supplement-review-v1.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_episodes(daily_counts: np.ndarray, min_gap: int = MIN_GAP) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return half-open active-span episodes and qualifying empty-gap spans."""
    daily_counts = np.asarray(daily_counts)
    if daily_counts.ndim != 1 or np.any(daily_counts < 0) or min_gap < 1:
        raise ValueError("daily_counts must be a nonnegative vector and min_gap positive")
    active = np.flatnonzero(daily_counts > 0)
    if not active.size:
        return [], []
    episodes, gaps = [], []
    start = previous = int(active[0])
    for value in active[1:]:
        day = int(value)
        gap_start, gap_size = previous + 1, day - previous - 1
        if gap_size >= min_gap:
            episodes.append((start, previous + 1))
            gaps.append((gap_start, day))
            start = day
        previous = day
    episodes.append((start, previous + 1))
    return episodes, gaps


def split_episode(start: int, stop: int) -> dict[str, tuple[int, int]]:
    """Chronological 70/10/20 split using floor on the first two partitions."""
    length = stop - start
    train_len = int(np.floor(SPLIT_FRACTIONS[0] * length))
    validation_len = int(np.floor(SPLIT_FRACTIONS[1] * length))
    return {
        "train": (start, start + train_len),
        "validation": (start + train_len, start + train_len + validation_len),
        "test": (start + train_len + validation_len, stop),
    }


def legal_origin_count(bounds: tuple[int, int], *, episode_start: int, history: int = HISTORY, hmax: int = HMAX) -> int:
    """Count origins using PreparedData split semantics: targets stay in split."""
    lo, hi = bounds
    first = max(episode_start + history - 1, lo - 1)
    last = hi - hmax - 1
    return max(0, last - first + 1)


def fit_train_statistics(raw: np.ndarray, mask: np.ndarray, train_bounds: tuple[int, int], *, episode_start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = train_bounds
    train_slice = slice(lo - episode_start, hi - episode_start)
    if train_slice.start < 0 or train_slice.stop > len(raw):
        raise ValueError("train bounds are outside the supplied episode arrays")
    return fit_normalizer(raw[train_slice], mask[train_slice], True)


def choose_earliest_feasible(episodes: Iterable[tuple[int, int]]) -> tuple[int, tuple[int, int], dict[str, tuple[int, int]], dict[str, int]]:
    for index, episode in enumerate(episodes):
        splits = split_episode(*episode)
        origins = {name: legal_origin_count(bounds, episode_start=episode[0]) for name, bounds in splits.items()}
        if all(count > 0 for count in origins.values()):
            return index, episode, splits, origins
    raise ValueError("no chronological episode has legal origins in all three partitions")


def build_episode_counts(rows: np.ndarray, episode: tuple[int, int], train_split: tuple[int, int]):
    """Build count tensor/cohort/events from source rows without dropping core incidences."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError("rows must have columns SRC DST UNIXTS")
    src_all, dst_all = rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64)
    ts_all = rows[:, 2]
    day_all = np.floor(ts_all / DAY_SECONDS).astype(np.int64)
    start, stop = episode
    within = (day_all >= start) & (day_all < stop)
    train = within & (day_all >= train_split[0]) & (day_all < train_split[1])
    train_nodes = np.unique(np.concatenate([src_all[train], dst_all[train]]))
    if not len(train_nodes):
        raise ValueError("episode training partition has no endpoint nodes")
    node_to_idx = {int(node): index for index, node in enumerate(train_nodes)}
    raw = np.zeros((stop - start, len(train_nodes), 2), dtype=np.float32)
    relation_events = []
    rows_with_noncore_endpoint = 0
    rows_contributing_to_counts = 0
    for row_i in np.flatnonzero(within):
        day = int(day_all[row_i])
        src, dst = int(src_all[row_i]), int(dst_all[row_i])
        src_core, dst_core = src in node_to_idx, dst in node_to_idx
        if src_core:
            raw[day - start, node_to_idx[src], 1] += 1.0
        if dst_core:
            raw[day - start, node_to_idx[dst], 0] += 1.0
        if src_core or dst_core:
            rows_contributing_to_counts += 1
        if src_core and dst_core:
            relation_events.append((day - start, node_to_idx[src], node_to_idx[dst]))
        else:
            rows_with_noncore_endpoint += 1
    events = np.asarray(relation_events, dtype=np.int64).reshape((-1, 3))
    return raw, train_nodes, events, {
        "source_rows_within_episode": int(within.sum()),
        "source_rows_contributing_to_core_counts": rows_contributing_to_counts,
        "source_rows_with_at_least_one_noncore_endpoint": rows_with_noncore_endpoint,
        "relation_event_rows_both_endpoints_core": int(len(events)),
    }


def prepare(raw_path: Path = RAW_DEFAULT, out_dir: Path = OUT_DEFAULT) -> dict:
    raw_path, out_dir = Path(raw_path), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing supplement output: {out_dir}")
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    source_rows = []
    for _, src, dst, ts in raw_record_stream(raw_path):
        source_rows.append((src, dst, ts))
    rows = np.asarray(source_rows, dtype=np.float64)
    ts = rows[:, 2]
    n_calendar = int(np.floor(ts.max() / DAY_SECONDS))
    day = np.floor(ts / DAY_SECONDS).astype(np.int64)
    retained = day < n_calendar
    daily = np.bincount(day[retained], minlength=n_calendar)[:n_calendar]
    episodes, gaps = find_episodes(daily, MIN_GAP)
    candidate_index, candidate, splits, origins = choose_earliest_feasible(episodes)
    if candidate != EXPECTED_EPISODE or not gaps or gaps[0] != EXPECTED_GAP:
        raise RuntimeError(f"source falsifier: expected first episode/gap {EXPECTED_EPISODE}/{EXPECTED_GAP}, got {candidate}/{gaps[:1]}")
    if splits != EXPECTED_SPLITS or origins != EXPECTED_ORIGINS:
        raise RuntimeError(f"source falsifier: split/origin counts changed: {splits}/{origins}")

    raw, node_ids, events, counts = build_episode_counts(rows[retained], candidate, splits["train"])
    if len(node_ids) != EXPECTED_TRAIN_NODES:
        raise RuntimeError(f"source falsifier: expected {EXPECTED_TRAIN_NODES} train-defined nodes; got {len(node_ids)}")
    mask = np.ones(raw.shape, dtype=bool)
    train_start, train_stop = splits["train"]
    train_slice = slice(train_start - candidate[0], train_stop - candidate[0])
    mean, std, scale = fit_train_statistics(raw, mask, splits["train"], episode_start=candidate[0])
    # Explicitly assure all fitted quantities are train-only using the same helper.
    expected_stats = fit_normalizer(raw[train_slice].copy(), mask[train_slice].copy(), True)
    if any(not np.array_equal(a, b) for a, b in zip((mean, std, scale), expected_stats)):
        raise AssertionError("train-only fitting reproducibility check failed")

    page_manifest_path = ROOT / "data/processed/email/manifest.json"
    original_manifest = json.loads(page_manifest_path.read_text(encoding="utf-8"))
    input_source_rows = int(len(rows))
    raw_split_stats = {}
    for name, (lo, hi) in splits.items():
        in_split = (day >= lo) & (day < hi)
        raw_split_stats[name] = {
            "bins": [lo, hi], "length_days": hi - lo,
            "source_event_rows": int(in_split.sum()),
            "active_days": int(np.count_nonzero(daily[lo:hi])),
            "legal_origins": origins[name],
        }
    train_defined_edges = (day >= train_start) & (day < train_stop)
    source_active_nodes = set(rows[train_defined_edges, :2].astype(np.int64).reshape(-1).tolist())
    if source_active_nodes != set(node_ids.tolist()):
        raise RuntimeError("training cohort source cross-check failed")

    dataset_meta = {
        "source": "SNAP email-Eu-core-temporal; same source as primary email regime",
        "regime_name": "email-episode-v1",
        "claim_scope": "within-source coverage-conditioned temporal supplement; not an independent dataset/source confirmation",
        "time_unit": "seconds_from_relative_zero",
        "bin_width_seconds": DAY_SECONDS,
        "bin_convention": "half-open [b*86400,(b+1)*86400); preserve original days; no time compression",
        "episode_day_bounds_half_open_original_calendar": [candidate[0], candidate[1]],
        "long_gap_removed_only_between_episodes": [gaps[0][0], gaps[0][1]],
        "internal_zero_calendar_bins_preserved": True,
        "n_source_records_full_file": input_source_rows,
        "n_source_records_in_candidate_episode": counts["source_rows_within_episode"],
        "n_bins_total": int(candidate[1] - candidate[0]),
        "n_nodes_training_defined": int(len(node_ids)),
        **counts,
        "n_events_with_noncore_endpoint_excluded_from_relation_events": counts["source_rows_with_at_least_one_noncore_endpoint"],
        "repeated_source_rows_preserved": True,
        "calendar_features_added": False,
        "raw_values_channel_order": ["incoming_count", "outgoing_count"],
        "input_mask_semantics": "all true for released-bin counts; source zero is an observed count zero, not an assertion of real-world absence",
    }

    # All source/protocol falsifiers pass before creating this never-overwrite output path.
    out_dir.mkdir(parents=True, exist_ok=False)
    bundle_info = write_bundle(
        out_dir / "email-episode.npz", raw_values=raw, input_mask=mask,
        interval_end=(np.arange(len(raw), dtype=np.float64) + candidate[0] + 1) * DAY_SECONDS,
        node_ids=node_ids, events=events, mean=mean, std=std, scale=scale,
        count_data=True, horizons=(1, 3, 7), history=HISTORY, hmax=HMAX,
        splits=splits, event_kind="email", metadata=dataset_meta, partition="chronological",
    )
    manifest = {
        "schema_version": 1, "dataset": "email-episode-v1", "created_at_utc": utc_now(),
        "same_source_as_primary_email": True,
        "independent_dataset_claim": False,
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "acceptance": {"path": str(ACCEPTANCE_PATH), "sha256": sha256(ACCEPTANCE_PATH)},
        "independent_review": {"path": str(REVIEW_PATH), "sha256": sha256(REVIEW_PATH)},
        "preparation_implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "source": {"url": original_manifest["download"]["url"], "path": str(raw_path.resolve()),
                   "bytes": raw_path.stat().st_size, "sha256": sha256(raw_path),
                   "records": input_source_rows, "format": "gzip of whitespace-separated SRC DST UNIXTS, no header"},
        "source_manifest_sha256": sha256(page_manifest_path),
        "license_and_use": original_manifest.get("license_and_use", {}),
        "selection_rule": {"minimum_empty_run_days": MIN_GAP, "history": HISTORY, "hmax": HMAX,
                           "selection": "earliest chronological episode with legal origins in every floor-based 70/10/20 split; no activity or model-score ranking",
                           "excluded_calendar_runs": [[int(a), int(b)] for a,b in gaps]},
        "episode": {"index": candidate_index, "day_bounds_half_open_original_calendar": list(candidate),
                    "split": {k: list(v) for k,v in splits.items()}, "legal_origins": origins,
                    "split_source_coverage": raw_split_stats},
        "preprocessing": {"normalization": "log1p(raw_count)", "fit_partition": "train only",
                          "mean": mean.tolist(), "std": std.tolist(), "raw_residual_scale": scale.tolist(),
                          "fit_input_day_bounds": list(splits["train"]), "bundle": bundle_info,
                          "counts": dataset_meta},
        "decision_roles": {"model_and_preprocessing_fit": "train [0,368) only",
                           "future_checkpoint_and_scheduler_calibration": "validation [368,420) only; not performed during preparation",
                           "test": "[420,526); no predictions or model metrics accessed during source coverage selection/preparation; metrics only after future freeze"},
        "test_metrics_access": "none during source coverage selection or bundle preparation; no model predictions or test metrics read",
        "test_policy": "test partition is not accessed for preprocessing statistics, parameter selection, or checkpoint/scheduler calibration",
        "temporal_limit": "Within-source supplement; selected by a source-coverage rule after observing the primary validation gap; not an independent dataset or external replication.",
    }
    (out_dir / "data-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)
    result = prepare(args.raw, args.output_dir)
    print(json.dumps({"bundle": result["preprocessing"]["bundle"], "episode": result["episode"],
                      "train_defined_nodes": result["preprocessing"]["counts"]["n_nodes_training_defined"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
