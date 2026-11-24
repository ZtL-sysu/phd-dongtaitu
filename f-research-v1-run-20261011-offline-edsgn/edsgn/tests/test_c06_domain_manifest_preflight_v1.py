import json
import shutil
import uuid
import zipfile

import numpy as np
import pytest

import c06_domain_manifest_preflight_v1 as preflight
import frozen_split_history_robustness_v1 as evaluator
import frozen_split_history_robustness_v2 as evaluator_v2
from edsgn.data import PreparedData, load_collection
from edsgn.scheduling import Policy


def _fixture(root):
    folder = root / f".c06-preflight-toy-{uuid.uuid4().hex}"
    collection = folder / "collection"
    collection.mkdir(parents=True)
    values = np.arange(24, dtype=np.float32).reshape(12, 2, 1) / 10
    targets = values + 11
    for seed in (2, 1):
        mask = np.ones_like(values, dtype=bool)
        mask[(seed + 3) % 12, seed % 2, 0] = False
        raw_values = values.copy()
        metadata = {"event_kind": "synthetic", "partition": "test",
            "splits": {"train": [0, 6], "test": [3, 10], "validation": [3, 10]},
            "count_data": False, "horizons": [1, 2], "history": 3, "hmax": 2,
            "metadata": {"source": "toy-source", "seed": seed}}
        np.savez_compressed(collection / f"test_seed{seed}.npz",
            metadata_json=np.asarray(json.dumps(metadata)), values=np.where(mask, values, 0),
            raw_values=raw_values, targets=targets, input_mask=mask, target_mask=np.ones_like(mask),
            obs_age=np.zeros_like(values), interval_end=np.arange(12), node_ids=np.asarray([7, 9]),
            mean=np.asarray([0.0]), std=np.asarray([1.0]), scale=np.ones((2, 1)),
            events=np.empty((0, 3), dtype=np.int64))
    return folder, collection


def test_npz_preflight_never_opens_outcome_or_value_members_and_matches_real_schema(monkeypatch):
    root = evaluator.ROOT / "logs"
    folder, collection = _fixture(root)
    checkpoint = folder / "checkpoint.pt"
    checkpoint.write_bytes(b"toy checkpoint commitment")
    try:
        forbidden = {"raw_values.npy", "targets.npy", "target_mask.npy"}
        original_open = zipfile.ZipFile.open
        opened = []
        def guarded_open(archive, name, *args, **kwargs):
            member = name.filename if isinstance(name, zipfile.ZipInfo) else name
            if member in forbidden:
                raise AssertionError(f"preflight attempted to decode forbidden member {member}")
            opened.append(member)
            return original_open(archive, name, *args, **kwargs)
        monkeypatch.setattr(zipfile.ZipFile, "open", guarded_open)

        paths, proxies, reads, doc = preflight.domain_document(collection, "test")
        assert [path.name for path in paths] == ["test_seed1.npz", "test_seed2.npz"]
        assert reads == ["metadata_json", "node_ids", "input_mask"] * 2
        assert not forbidden.intersection(opened)
        assert opened.count("values.npy") == 2  # header-only opens; no values decode is audited below
        assert all("values" not in member for member in reads)
        monkeypatch.undo()  # Normal PreparedData loading below may read the toy targets.
        manifest = folder / "domain.json"
        manifest.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        descriptor = preflight.collection_descriptor(paths, proxies, reads, manifest, "test")

        fake_sources = {rel: "a" * 64 for rel in evaluator._SOURCE_PATHS}
        monkeypatch.setattr(evaluator, "_source_hashes", lambda _root=evaluator.ROOT: fake_sources.copy())
        ckpt = {"statistics": {"mean": [0.0], "std": [1.0]},
                "task": {"n_nodes": 2, "n_features": 1, "history": 3,
                         "hmax": 2, "count_data": False},
                "config": {"dataset_path": "train-only-config-path"}}
        recipe = {"schema_version": 1, "seed": 2027,
            "missing": {"kind": "time_block", "rate": 0.3},
            "count_noise_max_abs": 0, "event_time_jitter_bins": 0}
        extras_pre, preflight_descriptor = preflight.build_pretest_freeze_extras(
            checkpoint=checkpoint, ckpt=ckpt, policy=Policy(), dataset_path=collection,
            domain_manifest_path=manifest, recipe=recipe, split="test", root=evaluator.ROOT)
        assert preflight_descriptor["ordered_collection"] == descriptor["ordered_collection"]

        # Now decode the temporary toy bundle through the production loader; the
        # production PreparedData collection must reproduce the frozen domain.
        real_data = load_collection(collection, "test")
        verify = preflight.verify_after_normal_load(descriptor, manifest, collection,
            real_data, "test")
        assert verify["domain_manifest_matches"] and verify["ordered_collection_matches"]
        extras_post = evaluator_v2.build_freeze_extras(checkpoint=checkpoint, ckpt=ckpt,
            policy=Policy(), datasets=real_data, dataset_path=collection,
            domain_manifest_path=manifest, recipe=recipe, split="test", root=evaluator.ROOT)
        assert extras_pre == extras_post
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_metadata_preflight_rejects_collection_order_shape_or_archive_drift():
    folder, collection = _fixture(evaluator.ROOT / "logs")
    try:
        paths, proxies, reads, doc = preflight.domain_document(collection, "test")
        manifest = folder / "domain.json"
        manifest.write_text(json.dumps(doc), encoding="utf-8")
        descriptor = preflight.collection_descriptor(paths, proxies, reads, manifest, "test")
        data = load_collection(collection, "test")
        assert preflight.verify_after_normal_load(descriptor, manifest, collection, data, "test")["trajectory_count"] == 2
        bad = list(reversed(data))
        with pytest.raises(RuntimeError, match="domain|order"):
            preflight.verify_after_normal_load(descriptor, manifest, collection, bad, "test")
        (collection / "test_seed1.npz").write_bytes(b"changed bytes")
        with pytest.raises(RuntimeError, match="path/order/bytes"):
            preflight.verify_after_normal_load(descriptor, manifest, collection, data, "test")
    finally:
        shutil.rmtree(folder, ignore_errors=True)
