"""Build v3 C06 domain/freeze extras without decoding forecast/value arrays.

The loader reads only metadata_json, node_ids, input_mask, and the NPY header
for values.shape. It never opens raw_values, values payload, targets, or
target_mask members. Whole-NPZ SHA-256 fingerprints are byte-level commitments
only; they do not decompress or decode any member.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from edsgn.common import sha256
import frozen_split_history_robustness_v1 as evaluator


class _ShapeOnly:
    def __init__(self, shape):
        self.shape = tuple(int(x) for x in shape)


class MetadataMaskPrepared:
    """Minimal data protocol consumed by the accepted split-history helpers."""
    def __init__(self, path, *, split, read_audit):
        self.path = Path(path).resolve()
        self.read_audit = read_audit
        required = {"metadata_json.npy", "values.npy", "input_mask.npy", "node_ids.npy"}
        with zipfile.ZipFile(self.path, "r") as archive:
            members = set(archive.namelist())
            if not required.issubset(members):
                raise ValueError(f"Prepared bundle lacks required metadata/mask members: {required - members}")
            # The metadata scalar and identity/mask arrays are the only payload
            # members intentionally decoded by this preflight.
            meta_array = self._load_allowed(archive, "metadata_json.npy")
            metadata_json = str(np.asarray(meta_array).item())
            bundle = json.loads(metadata_json)
            self.node_ids = self._load_allowed(archive, "node_ids.npy")
            self.input_mask = np.asarray(self._load_allowed(archive, "input_mask.npy"), dtype=bool)
            shape, value_dtype = self._read_npy_header(archive, "values.npy")

        self.values = _ShapeOnly(shape)
        self.value_dtype = value_dtype.str
        if len(shape) != 3 or self.input_mask.shape != shape:
            raise ValueError("values header and input_mask must agree on [T,N,F]")
        if self.node_ids.shape != (shape[1],):
            raise ValueError("node_ids shape disagrees with values header")
        self.event_kind = str(bundle["event_kind"])
        self.partition = str(bundle.get("partition", "chronological"))
        self.metadata = dict(bundle.get("metadata", {}))
        self.splits = {str(k): tuple(int(v) for v in pair)
                       for k, pair in bundle["splits"].items()}
        self.count_data = bool(bundle["count_data"])
        self.horizons = tuple(int(h) for h in bundle["horizons"])
        self.history = int(bundle["history"])
        self.hmax = int(bundle["hmax"])
        self.declared_split = "validation" if split == "val" else str(split)
        if self.declared_split not in self.splits:
            raise ValueError(f"Bundle lacks declared split {self.declared_split!r}")
        # Audit that forbidden numerical members exist but were never opened.
        self.forbidden_members = ("raw_values.npy", "values.npy", "targets.npy", "target_mask.npy")

    def _load_allowed(self, archive, member):
        self.read_audit.append(member.removesuffix(".npy"))
        with archive.open(member, "r") as stream:
            return np.load(io.BytesIO(stream.read()), allow_pickle=False)

    @staticmethod
    def _read_npy_header(archive, member):
        with archive.open(member, "r") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY header version {version}")
        return tuple(shape), np.dtype(dtype)

    def origins(self, split):
        split = "validation" if split == "val" else str(split)
        if split not in self.splits:
            raise KeyError(f"unknown split {split!r}")
        if self.partition not in (None, "chronological", split):
            return np.empty(0, dtype=np.int64)
        start, stop = self.splits[split]
        first = max(self.history - 1, start - 1)
        last = stop - self.hmax - 1
        return np.arange(first, last + 1, dtype=np.int64) if last >= first else np.empty(0, dtype=np.int64)


def ordered_bundle_paths(dataset_path, split):
    """Mirror engine.datasets_at/data.load_collection file and order rules."""
    path = Path(dataset_path).resolve()
    if path.is_file():
        return [path]
    key = "validation" if split in {"val", "validation"} else str(split)
    if key not in {"train", "validation", "test"}:
        raise ValueError("split must be train, val/validation, or test")
    paths = sorted(path.glob(f"{key}_seed*.npz"))
    if not paths:
        raise FileNotFoundError(f"No ordered {key}_seed*.npz bundles in {path}")
    return paths


def load_metadata_collection(dataset_path, split):
    audit = []
    paths = ordered_bundle_paths(dataset_path, split)
    datasets = [MetadataMaskPrepared(path, split=split, read_audit=audit) for path in paths]
    return paths, datasets, audit


def domain_document(dataset_path, split):
    paths, datasets, audit = load_metadata_collection(dataset_path, split)
    document = evaluator.domain_manifest_from_metadata(datasets, split)
    return paths, datasets, audit, document


def write_domain_manifest_exclusive(domain_path, dataset_path, split):
    """Write the same strict v2 schema from the metadata/mask-only collection."""
    target = Path(domain_path).resolve()
    root = evaluator.ROOT.resolve()
    if not target.is_relative_to(root):
        raise ValueError("Domain manifest output must remain within the run root")
    paths, datasets, audit, document = domain_document(dataset_path, split)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = collection_descriptor(paths, datasets, audit, target, split)
    return target, descriptor


def collection_descriptor(paths, datasets, audit, domain_path, split):
    """Bind ordered source archives, metadata-derived domain and code bytes."""
    root = evaluator.ROOT.resolve()
    if not paths or len(paths) != len(datasets):
        raise ValueError("Descriptor requires the complete ordered collection")
    domain_path = Path(domain_path).resolve()
    sources = ["edsgn/staging/c06_domain_manifest_preflight_v2.py",
               "edsgn/staging/split_history_robustness_v2.py",
               "edsgn/staging/frozen_split_history_robustness_v1.py",
               "edsgn/staging/frozen_split_history_robustness_v3.py",
               "edsgn/src/edsgn/data.py", "edsgn/src/edsgn/engine.py",
               "edsgn/src/edsgn/robustness.py", "edsgn/src/edsgn/prepare.py"]
    return {
        "schema": "edsgn.c06-domain-preflight.v1",
        "split": "val" if split == "validation" else split,
        "ordered_collection": [{"trajectory_index": index,
            "path": Path(path).resolve().relative_to(root).as_posix(),
            "archive_sha256": sha256(path),
            "shape_tnf": list(dataset.values.shape)}
            for index, (path, dataset) in enumerate(zip(paths, datasets))],
        "domain_manifest_path": str(domain_path.relative_to(root)),
        "domain_manifest_sha256": sha256(domain_path),
        "source_sha256": {rel: sha256(root / rel) for rel in sources},
        "decoded_npz_members": list(audit),
        "header_only_npz_members": ["values"],
        "forbidden_decoded_members": ["raw_values", "values", "targets", "target_mask"],
        "normalization_source": "checkpoint.statistics bound from training-only checkpoint",
        "target_dataset_fingerprint_source": "engine.dataset_fingerprint hashes original archive bytes without decoding members",
    }


def build_pretest_freeze_extras(*, checkpoint, ckpt, policy, dataset_path,
                                domain_manifest_path, recipe, split,
                                authorized_recipes=None, root=evaluator.ROOT):
    """Build evaluator extras using only metadata/mask proxies, before test load."""
    paths, datasets, audit = load_metadata_collection(dataset_path, split)
    domain_path = Path(domain_manifest_path).resolve()
    document = evaluator.domain_manifest_from_metadata(datasets, split)
    actual = json.loads(domain_path.read_text(encoding="utf-8"))
    if actual != document:
        raise RuntimeError("Domain manifest differs from the metadata-only ordered collection")
    # Delayed import avoids a module cycle: the evaluator itself imports this
    # loader to verify the already-frozen descriptor before target decoding.
    import frozen_split_history_robustness_v3 as evaluator_v3
    extras = evaluator_v3.build_freeze_extras(checkpoint=checkpoint, ckpt=ckpt,
        policy=policy, datasets=datasets, dataset_path=dataset_path,
        domain_manifest_path=domain_path, recipe=recipe, split=split,
        authorized_recipes=authorized_recipes, root=root)
    descriptor = collection_descriptor(paths, datasets, audit, domain_path, split)
    return extras, descriptor


def verify_after_normal_load(descriptor, domain_path, dataset_path, prepared_datasets, split):
    """Ensure ordinary later PreparedData loading reproduces exact manifest."""
    expected = evaluator.domain_manifest_from_metadata(prepared_datasets, split)
    stored = json.loads(Path(domain_path).read_text(encoding="utf-8"))
    if expected != stored:
        raise RuntimeError("Normal PreparedData load does not reproduce preflight domain")
    if descriptor["domain_manifest_sha256"] != sha256(domain_path):
        raise RuntimeError("Domain manifest changed after metadata-only preflight")
    actual_order = [{"trajectory_index": i, "shape_tnf": list(data.values.shape)}
                    for i, data in enumerate(prepared_datasets)]
    expected_order = [{"trajectory_index": row["trajectory_index"],
                       "shape_tnf": row["shape_tnf"]}
                      for row in descriptor["ordered_collection"]]
    if actual_order != expected_order:
        raise RuntimeError("Normal PreparedData collection order/shape differs from preflight")
    paths = ordered_bundle_paths(dataset_path, split)
    expected_files = descriptor["ordered_collection"]
    if len(paths) != len(expected_files):
        raise RuntimeError("Normal collection file count differs from metadata preflight")
    for path, row in zip(paths, expected_files):
        if (Path(path).resolve().relative_to(evaluator.ROOT.resolve()).as_posix() != row["path"]
                or sha256(path) != row["archive_sha256"]):
            raise RuntimeError("Prepared bundle path/order/bytes changed since metadata preflight")
    return {"domain_manifest_matches": True, "ordered_collection_matches": True,
            "trajectory_count": len(prepared_datasets)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split", choices=("val", "validation", "test"), required=True)
    parser.add_argument("--domain-output", required=True)
    parser.add_argument("--descriptor-output", required=True)
    args = parser.parse_args()
    domain, descriptor = write_domain_manifest_exclusive(
        args.domain_output, args.dataset_path, args.split)
    out = Path(args.descriptor_output).resolve()
    if not out.is_relative_to(evaluator.ROOT.resolve()):
        raise ValueError("Descriptor output must remain within run root")
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(descriptor, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"domain_manifest": str(domain), "descriptor": str(out),
                      "trajectory_count": len(descriptor["ordered_collection"]),
                      "decoded_members": descriptor["decoded_npz_members"]}, indent=2))
