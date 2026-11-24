"""Metadata-only admission for the prospective case_v3 frozen-test job.

This helper binds a completed, validation-only training record and a frozen
case selection/domain to one exact global-v5 queued job. It never opens NPZ
members or checkpoints. No case-capable dispatcher is currently accepted, so
this validates preparation only and does not authorize launching a job.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from . import ordinary_frozen_test_admission_v1 as ordinary
    from . import global_frozen_test_completion_v5 as global_completion
except ImportError:
    import ordinary_frozen_test_admission_v1 as ordinary
    import global_frozen_test_completion_v5 as global_completion

SOURCE_REL = "edsgn/staging/case_frozen_test_admission_v2.py"
DRIVER_ADMISSION_REL = "edsgn/staging/case_frozen_test_admission_v1.py"
GLOBAL_REL = "edsgn/staging/global_frozen_test_completion_v5.py"
DRIVER_GLOBAL_REL = "edsgn/staging/global_frozen_test_completion_v3.py"
ORDINARY_REL = "edsgn/staging/ordinary_frozen_test_admission_v1.py"
DRIVER_REL = "edsgn/staging/frozen_case_signal_sensitivity_v3.py"
C06_REL = "edsgn/staging/c06_domain_manifest_preflight_v4.py"
KERNEL_REL = "edsgn/staging/case_signal_sensitivity_kernel_v2.py"
H2_KERNEL_REL = "edsgn/staging/h2_matched_graph_retention_v2.py"
KERNEL_ACCEPTANCE_REL = "reviews/case-signal-sensitivity-kernel-v2-root-engineering-v1.json"
SELECTION_SCOPE = "root_accepted_case_signal_validation_selection"
DATASETS = {"synthetic", "email", "mooc", "email-episode"}
CASE_COLLECTIONS = {
    "synthetic": [f"data/processed/synthetic-n64-identity-change0.3-miss0/test_seed{s}.npz" for s in (200, 201, 202, 203)],
    "email": ["data/processed/email/email.npz"],
    "mooc": ["data/processed/mooc/mooc.npz"],
    "email-episode": ["data/processed/email-episode-v1/email-episode.npz"],
}


class AdmissionError(ValueError):
    """Case inputs are incomplete, stale, or inconsistent with frozen records."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise AdmissionError(f"duplicate JSON key {k!r} in {path}")
            out[k] = v
        return out
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(AdmissionError(f"invalid JSON constant {x}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"JSON object required: {path}")
    return value


def _resolve(root: Path, raw: str, *, exists=True) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise AdmissionError("path must be a nonempty string")
    p = Path(raw)
    q = (p if p.is_absolute() else root / p).resolve(strict=exists)
    if not q.is_relative_to(root):
        raise AdmissionError(f"path escapes RUN: {raw}")
    lexical = p if p.is_absolute() else root / p
    try:
        rel = lexical.absolute().relative_to(root)
        cur = root
        for part in rel.parts:
            cur = cur / part
            if cur.is_symlink():
                raise AdmissionError(f"symlink forbidden: {raw}")
    except ValueError:
        raise AdmissionError(f"path escapes RUN: {raw}")
    return q


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _ref(root: Path, ref: Any, label: str) -> tuple[Path, str]:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise AdmissionError(f"{label} must be exactly path/sha256")
    path = _resolve(root, ref["path"])
    digest = ref["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or _sha(path) != digest:
        raise AdmissionError(f"{label} hash mismatch")
    return path, digest


def _add(root: Path, consumed: dict[str, str], path: Path, digest: str | None = None):
    rel = _rel(root, path)
    digest = digest or _sha(path)
    if rel in consumed and consumed[rel] != digest:
        raise AdmissionError(f"conflicting digest for {rel}")
    consumed[rel] = digest


def _refs(root: Path, values: Any, label: str) -> dict[str, str]:
    if not isinstance(values, list):
        raise AdmissionError(f"{label} must be a list")
    out = {}
    for ref in values:
        p, h = _ref(root, ref, label)
        r = _rel(root, p)
        if r in out:
            raise AdmissionError(f"duplicate {label}: {r}")
        out[r] = h
    return out


def _dataset_identity(dataset_id: str, source_name: str, event_kind: str, partition: str) -> bool:
    if dataset_id == "synthetic":
        return source_name == "identity" and event_kind == "synthetic" and partition == "test"
    if dataset_id in {"email", "email-episode"}:
        return source_name == "SNAP email-Eu-core-temporal" and event_kind == "email" and partition == "chronological"
    return source_name == "SNAP act-mooc user action log" and event_kind == "mooc" and partition == "chronological"


def _validate_case_domain(root: Path, dataset_id: str, domain_ref: dict, descriptor_ref: dict,
                          consumed: dict[str, str]) -> tuple[dict, dict, Path]:
    domain_path, domain_sha = _ref(root, domain_ref, "case domain")
    descriptor_path, descriptor_sha = _ref(root, descriptor_ref, "C06 descriptor")
    domain = _json(domain_path)
    descriptor = _json(descriptor_path)
    ds = domain.get("dataset")
    if (domain.get("schema") != "edsgn.split-history-domain.v1" or not isinstance(ds, dict)
            or ds.get("partition") != "test"
            or not _dataset_identity(dataset_id, ds.get("source_name"), ds.get("event_kind"), ds.get("partition"))):
        raise AdmissionError("case domain identity/split is not the declared test collection")
    if (descriptor.get("schema") != "edsgn.c06-domain-preflight.v1" or descriptor.get("split") != "test"
            or descriptor.get("domain_manifest_sha256") != domain_sha
            or _resolve(root, descriptor.get("domain_manifest_path", "")) != domain_path):
        raise AdmissionError("C06 descriptor does not bind the exact case domain")
    rows = descriptor.get("ordered_collection")
    trajectories = domain.get("trajectories")
    if not isinstance(rows, list) or not rows or not isinstance(trajectories, list) or len(rows) != len(trajectories):
        raise AdmissionError("domain and C06 ordered collection must contain the same nonempty trajectory set")
    expected = []
    for i, (tr, row) in enumerate(zip(trajectories, rows)):
        if not isinstance(tr, dict) or not isinstance(row, dict):
            raise AdmissionError("malformed ordered trajectory metadata")
        if row.get("trajectory_index") != i or tr.get("trajectory_index") != i:
            raise AdmissionError("trajectory indices must be complete and ordered")
        tp = _resolve(root, row.get("path", ""))
        if row.get("archive_sha256") != _sha(tp):
            raise AdmissionError("C06 row does not hash-bind its exact test trajectory")
        expected.append({"trajectory_index": i, "path": _rel(root, tp), "archive_sha256": _sha(tp)})
    if descriptor.get("ordered_collection") != rows:
        raise AdmissionError("unordered or modified C06 trajectory list")
    first_path = _resolve(root, rows[0]["path"])
    if ds.get("event_kind") == "synthetic":
        collection = first_path.parent
        # Match the loader's nonrecursive synthetic test_seed*.npz selection.
        actual = sorted(collection.glob("test_seed*.npz"), key=lambda p: p.name)
    else:
        if len(rows) != 1:
            raise AdmissionError("chronological case collection must resolve to one archive")
        collection = first_path
        actual = [collection]
    if [_rel(root, p) for p in actual] != [_rel(root, _resolve(root, r["path"])) for r in rows]:
        raise AdmissionError("C06 descriptor does not enumerate the exact ordered test trajectory set")
    if [_rel(root, p) for p in actual] != CASE_COLLECTIONS[dataset_id]:
        raise AdmissionError("case domain paths differ from the reviewed per-dataset frozen collections")
    expected_seeds = [200, 201, 202, 203] if dataset_id == "synthetic" else [0]
    if [tr.get("source_trajectory_seed") for tr in trajectories] != expected_seeds:
        raise AdmissionError("case domain trajectory-seed order differs from the reviewed collection")
    source_map = descriptor.get("source_sha256")
    if not isinstance(source_map, dict):
        raise AdmissionError("C06 descriptor must bind its metadata/code source inventory")
    for rel, h in source_map.items():
        p = _resolve(root, rel)
        if _sha(p) != h:
            raise AdmissionError(f"stale C06 source inventory: {rel}")
        _add(root, consumed, p, h)
    for p in actual:
        _add(root, consumed, p)
    _add(root, consumed, domain_path, domain_sha)
    _add(root, consumed, descriptor_path, descriptor_sha)
    return domain, descriptor, collection


def _validate_selection(root: Path, dataset_id: str, selection_ref: dict, acceptance_ref: dict,
                        config: dict, checkpoint_sha: str, source_sha: str,
                        training_binding_path: Path, freeze: dict,
                        consumed: dict[str, str]) -> dict:
    selection_path, selection_sha = _ref(root, selection_ref, "validation case selection")
    acceptance_path, acceptance_sha = _ref(root, acceptance_ref, "case selection root acceptance")
    selection = _json(selection_path)
    acceptance = _json(acceptance_path)
    if (selection.get("schema_version") != 1 or selection.get("scope") != SELECTION_SCOPE
            or selection.get("status") != "accepted" or selection.get("accepted_by") != "root"
            or selection.get("test_feedback_used") is not False):
        raise AdmissionError("case selection must be explicitly root-accepted and validation-only")
    if (selection.get("dataset_id") != dataset_id or selection.get("variant") != "V4"
            or selection.get("model_seed") != 0 or selection.get("split") != "validation"):
        raise AdmissionError("case selection dataset/variant/seed/split mismatch")
    if selection.get("checkpoint_sha256") != checkpoint_sha:
        raise AdmissionError("case selection checkpoint does not match frozen execution checkpoint")
    if selection.get("source_manifest_sha256") != source_sha:
        raise AdmissionError("case selection source manifest differs from current frozen driver inputs")
    if selection.get("training_dataset_path") != config.get("dataset_path"):
        raise AdmissionError("case selection training dataset differs from selected training config")
    required = {"dataset_fingerprint", "source_manifest_sha256", "node_order_sha256",
                "full_validation_origin_count", "validation_origin_record", "selection_rule",
                "focal_node", "neighbor_node"}
    if not required.issubset(selection):
        raise AdmissionError("case selection lacks required validation provenance and node identities")
    if (type(selection["full_validation_origin_count"]) is not int or selection["full_validation_origin_count"] <= 0
            or selection["focal_node"] == selection["neighbor_node"]):
        raise AdmissionError("case selection origin count or focal/neighbor IDs are invalid")
    expected_rules = {"focal": "largest_training_activity_score_then_node_id",
                      "neighbor": "largest_mean_valid_neighbor_weight_in_full_validation_then_node_id"}
    if selection["selection_rule"] != expected_rules:
        raise AdmissionError("case selection rule differs from the preregistered validation-only rule")
    for label in ("focal_node", "neighbor_node"):
        node = selection[label]
        if (not isinstance(node, dict) or set(node) != {"index", "node_id"}
                or type(node.get("index")) is not int or node["index"] < 0
                or node.get("node_id") is None):
            raise AdmissionError(f"{label} must bind an integer node index and real node ID")
    if selection["focal_node"]["index"] == selection["neighbor_node"]["index"]:
        raise AdmissionError("focal and neighbor indices must be distinct")
    origin_path, origin_sha = _ref(root, selection["validation_origin_record"], "validation origin record")
    origin = _json(origin_path)
    if (origin.get("split") != "validation" or origin.get("origins_complete_and_ordered") is not True
            or origin.get("full_validation_origin_count") != selection["full_validation_origin_count"]
            or origin.get("checkpoint_sha256") != checkpoint_sha
            or origin.get("dataset_fingerprint") != selection["dataset_fingerprint"]
            or origin.get("source_manifest_sha256") != selection["source_manifest_sha256"]
            or origin.get("node_order_sha256") != selection["node_order_sha256"]):
        raise AdmissionError("validation-origin evidence disagrees with frozen selection")
    if (acceptance.get("status") != "accepted" or acceptance.get("accepted_by") != "root"
            or acceptance.get("verdict") != "pass" or acceptance.get("test_feedback_used") is not False):
        raise AdmissionError("selection root receipt must be accepted by root without test feedback")
    evidence = acceptance.get("evidence_sha256")
    required_evidence = {_rel(root, selection_path): selection_sha,
                         _rel(root, origin_path): origin_sha,
                         _rel(root, training_binding_path): _sha(training_binding_path)}
    actual_binding = _json(training_binding_path)
    for field in ("dataset_sources", "code_sources"):
        if not isinstance(actual_binding.get(field), list):
            raise AdmissionError(f"training binding lacks {field} inventory")
        for source_ref in actual_binding[field]:
            p, h = _ref(root, source_ref, f"training binding {field} source")
            required_evidence[_rel(root, p)] = h
    config_source = actual_binding.get("config_source")
    if not isinstance(config_source, dict):
        raise AdmissionError("training binding lacks config source")
    config_source_path = _resolve(root, config_source.get("path", ""))
    required_evidence[_rel(root, config_source_path)] = config_source.get("sha256")
    checkpoint_source = _resolve(root, str(Path(config.get("output_dir", "")) / "checkpoint.pt"))
    required_evidence[_rel(root, checkpoint_source)] = checkpoint_sha
    if not isinstance(evidence, dict) or any(evidence.get(k) != v for k, v in required_evidence.items()):
        raise AdmissionError("root acceptance does not attest this exact validation selection")
    for rel, digest in evidence.items():
        p = _resolve(root, rel)
        if _sha(p) != digest:
            raise AdmissionError(f"selection acceptance evidence is stale: {rel}")
        _add(root, consumed, p, digest)
    for path, digest in ((selection_path, selection_sha), (acceptance_path, acceptance_sha),
                         (origin_path, origin_sha)):
        _add(root, consumed, path, digest)
    return selection


def _matrix(root: Path, matrix_ref: dict, consumed: dict[str, str]) -> tuple[dict, dict]:
    path, digest = _ref(root, matrix_ref, "case admission matrix")
    doc = _json(path)
    if (set(doc) != {"schema_version", "scope", "status", "source_refs", "jobs"}
            or doc.get("schema_version") != 1
            or doc.get("scope") != "prospective_case_pretest_job_admission_matrix"
            or doc.get("status") != "accepted_before_test"):
        raise AdmissionError("case admission matrix schema/scope/status mismatch")
    _add(root, consumed, path, digest)
    src = _refs(root, doc["source_refs"], "matrix source")
    required = {SOURCE_REL, DRIVER_ADMISSION_REL, GLOBAL_REL, DRIVER_GLOBAL_REL, ORDINARY_REL, DRIVER_REL, C06_REL, KERNEL_REL, H2_KERNEL_REL,
                KERNEL_ACCEPTANCE_REL}
    if not required.issubset(src):
        raise AdmissionError("matrix source_refs omit a required case source")
    # Dispatcher is intentionally not pinned: no reviewed case-capable dispatcher exists yet.
    for rel, sha in src.items():
        p = _resolve(root, rel)
        if _sha(p) != sha:
            raise AdmissionError(f"stale matrix source: {rel}")
        _add(root, consumed, p, sha)
    rows = doc["jobs"]
    if not isinstance(rows, list) or not rows:
        raise AdmissionError("case matrix must have rows")
    ids = set()
    for row in rows:
        keys = {"job_id", "driver", "method_id", "dataset_id", "seed_id", "training_variant",
                "training_preparation", "expected_driver_arguments", "output_dir", "phase_receipt_alias_path",
                "case_selection_path", "case_domain_path", "c06_descriptor_path"}
        if not isinstance(row, dict) or set(row) != keys:
            raise AdmissionError("case matrix row schema mismatch")
        if (row["driver"] != "case_v3" or row["training_variant"] != "V4"
                or row["dataset_id"] not in DATASETS or type(row["seed_id"]) is not int or row["seed_id"] != 0
                or not all(isinstance(row[k], str) and row[k] for k in ("job_id", "method_id"))):
            raise AdmissionError("case matrix must identify case_v3, V4, model seed zero and a known dataset")
        if row["job_id"] in ids:
            raise AdmissionError("duplicate case matrix job_id")
        ids.add(row["job_id"])
        out = _resolve(root, row["output_dir"], exists=False)
        alias = _resolve(root, row["phase_receipt_alias_path"], exists=False)
        if not _rel(root, out).startswith("runs/") or alias.parent.name != "evidence" or not _rel(root, alias).startswith("runs/"):
            raise AdmissionError("case output and alias must be under RUN runs")
        argv = row["expected_driver_arguments"]
        if not isinstance(argv, list) or len(argv) % 2 or not all(isinstance(x, str) and x for x in argv):
            raise AdmissionError("case argv must be flag/value pairs")
        if not isinstance(row["training_preparation"], dict):
            raise AdmissionError("case training preparation must be an object")
    return doc, rows


def validate_admission(root: str | Path, binding_ref: dict, job_doc: dict,
                       job_path: str | Path) -> dict[str, str]:
    """Validate one prospective case_v3 job; return strict snapshot path→SHA refs."""
    root = Path(root).resolve(strict=True)
    consumed: dict[str, str] = {}
    admission_path, admission_sha = _ref(root, binding_ref, "case admission binding")
    admission = _json(admission_path)
    required_fields = {"schema_version", "scope", "status", "accepted_by", "test_feedback_used",
                       "matrix", "matrix_job_id", "global_test_queue", "config", "training_binding",
                       "training_phase_receipt", "training_summary", "checkpoint", "freeze", "policy_config",
                       "candidate_selection", "root_numeric_acceptance", "input_refs", "case_selection",
                       "case_domain", "c06_descriptor", "selection_root_acceptance"}
    if set(admission) != required_fields or admission.get("schema_version") != 1:
        raise AdmissionError("case admission binding exact schema mismatch")
    if (admission.get("scope") != "case_pretest_job_admission" or admission.get("status") != "accepted_by_root"
            or admission.get("accepted_by") != "root" or admission.get("test_feedback_used") is not False):
        raise AdmissionError("case admission must be root-accepted without test feedback")
    _add(root, consumed, admission_path, admission_sha)
    matrix_ref = admission["matrix"]
    ordinary._official_gate(root, matrix_ref, consumed)
    matrix, rows = _matrix(root, matrix_ref, consumed)
    matches = [r for r in rows if r["job_id"] == admission["matrix_job_id"]]
    if len(matches) != 1:
        raise AdmissionError("matrix_job_id must match exactly one prospective case row")
    row = matches[0]
    if not isinstance(job_doc, dict):
        raise AdmissionError("job document must be an object")
    jp = _resolve(root, str(job_path))
    disk_job = _json(jp)
    if disk_job != job_doc:
        raise AdmissionError("provided job differs from immutable job bytes")
    if set(job_doc) != {"schema_version", "job_id", "driver", "argv", "sources", "pretest_execution_bindings", "phase_receipt_alias"}:
        raise AdmissionError("job must use exact global dispatcher schema v2")
    if job_doc.get("schema_version") != 2 or job_doc.get("job_id") != row["job_id"] or job_doc.get("driver") != "case_v3":
        raise AdmissionError("job identity is not the frozen case_v3 row")
    if job_doc.get("argv") != row["expected_driver_arguments"]:
        raise AdmissionError("case job argv differs from frozen prospective argv")
    flags = ordinary._flag_map(job_doc["argv"])
    required_flags = {"--checkpoint", "--dataset-path", "--selection", "--dataset-id", "--split", "--output",
                      "--managed-job-spec", "--pretest-admission-binding", "--device"}
    if set(flags) != required_flags or flags["--split"] != "test" or flags["--dataset-id"] != row["dataset_id"]:
        raise AdmissionError("case job flags must be exact test-only driver arguments")
    if flags["--output"] != row["output_dir"] or flags["--managed-job-spec"] != _rel(root, jp):
        raise AdmissionError("case output/job-spec flags differ from the admitted path")
    if flags["--pretest-admission-binding"] != _rel(root, admission_path):
        raise AdmissionError("case pretest binding flag differs from this admission")
    if job_doc.get("phase_receipt_alias") != row["phase_receipt_alias_path"]:
        raise AdmissionError("case phase alias differs from matrix row")
    if _resolve(root, row["output_dir"], exists=False).exists() or _resolve(root, row["phase_receipt_alias_path"], exists=False).exists():
        raise AdmissionError("case output and phase alias must be absent before execution")
    if _resolve(root, flags["--selection"]) != _resolve(root, admission["case_selection"]["path"]):
        raise AdmissionError("case selection argv differs from admission reference")
    for refname, rowpath in (("case_selection", "case_selection_path"),
                             ("case_domain", "case_domain_path"),
                             ("c06_descriptor", "c06_descriptor_path")):
        if _resolve(root, admission[refname]["path"]) != _resolve(root, row[rowpath]):
            raise AdmissionError(f"{refname} path differs from its prospective matrix row")
    if _resolve(root, flags["--checkpoint"]) != _resolve(root, admission["checkpoint"]["path"]):
        raise AdmissionError("case checkpoint argv differs from admitted checkpoint")
    if not isinstance(job_doc["pretest_execution_bindings"], list) or job_doc["pretest_execution_bindings"] != [binding_ref]:
        raise AdmissionError("job must link exactly its case admission binding")
    _add(root, consumed, jp)

    # All source refs are explicit, current, and include matrix bindings. No unsupported dispatcher is declared.
    job_sources = _refs(root, job_doc["sources"], "job source")
    matrix_sources = _refs(root, matrix["source_refs"], "matrix source")
    if any(job_sources.get(k) != v for k, v in matrix_sources.items()):
        raise AdmissionError("job sources must retain all frozen matrix source pins")
    if job_sources.get(DRIVER_REL) != _sha(_resolve(root, DRIVER_REL)):
        raise AdmissionError("case driver source missing or stale")
    for rel, sha in job_sources.items():
        p = _resolve(root, rel)
        if _sha(p) != sha:
            raise AdmissionError(f"stale job source {rel}")
        _add(root, consumed, p, sha)

    queue_path, queue_sha = _ref(root, admission["global_test_queue"], "global test queue")
    queue_ref = {"path": _rel(root, queue_path), "sha256": queue_sha}
    if admission["global_test_queue"] != queue_ref:
        raise AdmissionError("global queue ref must be normalized and exact")
    try:
        queued = global_completion.verify_queued_job(root, queue_ref, jp)
    except Exception as exc:
        raise AdmissionError(f"global-v5 queue verification failed: {exc}") from exc
    if not isinstance(queued, dict):
        raise AdmissionError("global-v5 queued-job verifier returned no input snapshot")
    for rel, sha in queued.items():
        _add(root, consumed, _resolve(root, rel), sha)
    _add(root, consumed, queue_path, queue_sha)

    config_path, config_sha = _ref(root, admission["config"], "training config")
    training_binding_path, training_binding_sha = _ref(root, admission["training_binding"], "training binding")
    prep = row["training_preparation"]
    if prep.get("mode") == "static_config":
        config, expected_binding = ordinary._validate_static_preparation(
            root, {"training_preparation": prep, "training_variant": "V4", "seed_id": 0, "method_id": row["method_id"]},
            admission["config"], prep.get("input_blueprint"), prep.get("expected_training_binding_path"), consumed)
    elif prep.get("mode") == "selected_candidate":
        config, expected_binding = ordinary._validate_selected_preparation(
            root, {"training_preparation": prep, "training_variant": "V4", "seed_id": 0, "method_id": row["method_id"]},
            admission["config"], admission["training_binding"], admission["candidate_selection"], consumed)
    else:
        raise AdmissionError("case training preparation mode must be static_config or selected_candidate")
    if config.get("variant") != "V4" or config.get("seed") != 0 or expected_binding != training_binding_path:
        raise AdmissionError("case training input is not exact V4 model-seed-0 provenance")
    if admission.get("policy_config") is not None:
        raise AdmissionError("case_v3 full-policy replay does not accept a policy override")
    if (prep.get("mode") == "static_config") != (admission.get("candidate_selection") is None):
        raise AdmissionError("candidate_selection ref must be present exactly for selected-candidate training")
    checkpoint_path, checkpoint_sha = _ref(root, admission["checkpoint"], "execution checkpoint")
    freeze_path, freeze_sha = _ref(root, admission["freeze"], "test freeze")
    if freeze_path != checkpoint_path.parent / "test-freeze.json":
        raise AdmissionError("case freeze must be checkpoint-sibling test-freeze.json")
    freeze = _json(freeze_path)
    if freeze.get("checkpoint_sha256") != checkpoint_sha:
        raise AdmissionError("case freeze checkpoint hash mismatch")
    ordinary._validate_frozen_policy(root, "case_v3", flags, freeze_path, None)
    if flags["--dataset-path"] != row["expected_driver_arguments"][row["expected_driver_arguments"].index("--dataset-path") + 1]:
        raise AdmissionError("case dataset path argv was altered")
    if flags["--device"] not in {"cuda"}:
        raise AdmissionError("case_v3 currently requires its frozen CUDA path")
    _add(root, consumed, config_path, config_sha)
    _add(root, consumed, training_binding_path, training_binding_sha)
    _add(root, consumed, checkpoint_path, checkpoint_sha)
    _add(root, consumed, freeze_path, freeze_sha)
    provenance = ordinary._validate_training_provenance(
        root, config_path=config_path, config=config, training_binding_path=training_binding_path,
        phase_ref=admission["training_phase_receipt"], summary_ref=admission["training_summary"],
        checkpoint_ref=admission["checkpoint"],
        root_acceptance_ref=admission["root_numeric_acceptance"], consumed=consumed)
    if provenance["checkpoint_sha256"] != admission["checkpoint"].get("sha256"):
        raise AdmissionError("training provenance checkpoint hash mismatch")
    # Recreate the case driver's byte-only source manifest, without importing
    # model weights or opening dataset members.
    try:
        import sys
        scripts = str(root / "edsgn/src")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from edsgn import engine
        source_map = {f"edsgn/src/edsgn/{name}": digest
                      for name, digest in engine.source_manifest().items()}
    except Exception as exc:
        raise AdmissionError(f"cannot verify case runtime source manifest: {exc}") from exc
    # Match frozen_case_signal_sensitivity_v3._source_hashes exactly. The
    # admission-specific global-v5/C06 helpers are additional queue-bound
    # inputs, but they are not part of the validation-selection source digest.
    for rel in (DRIVER_ADMISSION_REL, DRIVER_GLOBAL_REL, DRIVER_REL, KERNEL_REL, H2_KERNEL_REL, KERNEL_ACCEPTANCE_REL,
                "edsgn/staging/simple_baseline_frozen_evaluation_v2.py",
                "edsgn/scripts/aggregate_budget.py", "edsgn/scripts/owned_python_job_inventory_v1.py",
                "reviews/count-loss-training-controller-v5-root-engineering-v1.json",
                "edsgn/scripts/frozen_synthetic_stress.py", "edsgn/src/edsgn/prepare.py"):
        source_map[rel] = _sha(_resolve(root, rel))
    source_sha = hashlib.sha256(json.dumps(source_map, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False).encode("utf-8")).hexdigest()
    selection = _validate_selection(root, row["dataset_id"], admission["case_selection"],
                                    admission["selection_root_acceptance"], config,
                                    checkpoint_sha, source_sha, training_binding_path, freeze, consumed)
    domain, descriptor, collection = _validate_case_domain(
        root, row["dataset_id"], admission["case_domain"], admission["c06_descriptor"], consumed)
    if _resolve(root, flags["--dataset-path"]) != collection:
        raise AdmissionError("case driver collection path differs from C06 descriptor")
    # Selection’s source identity is metadata-bound to training provenance; test arrays stay unopened.
    if selection.get("dataset_fingerprint") != engine.dataset_fingerprint(config["dataset_path"]):
        raise AdmissionError("selection dataset fingerprint differs from byte-bound training collection")

    input_refs = _refs(root, admission["input_refs"], "case admission input")
    # The admission binding, queued job and queue itself are post-freeze refs;
    # including any of them inside input_refs would form a hash cycle.
    excluded_self_refs = {_rel(root, admission_path), _rel(root, queue_path), _rel(root, jp)}
    required = {k: v for k, v in consumed.items() if k not in excluded_self_refs}
    for refname in ("config", "training_binding", "training_phase_receipt", "training_summary", "checkpoint", "freeze",
                    "root_numeric_acceptance", "case_selection", "case_domain", "c06_descriptor", "selection_root_acceptance"):
        p, h = _ref(root, admission[refname], refname)
        required[_rel(root, p)] = h
    if admission.get("policy_config") is not None:
        p, h = _ref(root, admission["policy_config"], "policy config")
        required[_rel(root, p)] = h
    for ref in job_doc["sources"]:
        p, h = _ref(root, ref, "job source")
        required[_rel(root, p)] = h
    if any(input_refs.get(k) != v for k, v in required.items()):
        raise AdmissionError("input_refs omit or stale-bind an admitted input/source")
    qpath, qsha = queue_path, queue_sha
    q, rows_by_id = global_completion._load_queue(root, qpath)
    qrow = rows_by_id.get(row["job_id"])
    if not isinstance(qrow, dict):
        raise AdmissionError("global queue row missing case job")
    expected_origins = sum(tr.get("origin_count", 0) for tr in domain["trajectories"])
    if type(qrow.get("expected_origin_count")) is not int or qrow["expected_origin_count"] != expected_origins:
        raise AdmissionError("global queue origin count differs from complete C06 domain")
    qrefs = {x["path"]: x["sha256"] for x in qrow["pretest_refs"]}
    for k, v in required.items():
        if k not in {_rel(root, admission_path), _rel(root, qpath), _rel(root, jp)} and qrefs.get(k) != v:
            raise AdmissionError(f"queue pretest_refs omit {k}")
    for rel, sha in sorted({**input_refs, **required}.items()):
        _add(root, consumed, _resolve(root, rel), sha)
    for rel, sha in consumed.items():
        p = _resolve(root, rel)
        if _sha(p) != sha:
            raise AdmissionError(f"admission input changed during validation: {rel}")
    return dict(sorted(consumed.items()))


