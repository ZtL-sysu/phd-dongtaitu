"""Metadata-only verifier for a globally closed, frozen-test job campaign.

This is a staging helper, not scientific acceptance. It never opens NPZ/model
payloads. Queue schema v1 is frozen before any test job: top level has exactly
``schema_version, scope, status, queue_id, jobs``. Each job has exactly
``job_id, job_spec_path, pretest_execution_binding_paths, driver, output_dir,
expected_origin_count, coverage_field, expected_report_scope, expected_phase,
report_path, phase_receipt_alias_path, principal_files, pretest_refs``. The queue records prospective
paths only for the future job and binding files, avoiding a hash cycle through
the pretest binding's global_test_queue reference. ``pretest_refs`` contains
only independent files that exist before queue freeze. Completion schema v1 is root-attested
and has exactly ``schema_version, scope, status, accepted_by, verdict, queue,
jobs``; each completion job has ``job_id, job_spec, dispatcher_receipt,
phase_receipt_original, phase_receipt_alias, driver_report, output_files``.
All paths are RUN-relative and every output directory is closed-world hashed.

Supported metadata report contracts are graph_v8, prediction_v5, h2_v4,
profile_v5, c06_v5, simple_v3, and case_v3. Additional mandatory drivers must be added before a
complete global campaign can be accepted; this helper must not be used to
silently omit them. Graph v8 must be oracle-free. A successful return proves
byte/path/receipt closure only; producer-side semantics and root scientific
review remain necessary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CompletionError(RuntimeError):
    pass


# This registry mirrors the report/phase semantics of the exact allowlisted
# frozen drivers. Queue declarations must agree; they cannot weaken coverage.
DRIVER_CONTRACTS = {
    "graph_v8": {"scope": "frozen_test_replay_oracle_free", "coverage": "full_test_origin_count",
                 "phase": "evaluation", "report": "frozen-graph-state-diagnostic.json",
                 "principal": {"predictions.npz", "counter-rows.json", "frozen-graph-state-diagnostic.json",
                               "trace-rows.jsonl.gz", "metrics_by_horizon.csv"}},
    "prediction_v5": {"scope": "full_protocol", "coverage": "n_windows", "phase": "evaluation",
                      "report": "evaluation_manifest.json", "principal": {"predictions.npz", "evaluation_manifest.json"}},
    "h2_v4": {"scope": "frozen_h2_matched_retention_diagnostic", "coverage": "origin_count",
              "phase": "evaluation", "report": "h2-matched-retention-report.json",
              "principal": {"h2-matched-retention-report.json"}},
    "profile_v5": {"scope": "full_selected_split", "coverage": "n_windows", "phase": "profiling",
                   "report": "runtime_summary.json", "principal": {"runtime_summary.json", "window_timings.csv",
                   "trace.jsonl", "profile-budget-accounting.json", "profile-guard-overhead.json"}},
    "c06_v5": {"scope": "full_protocol", "coverage": "n_origins", "phase": "evaluation",
               "report": "perturbation-manifest.json", "principal": {"predictions.npz", "metrics_by_horizon.csv",
               "counter-rows.json", "perturbation-manifest.json"}},
    "simple_v3": {"scope": "full_protocol", "coverage": "n_windows", "phase": "evaluation",
                  "report": "evaluation_manifest.json", "principal": {"predictions.npz",
                  "metrics_by_horizon.csv", "evaluation_manifest.json"}},
    "case_v3": {"scope": "frozen_case_signal_sensitivity_diagnostic", "coverage": "origin_count",
                 "phase": "evaluation", "report": "case-signal-sensitivity-report.json",
                 "principal": {"predictions-teacher.npz", "predictions-relationship_removal.npz",
                 "predictions-focal_retention_removal.npz", "metrics-teacher-by-horizon-feature.csv",
                 "metrics-relationship_removal-by-horizon-feature.csv",
                 "metrics-focal_retention_removal-by-horizon-feature.csv",
                 "case-signal-semantic-proof.jsonl", "evaluation_manifest.json",
                 "case-signal-sensitivity-report.json", "output-closure.json"}},
}


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise CompletionError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(CompletionError(f"invalid JSON constant {x}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompletionError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompletionError(f"JSON object required: {path}")
    return value


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(root, raw):
    if not isinstance(raw, str) or not raw:
        raise CompletionError("path must be a nonempty RUN-relative string")
    p = Path(raw)
    if p.is_absolute():
        raise CompletionError(f"absolute path forbidden: {raw}")
    try:
        q = (root / p).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CompletionError(f"missing/invalid RUN path: {raw}") from exc
    if not q.is_relative_to(root):
        raise CompletionError(f"path escapes RUN: {raw}")
    cur = root
    for part in p.parts:
        cur = cur / part
        if cur.is_symlink():
            raise CompletionError(f"symlink forbidden: {raw}")
    return q


def _resolve_contained(root, raw):
    """Resolve either RUN-relative or absolute receipt paths, still RUN-contained."""
    if not isinstance(raw, str) or not raw:
        raise CompletionError("path must be a nonempty string")
    p = Path(raw)
    if p.is_absolute():
        try:
            q = p.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CompletionError(f"missing/invalid absolute RUN path: {raw}") from exc
        if not q.is_relative_to(root):
            raise CompletionError(f"absolute path escapes RUN: {raw}")
        cur = root
        for part in q.relative_to(root).parts:
            cur = cur / part
            if cur.is_symlink():
                raise CompletionError(f"symlink forbidden: {raw}")
        return q
    return _resolve(root, raw)


def _prospective_path(root, raw):
    """Containment/symlink check for a declared output that may not exist yet."""
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise CompletionError("prospective path must be a RUN-relative string")
    p = (root / raw).resolve(strict=False)
    if not p.is_relative_to(root):
        raise CompletionError(f"prospective path escapes RUN: {raw}")
    cur = root
    for part in Path(raw).parts:
        cur = cur / part
        if cur.exists() and cur.is_symlink():
            raise CompletionError(f"symlink forbidden in prospective path: {raw}")
    return p


def _ref(root, ref, label, consumed):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise CompletionError(f"malformed {label} ref")
    p = _resolve(root, ref["path"])
    if not p.is_file() or not isinstance(ref["sha256"], str) or _sha(p) != ref["sha256"]:
        raise CompletionError(f"{label} hash mismatch: {ref.get('path')}")
    rel = p.relative_to(root).as_posix()
    prior = consumed.get(rel)
    if prior is not None and prior != ref["sha256"]:
        raise CompletionError(f"conflicting hash for {rel}")
    consumed[rel] = ref["sha256"]
    return p


def _file_map(directory, root):
    if not directory.is_dir():
        raise CompletionError(f"output directory missing: {directory}")
    result = {}
    for p in sorted(directory.rglob("*")):
        if p.is_symlink():
            raise CompletionError(f"symlink in output directory: {p}")
        if p.is_file():
            result[p.relative_to(root).as_posix()] = _sha(p)
    return result


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False).encode("utf-8")).hexdigest()


def _simple_fit_domain_refs(root, qrow, report, consumed, method):
    """Bind simple report provenance to accepted train fit and complete C06 metadata."""
    documents = {}
    for ref in qrow["pretest_refs"]:
        path = _resolve(root, ref["path"])
        if path.suffix.lower() != ".json":
            continue
        try:
            doc = _json(path)
        except CompletionError:
            continue
        documents[ref["path"]] = (ref["sha256"], doc)
    fits = [(p, d) for p, (sha, d) in documents.items()
            if d.get("artifact_role") == "train_only_simple_baseline_fit"]
    devs = [(p, d) for p, (sha, d) in documents.items()
            if d.get("scope") == "simple-baseline-development-fit"]
    auths = [(p, d) for p, (sha, d) in documents.items()
             if d.get("status") == "accepted_pretest_authorization"]
    if len(fits) != 1 or len(devs) != 1 or len(auths) != 1:
        raise CompletionError("simple report pretest refs must identify exactly one fit, development acceptance, and test authorization")
    fit_path, fit = fits[0]; fit_sha = documents[fit_path][0]
    dev_path, dev = devs[0]; dev_sha = documents[dev_path][0]
    auth_path, auth = auths[0]; auth_sha = documents[auth_path][0]
    prov = report.get("provenance")
    if not isinstance(prov, dict) or prov.get("fit_manifest_sha256") != fit_sha or prov.get("authorization_sha256") != auth_sha:
        raise CompletionError("simple report provenance does not bind its fit and test authorization")
    if (fit.get("split") != "train" or fit.get("test_accessed") is not False
            or fit.get("artifact_role") != "train_only_simple_baseline_fit"
            or dev.get("accepted_by_root") is not True or dev.get("verdict") != "pass"
            or dev.get("fit_manifest_path") not in {fit_path, str(_resolve(root, fit_path))}
            or dev.get("fit_manifest_sha256") != fit_sha
            or auth.get("accepted_by_root") is not True or auth.get("verdict") != "pass"
            or auth.get("method") != method or fit.get("dataset_path") != auth.get("dataset_path")):
        raise CompletionError("simple train fit lacks the exact root-accepted development record")
    if method == "linear":
        coefficient_path = fit.get("linear_coefficients_path")
        coefficient_sha = fit.get("linear_coefficients_sha256")
        if isinstance(coefficient_path, str) and Path(coefficient_path).is_absolute():
            try: coefficient_rel = Path(coefficient_path).resolve().relative_to(root).as_posix()
            except ValueError as exc: raise CompletionError("simple coefficient escapes RUN") from exc
        else:
            coefficient_rel = coefficient_path
        coefficient_ref = next((r for r in qrow["pretest_refs"] if r.get("path") == coefficient_rel), None)
        if (not isinstance(coefficient_ref, dict) or coefficient_ref.get("sha256") != coefficient_sha
                or auth.get("coefficient_sha256") != coefficient_sha
                or prov.get("coefficient_sha256") != coefficient_sha):
            raise CompletionError("simple linear report does not bind the train-fit coefficient bytes")
    elif auth.get("parameter_identity") != "persistence-repeat-last-observation/no-parameters":
        raise CompletionError("simple persistence authorization lacks explicit no-parameter identity")
    descriptor_path = auth.get("domain_descriptor_path")
    descriptor_sha = auth.get("domain_descriptor_sha256")
    if isinstance(descriptor_path, str) and Path(descriptor_path).is_absolute():
        try: descriptor_rel = Path(descriptor_path).resolve().relative_to(root).as_posix()
        except ValueError as exc: raise CompletionError("simple C06 descriptor escapes RUN") from exc
    else:
        descriptor_rel = descriptor_path
    descriptor_pair = documents.get(descriptor_rel)
    if descriptor_pair is None or descriptor_pair[0] != descriptor_sha:
        raise CompletionError("simple authorization does not bind a pretest C06 descriptor")
    descriptor = descriptor_pair[1]
    if (descriptor.get("schema") != "edsgn.c06-domain-preflight.v1" or descriptor.get("split") != "test"
            or not isinstance(descriptor.get("ordered_collection"), list) or not descriptor["ordered_collection"]):
        raise CompletionError("simple C06 descriptor is not a complete test-domain declaration")
    domain_rel = descriptor.get("domain_manifest_path")
    domain_sha = descriptor.get("domain_manifest_sha256")
    if isinstance(domain_rel, str) and Path(domain_rel).is_absolute():
        try: domain_rel = Path(domain_rel).resolve().relative_to(root).as_posix()
        except ValueError as exc: raise CompletionError("simple C06 domain escapes RUN") from exc
    domain_pair = documents.get(domain_rel)
    if domain_pair is None or domain_pair[0] != domain_sha:
        raise CompletionError("simple C06 descriptor does not bind its complete domain manifest")
    domain = domain_pair[1]
    if not isinstance(domain.get("trajectories"), list) or not domain["trajectories"]:
        raise CompletionError("simple C06 domain manifest has no trajectories")
    ordered = descriptor["ordered_collection"]
    if len(ordered) != len(domain["trajectories"]):
        raise CompletionError("simple C06 descriptor/domain trajectory membership differs")
    for index, row in enumerate(ordered):
        trajectory = domain["trajectories"][index]
        if (not isinstance(row, dict) or row.get("trajectory_index") != index
                or row.get("split") != "test" or row.get("trajectory_index") != trajectory.get("trajectory_index")
                or trajectory.get("split") != "test"):
            raise CompletionError("simple C06 ordered collection differs from the complete test domain")


def _load_queue(root, queue_path):
    q = _json(queue_path)
    if (set(q) != {"schema_version", "scope", "status", "queue_id", "jobs"}
            or q["schema_version"] != 1 or q["scope"] != "prospective_complete_test_job_set"
            or q["status"] != "frozen_before_test" or not isinstance(q["queue_id"], str)
            or not q["queue_id"] or not isinstance(q["jobs"], list) or not q["jobs"]):
        raise CompletionError("invalid frozen global queue schema/status")
    queue_rows = {}
    global_job_specs = set()
    global_output_dirs = {}
    global_reports = set()
    global_principals = set()
    global_aliases = set()
    expected_fields = {"job_id", "job_spec_path", "pretest_execution_binding_paths", "driver",
                       "output_dir", "expected_origin_count", "coverage_field", "expected_report_scope",
                       "expected_phase", "report_path", "phase_receipt_alias_path", "principal_files", "pretest_refs"}
    for row in q["jobs"]:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise CompletionError("malformed queue job row")
        jid = row["job_id"]
        if not isinstance(jid, str) or not jid or jid in queue_rows:
            raise CompletionError("duplicate/invalid queue job id")
        if row["driver"] not in DRIVER_CONTRACTS:
            raise CompletionError(f"unsupported mandatory driver {row['driver']}")
        contract = DRIVER_CONTRACTS[row["driver"]]
        if (row["coverage_field"] != contract["coverage"]
                or row["expected_report_scope"] != contract["scope"]
                or row["expected_phase"] != contract["phase"]):
            raise CompletionError(f"queue weakens actual {row['driver']} report contract")
        if type(row["expected_origin_count"]) is not int or row["expected_origin_count"] <= 0:
            raise CompletionError("expected_origin_count must be positive integer")
        for key in ("job_spec_path", "output_dir", "coverage_field", "expected_report_scope", "expected_phase", "report_path", "phase_receipt_alias_path"):
            if not isinstance(row[key], str) or not row[key]:
                raise CompletionError("queue path/report contract malformed")
        spec_path = _prospective_path(root, row["job_spec_path"])
        output_path = _prospective_path(root, row["output_dir"])
        report_path = _prospective_path(root, row["report_path"])
        alias_path = _prospective_path(root, row["phase_receipt_alias_path"])
        if spec_path in global_job_specs:
            raise CompletionError("job_spec_path must be unique across queue jobs")
        global_job_specs.add(spec_path)
        if report_path in global_reports:
            raise CompletionError("report_path must be unique across queue jobs")
        global_reports.add(report_path)
        if alias_path in global_aliases:
            raise CompletionError("phase receipt alias must be unique across queue jobs")
        global_aliases.add(alias_path)
        global_output_dirs[jid] = output_path
        if alias_path.parent.name != "evidence" or not alias_path.is_relative_to(root / "runs"):
            raise CompletionError("phase receipt alias must be under RUN runs/.../evidence")
        if not isinstance(row["pretest_execution_binding_paths"], list) or not row["pretest_execution_binding_paths"]:
            raise CompletionError("queue must freeze ordered pretest binding paths")
        if (not all(isinstance(x, str) and x for x in row["pretest_execution_binding_paths"])
                or len(set(row["pretest_execution_binding_paths"])) != len(row["pretest_execution_binding_paths"])):
            raise CompletionError("invalid/duplicate pretest binding path")
        if (not isinstance(row["principal_files"], list) or not row["principal_files"]
                or not all(isinstance(x, str) and x for x in row["principal_files"])):
            raise CompletionError("principal_files must be a nonempty path list")
        if not contract["principal"].issubset({Path(x).name for x in row["principal_files"]}):
            raise CompletionError(f"queue omits principal outputs for {row['driver']}")
        if len(set(row["principal_files"])) != len(row["principal_files"]):
            raise CompletionError("duplicate principal output path")
        for principal in row["principal_files"]:
            principal_path = _prospective_path(root, principal)
            if principal_path == output_path or not principal_path.is_relative_to(output_path):
                raise CompletionError(f"principal output must be inside its own output directory: {jid}/{principal}")
            if principal_path in global_principals:
                raise CompletionError("principal output path must be unique across queue jobs")
            global_principals.add(principal_path)
        if Path(row["report_path"]).name != contract["report"]:
            raise CompletionError(f"queue report filename differs from actual {row['driver']} output")
        if _prospective_path(root, row["report_path"]).parent != _prospective_path(root, row["output_dir"]):
            raise CompletionError(f"report is not in declared output directory for {jid}")
        if not isinstance(row["pretest_refs"], list) or not row["pretest_refs"]:
            raise CompletionError("each test job must bind independent pretest inputs")
        refkeys = set()
        for ref in row["pretest_refs"]:
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or not isinstance(ref["path"], str):
                raise CompletionError("malformed independent pretest ref")
            _resolve(root, ref["path"])
            if ref["path"] in refkeys:
                raise CompletionError("duplicate independent pretest ref")
            refkeys.add(ref["path"])
        queue_rows[jid] = row
    # Output trees are independent closed-world archives. Reject equality and
    # ancestor/descendant overlap across jobs before any job is dispatched.
    output_items = list(global_output_dirs.items())
    for index, (left_id, left) in enumerate(output_items):
        for right_id, right in output_items[index + 1:]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise CompletionError(f"job output directories overlap: {left_id} and {right_id}")
    for alias in global_aliases:
        if any(alias.is_relative_to(output) for output in global_output_dirs.values()):
            raise CompletionError("phase receipt aliases must remain outside every job output directory")
    return q, queue_rows


def _read_job_and_bindings(root, queue_ref, qrow, job_path, consumed):
    job_path = _resolve(root, job_path)
    if job_path.relative_to(root).as_posix() != qrow["job_spec_path"]:
        raise CompletionError("job spec path differs from frozen queue")
    job_sha = _sha(job_path)
    consumed[job_path.relative_to(root).as_posix()] = job_sha
    job = _json(job_path)
    if (set(job) != {"schema_version", "job_id", "driver", "argv", "sources", "pretest_execution_bindings", "phase_receipt_alias"}
            or job["schema_version"] != 2 or job["job_id"] != qrow["job_id"] or job["driver"] != qrow["driver"]):
        raise CompletionError("dispatcher job spec identity/schema mismatch")
    argv = job["argv"]
    if not isinstance(argv, list) or len(argv) % 2 or any(not isinstance(x, str) for x in argv):
        raise CompletionError("malformed dispatcher argv")
    flags = dict(zip(argv[0::2], argv[1::2]))
    if (len(flags) != len(argv) // 2 or "--output" not in flags
            or _prospective_path(root, flags["--output"]) != _prospective_path(root, qrow["output_dir"])):
        raise CompletionError("dispatcher output path/duplicate flags differ from queue")
    if flags.get("--split", "test") != "test":
        raise CompletionError("test split not bound in dispatcher job")
    if not isinstance(job.get("phase_receipt_alias"), str):
        raise CompletionError("dispatcher phase receipt alias missing")
    if _prospective_path(root, job["phase_receipt_alias"]) != _prospective_path(root, qrow["phase_receipt_alias_path"]):
        raise CompletionError("dispatcher phase alias path differs from frozen queue")
    bindings = job["pretest_execution_bindings"]
    if not isinstance(bindings, list) or len(bindings) != len(qrow["pretest_execution_binding_paths"]):
        raise CompletionError("dispatcher pretest binding list differs from frozen queue")
    binding_paths = []
    discovered_pretest = {}

    def add_discovered(path, digest):
        if not isinstance(path, str) or not isinstance(digest, str):
            raise CompletionError("malformed pretest input hash reference")
        prior = discovered_pretest.get(path)
        if prior is not None and prior != digest:
            raise CompletionError(f"conflicting pretest input hashes for {path}")
        discovered_pretest[path] = digest

    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise CompletionError("malformed dispatcher pretest binding ref")
        binding_path = _ref(root, binding, "pretest execution binding", consumed)
        rel = binding_path.relative_to(root).as_posix()
        binding_paths.append(rel)
        binding_doc = _json(binding_path)
        if binding_doc.get("global_test_queue") != queue_ref:
            raise CompletionError("pretest binding does not bind exact frozen global queue")
        for field, path_field, sha_field in (("plan", "plan_path", "plan_sha256"),
                                             ("registry", "registry_path", "registry_sha256")):
            if path_field in binding_doc or sha_field in binding_doc:
                if not isinstance(binding_doc.get(path_field), str) or not isinstance(binding_doc.get(sha_field), str):
                    raise CompletionError(f"malformed pretest binding {field} path/hash")
                add_discovered(binding_doc[path_field], binding_doc[sha_field])

        def visit(value):
            if isinstance(value, dict):
                if set(value) == {"path", "sha256"} and isinstance(value["path"], str) and isinstance(value["sha256"], str):
                    add_discovered(value["path"], value["sha256"])
                    return
                for child_value in value.values():
                    visit(child_value)
            elif isinstance(value, list):
                for child_value in value:
                    visit(child_value)
        visit(binding_doc)
    if binding_paths != qrow["pretest_execution_binding_paths"]:
        raise CompletionError("dispatcher pretest binding path/order differs from frozen queue")
    source_refs = job["sources"]
    if not isinstance(source_refs, list):
        raise CompletionError("dispatcher sources must be a list")
    independent = {x["path"]: x["sha256"] for x in qrow["pretest_refs"]}
    for ref in source_refs:
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or independent.get(ref["path"]) != ref["sha256"]:
            raise CompletionError("queue omits or mismatches a declared job source")
        add_discovered(ref["path"], ref["sha256"])
    for ref in qrow["pretest_refs"]:
        _ref(root, ref, "independent pretest input", consumed)
    queued_independent = {ref["path"]: ref["sha256"] for ref in qrow["pretest_refs"]}
    discovered_independent = {path: digest for path, digest in discovered_pretest.items()
                              if path != queue_ref["path"]}
    if queued_independent != discovered_independent:
        raise CompletionError("queue pretest_refs do not exactly cover binding/job independent hash references")
    return job, job_sha


def _same_ref_path(ref, expected_path, label):
    if ref.get("path") != expected_path:
        raise CompletionError(f"{label} path differs from frozen queue")


def verify_queued_job(root, queue_ref, job_path):
    """Pre-dispatch metadata gate for one exact frozen queue member.

    Returns hashes to merge into the dispatcher's immutable BEFORE snapshot.
    It does not execute the job, deserialize arrays/checkpoints, or inspect
    numeric artifacts.
    """
    root = Path(root).resolve(strict=True)
    consumed = {}
    qpath = _ref(root, queue_ref, "pretest global queue", consumed)
    _, rows = _load_queue(root, qpath)
    candidate = _prospective_path(root, job_path)
    rel = candidate.relative_to(root).as_posix()
    matches = [row for row in rows.values() if row["job_spec_path"] == rel]
    if len(matches) != 1:
        raise CompletionError("job path is not a unique member of the frozen global queue")
    if not candidate.is_file():
        raise CompletionError("registered dispatcher job file is missing")
    _read_job_and_bindings(root, queue_ref, matches[0], job_path, consumed)
    # Recheck the queue and consumed inputs at return, before dispatch begins.
    for path, expected in list(consumed.items()):
        p = _resolve(root, path)
        if not p.is_file() or _sha(p) != expected:
            raise CompletionError(f"pre-dispatch binding changed during validation: {path}")
    return dict(sorted(consumed.items()))


def verify_completion(root, completion_ref, expected_queue_ref):
    """Verify exact frozen-job closure; return all consumed RUN-relative hashes.

    expected_queue_ref is supplied by the accepted post-P4/pretest binding and
    must be {path, sha256}. No actual test arrays or model/checkpoint payloads
    are loaded or deserialized.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise CompletionError("RUN root must be a directory")
    consumed = {}
    completion_path = _ref(root, completion_ref, "global completion", consumed)
    queue_path = _ref(root, expected_queue_ref, "pretest global queue", consumed)
    queue_rel = queue_path.relative_to(root).as_posix()
    if not completion_path.is_file() or not queue_path.is_file():
        raise CompletionError("completion and queue must be files")

    q, queue_rows = _load_queue(root, queue_path)
    for qrow in queue_rows.values():
        _read_job_and_bindings(root, expected_queue_ref, qrow, qrow["job_spec_path"], consumed)

    c = _json(completion_path)
    if (set(c) != {"schema_version", "scope", "status", "accepted_by", "verdict", "queue", "jobs"}
            or c["schema_version"] != 1 or c["scope"] != "global_frozen_test_completion"
            or c["status"] != "accepted_by_root" or c["accepted_by"] != "root" or c["verdict"] != "pass"):
        raise CompletionError("completion lacks root-attested closure schema")
    _ref(root, c["queue"], "completion queue", consumed)
    if c["queue"]["path"] != queue_rel or c["queue"]["sha256"] != expected_queue_ref["sha256"]:
        raise CompletionError("completion references a different frozen queue")
    if not isinstance(c["jobs"], list):
        raise CompletionError("completion jobs must be a list")
    completion_rows = {}
    required_job_fields = {"job_id", "job_spec", "dispatcher_receipt", "phase_receipt_original",
                           "phase_receipt_alias", "driver_report", "output_files"}
    for row in c["jobs"]:
        if not isinstance(row, dict) or set(row) != required_job_fields:
            raise CompletionError("malformed completion job row")
        jid = row["job_id"]
        if not isinstance(jid, str) or jid in completion_rows:
            raise CompletionError("duplicate/invalid completion job id")
        completion_rows[jid] = row
    if set(completion_rows) != set(queue_rows):
        raise CompletionError("completion job membership differs from frozen queue")

    # Reverify all job products and semantics. No payload decoding occurs.
    for jid, qrow in queue_rows.items():
        row = completion_rows[jid]
        spec_path = _ref(root, row["job_spec"], f"completed job spec {jid}", consumed)
        if row["job_spec"]["path"] != qrow["job_spec_path"]:
            raise CompletionError(f"job spec ref differs for {jid}")
        job = _json(spec_path)
        dispatch_path = _ref(root, row["dispatcher_receipt"], f"dispatch receipt {jid}", consumed)
        if dispatch_path.relative_to(root).as_posix() != f"logs/verified-frozen-dispatch/{jid}.json":
            raise CompletionError(f"dispatcher receipt path is not canonical for {jid}")
        original_path = _ref(root, row["phase_receipt_original"], f"phase receipt original {jid}", consumed)
        alias_path = _ref(root, row["phase_receipt_alias"], f"phase receipt alias {jid}", consumed)
        report_path = _ref(root, row["driver_report"], f"driver report {jid}", consumed)
        if original_path.read_bytes() != alias_path.read_bytes():
            raise CompletionError(f"phase receipt alias bytes differ for {jid}")
        phase = _json(original_path)
        alias = _json(alias_path)
        output_dir = _resolve(root, qrow["output_dir"])
        if (original_path.parent.relative_to(root).as_posix() != "logs/offline-phase-wall"
                or original_path.stem != phase.get("job_id")):
            raise CompletionError(f"phase receipt path does not own its child job id for {jid}")
        phase_output = _resolve_contained(root, phase.get("output_dir", ""))
        alias_output = _resolve_contained(root, alias.get("output_dir", ""))
        if (phase.get("status") != "completed" or alias.get("status") != "completed"
                or not isinstance(phase.get("job_id"), str) or not phase.get("job_id")
                or alias.get("job_id") != phase.get("job_id")
                or phase_output != output_dir or alias_output != output_dir
                or phase.get("phase") != qrow["expected_phase"]):
            raise CompletionError(f"phase receipt status/identity/output mismatch for {jid}")
        dispatch = _json(dispatch_path)
        if (dispatch.get("status") != "accepted-output-pending-scientific-review"
                or dispatch.get("job_id") != jid or dispatch.get("driver") != qrow["driver"]
                or dispatch.get("child_returncode") != 0):
            raise CompletionError(f"dispatch child did not complete successfully for {jid}")
        for label in ("validator_before", "validator_after"):
            if not isinstance(dispatch.get(label), dict) or dispatch[label].get("returncode") != 0:
                raise CompletionError(f"{label} failed for {jid}")
        if dispatch.get("before") != dispatch.get("after"):
            raise CompletionError(f"frozen before/after snapshots differ for {jid}")
        child = dispatch.get("child_phase_receipt")
        if not isinstance(child, dict):
            raise CompletionError(f"dispatcher child phase receipt missing for {jid}")
        if (child.get("path") != row["phase_receipt_original"]["path"]
                or child.get("sha256") != row["phase_receipt_original"]["sha256"]
                or child.get("job_id") != phase.get("job_id")
                or child.get("phase") != qrow["expected_phase"] or child.get("status") != "completed"
                or _resolve_contained(root, child.get("output_dir", "")) != output_dir
                or child.get("receipt_doc") != phase
                or child.get("phase_receipt_original_path") != row["phase_receipt_original"]["path"]
                or child.get("phase_receipt_original_sha256") != row["phase_receipt_original"]["sha256"]
                or child.get("phase_receipt_alias_path") != row["phase_receipt_alias"]["path"]
                or child.get("phase_receipt_alias_sha256") != row["phase_receipt_alias"]["sha256"]):
            raise CompletionError(f"dispatcher nested child receipt ownership mismatch for {jid}")
        if (dispatch.get("phase_receipt_original_path") != row["phase_receipt_original"]["path"]
                or dispatch.get("phase_receipt_original_sha256") != row["phase_receipt_original"]["sha256"]
                or dispatch.get("phase_receipt_alias_path") != row["phase_receipt_alias"]["path"]
                or dispatch.get("phase_receipt_alias_sha256") != row["phase_receipt_alias"]["sha256"]):
            raise CompletionError(f"dispatcher receipt alias bindings mismatch for {jid}")
        bfiles = dispatch.get("before", {}).get("files")
        if not isinstance(bfiles, dict):
            raise CompletionError(f"dispatch snapshot absent for {jid}")
        if bfiles.get(spec_path.relative_to(root).as_posix()) != _sha(spec_path):
            raise CompletionError(f"dispatch BEFORE snapshot omits/mismatches jobspec for {jid}")
        for pref in qrow["pretest_refs"]:
            if bfiles.get(pref["path"]) != pref["sha256"]:
                raise CompletionError(f"dispatch snapshot omitted/mismatched pretest ref for {jid}: {pref['path']}")
        for binding in job["pretest_execution_bindings"]:
            if bfiles.get(binding["path"]) != binding["sha256"]:
                raise CompletionError(f"dispatch snapshot omitted/mismatched binding for {jid}: {binding['path']}")
        for source in job["sources"]:
            if bfiles.get(source["path"]) != source["sha256"]:
                raise CompletionError(f"dispatch snapshot omitted/mismatched source for {jid}: {source['path']}")
        if bfiles.get(queue_rel) != expected_queue_ref["sha256"]:
            raise CompletionError(f"dispatch BEFORE snapshot does not bind the global queue for {jid}")
        if (dispatch.get("phase_receipt_original_path") != row["phase_receipt_original"]["path"]
                or dispatch.get("phase_receipt_original_sha256") != row["phase_receipt_original"]["sha256"]
                or dispatch.get("phase_receipt_alias_path") != row["phase_receipt_alias"]["path"]
                or dispatch.get("phase_receipt_alias_sha256") != row["phase_receipt_alias"]["sha256"]):
            raise CompletionError(f"dispatch phase receipt refs mismatch for {jid}")
        report = _json(report_path)
        contract = DRIVER_CONTRACTS[qrow["driver"]]
        if (report.get("scope") != contract["scope"] or report.get("split") != "test"
                or report.get(contract["coverage"]) != qrow["expected_origin_count"]):
            raise CompletionError(f"report scope/split/origin coverage mismatch for {jid}")
        if qrow["driver"] == "graph_v8" and (report.get("oracle_accessed") is not False
                                                   or report.get("oracle_npz_decoded") is not False
                                                   or report.get("oracle_analysis_status") != "deferred_until_global_completion"):
            raise CompletionError(f"graph v8 report is not explicitly oracle-free for {jid}")
        if qrow["driver"] == "h2_v4":
            outputs = report.get("prediction_outputs")
            required_arms = {"informative", "shuffled", "constant", "reset"}
            if not isinstance(outputs, dict) or set(outputs) != required_arms:
                raise CompletionError(f"H2 report lacks all four frozen arm output records for {jid}")
            for arm, item in outputs.items():
                if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                    raise CompletionError(f"H2 output record malformed for {jid}/{arm}")
                arm_path = _resolve(root, (Path(qrow["output_dir"]) / item["path"]).as_posix())
                if arm_path.relative_to(root).as_posix() not in qrow["principal_files"]:
                    raise CompletionError(f"H2 arm output not predeclared as a principal file for {jid}/{arm}")
                if _sha(arm_path) != item["sha256"]:
                    raise CompletionError(f"H2 report arm output hash mismatch for {jid}/{arm}")
        if qrow["driver"] == "simple_v3":
            if (report.get("method") not in {"persistence", "linear"} or report.get("seed_id") != 0
                    or report.get("oracle_accessed") is not False or report.get("test_feedback_used") is not False):
                raise CompletionError(f"simple report violates frozen method/seed/oracle contract for {jid}")
            _simple_fit_domain_refs(root, qrow, report, consumed, report["method"])
            expected_outputs = {"predictions.npz", "metrics_by_horizon.csv"}
            output_hashes = report.get("output_hashes")
            if (not isinstance(output_hashes, dict) or set(output_hashes) != expected_outputs
                    or any(not isinstance(h, str) or len(h) != 64 for h in output_hashes.values())):
                raise CompletionError(f"simple report must bind exact prediction and metric output hashes for {jid}")
            for name, digest in output_hashes.items():
                artifact = _resolve(root, (Path(qrow["output_dir"]) / name).as_posix())
                if _sha(artifact) != digest:
                    raise CompletionError(f"simple report output hash mismatch for {jid}: {name}")
        if qrow["driver"] == "case_v3":
            selected = report.get("selection")
            if (report.get("variant") != "V4" or report.get("model_seed_id") != 0
                    or report.get("same_window_policy") != "full" or report.get("oracle_accessed") is not False
                    or report.get("targets_passed_to_kernel") is not False or not isinstance(selected, dict)):
                raise CompletionError(f"case report violates frozen policy/seed/oracle contract for {jid}")
            focal, neighbor = selected.get("focal_node_index"), selected.get("neighbor_node_index")
            if type(focal) is not int or type(neighbor) is not int or focal < 0 or neighbor < 0 or focal == neighbor:
                raise CompletionError(f"case report lacks fixed directed nonself node selection for {jid}")
            if report.get("informative_teacher_parity_all_windows") is not True or report.get("retention_graph_actions_exact_all_windows") is not True:
                raise CompletionError(f"case report omits required teacher/retention parity for {jid}")
            expected_records = {"teacher", "relationship_removal", "focal_retention_removal",
                                "metrics:teacher", "metrics:relationship_removal", "metrics:focal_retention_removal"}
            records = report.get("prediction_outputs")
            if not isinstance(records, dict) or set(records) != expected_records:
                raise CompletionError(f"case report must bind exactly three archives and three metric tables for {jid}")
            expected_names = {"teacher":"predictions-teacher.npz",
                "relationship_removal":"predictions-relationship_removal.npz",
                "focal_retention_removal":"predictions-focal_retention_removal.npz",
                "metrics:teacher":"metrics-teacher-by-horizon-feature.csv",
                "metrics:relationship_removal":"metrics-relationship_removal-by-horizon-feature.csv",
                "metrics:focal_retention_removal":"metrics-focal_retention_removal-by-horizon-feature.csv"}
            for key, item in records.items():
                if not isinstance(item, dict) or item.get("path") != expected_names[key] or not isinstance(item.get("sha256"), str):
                    raise CompletionError(f"case output record path/schema mismatch for {jid}/{key}")
                artifact = _resolve(root, (Path(qrow["output_dir"]) / item["path"]).as_posix())
                if _sha(artifact) != item["sha256"]:
                    raise CompletionError(f"case output record hash mismatch for {jid}/{key}")
            proof_name = "case-signal-semantic-proof.jsonl"
            if (report.get("semantic_proof_path") != proof_name
                    or not isinstance(report.get("semantic_proof_sha256"), str)
                    or report.get("semantic_proof_expected_lines") != qrow["expected_origin_count"] * report.get("history", 0)
                    or report.get("semantic_proof_lines") != report.get("semantic_proof_expected_lines")):
                raise CompletionError(f"case semantic proof count/path is inconsistent for {jid}")
            proof_path = _resolve(root, (Path(qrow["output_dir"]) / proof_name).as_posix())
            if _sha(proof_path) != report["semantic_proof_sha256"]:
                raise CompletionError(f"case semantic proof hash mismatch for {jid}")
            with proof_path.open("rb") as stream:
                line_count = sum(1 for line in stream if line.strip())
            if line_count != report["semantic_proof_expected_lines"]:
                raise CompletionError(f"case semantic proof physical line count mismatch for {jid}")
            manifest_ref = report.get("evaluation_manifest")
            if not isinstance(manifest_ref, dict) or set(manifest_ref) != {"path", "sha256"} or manifest_ref["path"] != "evaluation_manifest.json":
                raise CompletionError(f"case report does not bind its evaluation manifest for {jid}")
            manifest_path = _resolve(root, (Path(qrow["output_dir"]) / manifest_ref["path"]).as_posix())
            if _sha(manifest_path) != manifest_ref["sha256"]:
                raise CompletionError(f"case evaluation manifest hash mismatch for {jid}")
            case_manifest = _json(manifest_path)
            if (case_manifest.get("scope") != "case_signal_evaluation_manifest" or case_manifest.get("split") != "test"
                    or case_manifest.get("origin_count") != qrow["expected_origin_count"]
                    or case_manifest.get("expected_semantic_proof_lines") != report["semantic_proof_expected_lines"]):
                raise CompletionError(f"case evaluation manifest scope/coverage mismatch for {jid}")
            manifest_files = case_manifest.get("files")
            expected_manifest_files = {item["path"]: item["sha256"] for item in records.values()}
            expected_manifest_files[proof_name] = report["semantic_proof_sha256"]
            if manifest_files != expected_manifest_files:
                raise CompletionError(f"case evaluation manifest file map mismatch for {jid}")
            closure_path = _resolve(root, (Path(qrow["output_dir"]) / "output-closure.json").as_posix())
            closure = _json(closure_path)
            if (closure.get("scope") != "case_signal_output_directory_closure" or closure.get("self_excluded") is not True
                    or not isinstance(closure.get("files"), dict)
                    or closure.get("closure_sha256") != _canonical_sha(closure["files"])):
                raise CompletionError(f"case output closure record is malformed for {jid}")
        nested_principal = child.get("principal_artifacts")
        if not isinstance(nested_principal, dict) or not nested_principal:
            raise CompletionError(f"dispatcher nested receipt omits actual principal-artifact hashes for {jid}")
        if row["driver_report"]["path"] != qrow["report_path"]:
            raise CompletionError(f"report path differs from pretest plan for {jid}")
        if not isinstance(row["output_files"], list):
            raise CompletionError(f"output_files must be a list for {jid}")
        declared = {}
        for ref in row["output_files"]:
            p = _ref(root, ref, f"output file {jid}", consumed)
            rel = p.relative_to(root).as_posix()
            if rel in declared:
                raise CompletionError(f"duplicate output file in completion for {jid}")
            declared[rel] = ref["sha256"]
        actual = _file_map(output_dir, root)
        if qrow["driver"] == "simple_v3" and {Path(path).name for path in actual} != {
                "predictions.npz", "metrics_by_horizon.csv", "evaluation_manifest.json"}:
            raise CompletionError(f"simple output directory contains missing or extra files for {jid}")
        if qrow["driver"] == "case_v3" and {Path(path).name for path in actual} != DRIVER_CONTRACTS["case_v3"]["principal"]:
            raise CompletionError(f"case output directory contains missing or extra files for {jid}")
        if actual != declared:
            raise CompletionError(f"output directory membership/hash mismatch for {jid}")
        for name, digest in nested_principal.items():
            rel = (output_dir / name).relative_to(root).as_posix()
            if actual.get(rel) != digest:
                raise CompletionError(f"dispatcher principal-artifact hash differs from closed output for {jid}: {name}")
        for principal in qrow["principal_files"]:
            if principal not in actual:
                raise CompletionError(f"missing principal artifact for {jid}: {principal}")
        if row["driver_report"]["path"] not in actual:
            raise CompletionError(f"driver report not inside closed output tree for {jid}")
        phase_outputs = phase.get("outputs")
        if qrow["driver"] == "profile_v5":
            phase_outputs = phase.get("output_sha256")
        elif qrow["driver"] == "c06_v5":
            phase_outputs = phase.get("outputs")
        if phase_outputs is not None:
            if qrow["driver"] == "prediction_v5":
                normalized = {}
                if not isinstance(phase_outputs, list):
                    raise CompletionError(f"prediction phase outputs malformed for {jid}")
                for item in phase_outputs:
                    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                        raise CompletionError(f"prediction phase output ref malformed for {jid}")
                    p = _resolve_contained(root, item["path"])
                    normalized[p.relative_to(root).as_posix()] = item["sha256"]
            elif qrow["driver"] in {"profile_v5", "c06_v5", "simple_v3", "case_v3"}:
                if not isinstance(phase_outputs, dict):
                    raise CompletionError(f"phase output hash map malformed for {jid}")
                normalized = {}
                for name, digest in phase_outputs.items():
                    p = _resolve(root, (Path(qrow["output_dir"]) / name).as_posix())
                    normalized[p.relative_to(root).as_posix()] = digest
            else:
                raise CompletionError(f"unexpected phase outputs schema for {jid}")
            if normalized != actual:
                raise CompletionError(f"phase's full output hashes differ from closed output directory for {jid}")
        elif qrow["driver"] in {"prediction_v5", "profile_v5", "c06_v5"}:
            raise CompletionError(f"driver phase receipt omits its full output hash map for {jid}")

        if qrow["driver"] == "case_v3":
            closure_path = output_dir / "output-closure.json"
            closure = _json(closure_path)
            expected_closure_files = {Path(path).name: digest for path, digest in actual.items()
                                      if path != closure_path.relative_to(root).as_posix()}
            if closure["files"] != expected_closure_files:
                raise CompletionError(f"case closure file membership/hash map mismatch for {jid}")

    # Final TOCTOU barrier: all consumed refs and exact output memberships must
    # still match the hashes observed above.
    for rel, expected in list(consumed.items()):
        p = _resolve(root, rel)
        if not p.is_file() or _sha(p) != expected:
            raise CompletionError(f"input/output changed during global verification: {rel}")
    for qrow in queue_rows.values():
        expected = {rel: digest for rel, digest in consumed.items()
                    if Path(rel).is_relative_to(Path(qrow["output_dir"]))}
        if _file_map(_resolve(root, qrow["output_dir"]), root) != expected:
            raise CompletionError(f"output directory changed during final barrier: {qrow['job_id']}")
    return dict(sorted(consumed.items()))
