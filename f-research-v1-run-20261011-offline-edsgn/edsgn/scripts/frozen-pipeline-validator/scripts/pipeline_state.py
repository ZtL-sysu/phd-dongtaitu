#!/usr/bin/env python3
"""Validate ordered workflow evidence; run inside the project's local Conda environment."""
import argparse
import hashlib
import json
from pathlib import Path

import importlib.util

_citation_spec = importlib.util.spec_from_file_location(
    "_policy_citation_requirements", Path(__file__).resolve().parent / "citation_requirements.py"
)
if _citation_spec is None or _citation_spec.loader is None:
    raise ImportError("cannot load sibling citation_requirements.py")
_citation_module = importlib.util.module_from_spec(_citation_spec)
_citation_spec.loader.exec_module(_citation_module)
check_citations = _citation_module.check

import importlib.util

_readiness_spec = importlib.util.spec_from_file_location(
    "_policy_delivery_readiness", Path(__file__).resolve().parent / "delivery_readiness.py"
)
if _readiness_spec is None or _readiness_spec.loader is None:
    raise ImportError("cannot load sibling delivery_readiness.py")
_readiness_module = importlib.util.module_from_spec(_readiness_spec)
_readiness_spec.loader.exec_module(_readiness_module)
validate_readiness = _readiness_module.validate_readiness


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def definition():
    return json.loads((Path(__file__).resolve().parents[1] / "assets/pipeline-definition.json").read_text())


def validate(state, root, complete=False, before=None, submission_ready=False):
    spec = definition()
    errors = []
    if state.get("schema_version") != 1 or state.get("pipeline") != spec["pipeline"]:
        errors.append("wrong schema or pipeline")
    rows = state.get("stages", [])
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        return errors + ["stages must be objects in canonical order"]
    if [r.get("id") for r in rows] != spec["stages"]:
        return errors + ["missing, duplicate, extra or reordered stage"]
    version = state.get("contract_version")
    if not isinstance(version, str) or not version.strip():
        errors.append("contract_version required")
    def files_ok(items, label):
        if not isinstance(items, list) or not items:
            errors.append(label + ": nonempty file/hash list required")
            return
        for item in items:
            if not isinstance(item, dict):
                errors.append(label + ": malformed artifact")
                continue
            raw = item.get("path")
            if not isinstance(raw, str) or not raw:
                errors.append(label + ": path required")
                continue
            path = (root / raw).resolve()
            if Path(raw).is_absolute() or not path.is_relative_to(root):
                errors.append(label + ": artifact outside run root")
            elif not path.is_file() or path.stat().st_size == 0:
                errors.append(label + ": missing or empty artifact " + raw)
            elif item.get("sha256") != digest(path):
                errors.append(label + ": stale hash " + raw)
    files_ok(state.get("contract"), "contract")
    profile = state.get("execution_profile", "empirical")
    if profile not in {"empirical", "theory-only"}:
        errors.append("unknown execution profile")
    if profile == "theory-only":
        files_ok(state.get("profile"), "theory profile")
        files_ok(state.get("authorization"), "theory authorization")
    gap = False
    for row in rows:
        status = row.get("status")
        if status not in spec["statuses"]:
            errors.append(row["id"] + ": invalid status")
        if status == "passed":
            if profile == "theory-only":
                if row.get("gate_basis") != "theory-only":
                    errors.append(row["id"] + ": theory gate basis required")
                if not isinstance(row.get("actual_executor"), str) or not row["actual_executor"].strip():
                    errors.append(row["id"] + ": actual executor required")
            if gap:
                errors.append(row["id"] + ": predecessor has not passed")
            if row.get("contract_version") != version:
                errors.append(row["id"] + ": stale contract version")
            if row.get("gate_verdict") != "pass":
                errors.append(row["id"] + ": gate verdict must be pass")
            files_ok(row.get("inputs"), row["id"] + " inputs")
            files_ok(row.get("evidence"), row["id"] + " evidence")
        else:
            if gap and status in ("running", "blocked"):
                errors.append(row["id"] + ": work started beyond first unfinished stage")
            gap = True
    complete = complete or submission_ready
    if complete or submission_ready:
        mode = state.get("delivery_mode")
        if mode not in {"preparation", "submission-ready"}:
            errors.append("delivery_mode must be preparation or submission-ready")
        readiness = state.get("readiness")
        readiness_doc = None
        if not isinstance(readiness, list) or len(readiness) != 1:
            errors.append("readiness: exactly one file/hash record required for completion")
        else:
            files_ok(readiness, "readiness")
            item = readiness[0]
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                raw = item["path"]
                readiness_path = (root / raw).resolve()
                if not Path(raw).is_absolute() and readiness_path.is_relative_to(root) and readiness_path.is_file():
                    try:
                        readiness_doc = json.loads(readiness_path.read_text(encoding="utf-8-sig"))
                    except (OSError, ValueError) as exc:
                        errors.append("readiness: cannot load JSON: " + str(exc))
                    if readiness_doc is not None:
                        readiness_errors = validate_readiness(readiness_doc)
                        errors.extend("readiness: " + error for error in readiness_errors)
                        if isinstance(readiness_doc, dict) and readiness_doc.get("delivery_mode") != mode:
                            errors.append("readiness delivery_mode does not match state delivery_mode")
        if submission_ready and mode != "submission-ready":
            errors.append("--submission-ready requires state delivery_mode=submission-ready")
        citation_gate = state.get("citation_gate")
        if not isinstance(citation_gate, list) or len(citation_gate) != 1:
            errors.append("citation_gate: exactly one report file/hash record required for completion")
        else:
            files_ok(citation_gate, "citation_gate")
            record = citation_gate[0]
            if isinstance(record, dict) and isinstance(record.get("path"), str):
                raw_report = record["path"]
                report_path = (root / raw_report).resolve()
                if not Path(raw_report).is_absolute() and report_path.is_relative_to(root) and report_path.is_file():
                    try:
                        report_doc = json.loads(report_path.read_text(encoding="utf-8-sig"))
                    except (OSError, ValueError) as exc:
                        report_doc = None
                        errors.append("citation_gate: cannot load report JSON: " + str(exc))
                    actual_inputs = {}
                    roles = {"bib", "manuscript", "inventory", "policy"}
                    if report_doc is not None:
                        if not isinstance(report_doc, dict) or report_doc.get("verdict") != "pass":
                            errors.append("citation_gate: report verdict must be pass")
                        binding = report_doc.get("inputs") if isinstance(report_doc, dict) else None
                        if not isinstance(binding, dict) or set(binding) != roles:
                            errors.append("citation_gate: report inputs must bind exactly bib/manuscript/inventory/policy roles")
                        else:
                            for role in sorted(roles):
                                entry = binding[role]
                                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                                    errors.append("citation_gate: malformed input binding for " + role)
                                    continue
                                raw_input = entry["path"]
                                input_path = Path(raw_input)
                                if not input_path.is_absolute():
                                    input_path = root / input_path
                                input_path = input_path.resolve()
                                if not input_path.is_relative_to(root):
                                    errors.append("citation_gate: input outside state root: " + role)
                                elif not input_path.is_file() or input_path.stat().st_size == 0:
                                    errors.append("citation_gate: missing or empty bound input: " + role)
                                elif entry.get("sha256") != digest(input_path):
                                    errors.append("citation_gate: stale bound input hash: " + role)
                                else:
                                    actual_inputs[role] = input_path
                            if set(actual_inputs) == roles:
                                try:
                                    rerun = check_citations(actual_inputs)
                                    if rerun.get("verdict") != "pass":
                                        errors.extend("citation_gate rerun: " + str(e) for e in rerun.get("errors", ["gate failed"]))
                                except Exception as exc:
                                    errors.append("citation_gate rerun failed: " + str(exc))
        if any(r.get("status") != "passed" for r in rows):
            errors.append("workflow completion requires every stage passed")
    if before is not None:
        if before not in spec["stages"]:
            errors.append("unknown requested stage")
        elif any(r.get("status") != "passed" for r in rows[:spec["stages"].index(before)]):
            errors.append("requested stage has unfinished predecessors")
    events = state.get("history", [])
    if not isinstance(events, list):
        errors.append("history must be a list")
        events = []
    accepted = []
    for event in events:
        if not isinstance(event, dict):
            errors.append("malformed history event")
            continue
        stage = event.get("stage")
        if event.get("action") == "pass":
            if len(accepted) >= len(spec["stages"]) or stage != spec["stages"][len(accepted)]:
                errors.append("history has out-of-order pass")
            else:
                accepted.append(stage)
        elif event.get("action") == "reopen":
            if stage not in spec["stages"] or not isinstance(event.get("reason"), str) or not event["reason"].strip():
                errors.append("reopen requires valid stage and reason")
            elif spec["stages"].index(stage) > len(accepted):
                errors.append("cannot reopen beyond first unfinished stage")
            else:
                accepted = accepted[:spec["stages"].index(stage)]
        else:
            errors.append("unknown history action")
    if accepted != [r["id"] for r in rows if r.get("status") == "passed"]:
        errors.append("history does not match passed stages")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--init", action="store_true", help="create pending ledger, never overwrite")
    parser.add_argument("--workflow-complete", "--complete", dest="workflow_complete", action="store_true", help="require all local pipeline stages and readiness record")
    parser.add_argument("--submission-ready", action="store_true", help="also require submission-ready mode and no pending items")
    parser.add_argument("--before", help="require all predecessors passed")
    args = parser.parse_args()
    spec = definition()
    try:
        if args.init:
            state = {"schema_version": 1, "pipeline": spec["pipeline"], "contract_version": "v1",
                     "delivery_mode": "preparation", "readiness": [], "citation_gate": [],
                     "contract": [], "history": [], "stages": [
                         {"id": stage, "status": "pending", "contract_version": "v1",
                          "inputs": [], "evidence": [], "gate_verdict": "pending"}
                         for stage in spec["stages"]]}
            with args.state.open("x", encoding="utf-8") as stream:
                json.dump(state, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print("Initialized pending ledger; populate contract files and hashes before execution.")
            return 0
        state = json.loads(args.state.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("state must be an object")
        errors = validate(state, args.state.resolve().parent, args.workflow_complete or args.submission_ready, args.before, args.submission_ready)
        if errors:
            for error in errors:
                print("ERROR:", error)
            return 1
        print("PASS: ordered local workflow evidence and readiness-record checks; ")
        print("submission readiness is a separate recorded fact; scientific validity is not inferred.")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("ERROR:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
