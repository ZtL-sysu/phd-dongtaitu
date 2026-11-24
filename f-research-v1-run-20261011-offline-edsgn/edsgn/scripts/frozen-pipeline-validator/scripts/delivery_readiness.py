"""Validate deferred author/admin facts without treating them as workflow blockers.

This checks document structure and status consistency only. It cannot establish that
sources are truthful, approvals were actually granted, or scientific claims are valid.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


CATEGORIES = {"author-fact", "author-approval", "access-route", "production", "portal"}
STATUSES = {"pending", "resolved"}
PLACEHOLDER_MARKERS = re.compile(
    r"\b(?:tbd|todo|placeholder|unknown|not yet confirmed|awaiting approval|pending confirmation)\b",
    re.IGNORECASE,
)


def validate_readiness(doc: Any) -> list[str]:
    """Return structural/schema errors; never infer truth from prose or a source link."""
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]
    if doc.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    mode = doc.get("delivery_mode")
    if mode not in {"preparation", "submission-ready"}:
        errors.append("delivery_mode must be preparation or submission-ready")
    items = doc.get("deferred_items")
    if not isinstance(items, list):
        return errors + ["deferred_items must be a list"]

    seen: set[str] = set()
    pending_found = False
    for index, item in enumerate(items):
        prefix = f"deferred_items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            errors.append(f"{prefix}.id must be a nonempty string")
        elif item_id in seen:
            errors.append(f"{prefix}.id must be unique")
        else:
            seen.add(item_id)

        category = item.get("category")
        if category not in CATEGORIES:
            errors.append(f"{prefix}.category must be one of {', '.join(sorted(CATEGORIES))}")
        status = item.get("status")
        if status not in STATUSES:
            errors.append(f"{prefix}.status must be pending or resolved")
        if status == "pending":
            pending_found = True

        for field in ("owner", "requirement", "source", "unblock_action"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{field} must be a nonempty string")

        surfaces = item.get("affected_surfaces")
        if (not isinstance(surfaces, list) or not surfaces
                or any(not isinstance(surface, str) or not surface.strip() for surface in surfaces)):
            errors.append(f"{prefix}.affected_surfaces must be a nonempty list of nonempty strings")

        if item.get("scientific_effect") != "none":
            errors.append(f"{prefix}.scientific_effect must be exactly none")

        # A resolved item cannot use obvious placeholder/pending language as proof.
        # This is only a guardrail; semantic truth still requires human review.
        if status == "resolved":
            for field in ("owner", "requirement", "source", "unblock_action"):
                value = item.get(field)
                if isinstance(value, str) and PLACEHOLDER_MARKERS.search(value):
                    errors.append(f"{prefix}.{field} contains placeholder/pending language for a resolved item")

    if mode == "submission-ready" and pending_found:
        errors.append("submission-ready mode cannot contain pending deferred_items")
    derived_ready = mode == "submission-ready" and not pending_found
    if "submission_ready" in doc:
        declared = doc.get("submission_ready")
        if not isinstance(declared, bool):
            errors.append("submission_ready must be a boolean when present")
        elif declared and not derived_ready:
            errors.append("submission_ready=true conflicts with preparation mode or pending items")
    return errors


def build_report(doc: Any, errors: list[str]) -> dict[str, Any]:
    mode = doc.get("delivery_mode") if isinstance(doc, dict) else None
    items = doc.get("deferred_items", []) if isinstance(doc, dict) else []
    items = items if isinstance(items, list) else []
    pending_count = sum(
        1 for item in items if isinstance(item, dict) and item.get("status") == "pending"
    )
    ready = (
        not errors
        and mode == "submission-ready"
        and pending_count == 0
    )
    return {
        "ok": not errors,
        "delivery_mode": mode,
        "submission_ready": ready,
        "deferred_items": len(items),
        "pending_items": pending_count,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path, help="readiness JSON document")
    parser.add_argument(
        "--submission-ready",
        action="store_true",
        help="require submission-ready mode and no pending items",
    )
    parser.add_argument("--report-file", type=Path, help="optional JSON report output path")
    args = parser.parse_args(argv)
    try:
        doc = json.loads(args.file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        report = {
            "ok": False,
            "delivery_mode": None,
            "submission_ready": False,
            "deferred_items": 0,
            "pending_items": 0,
            "errors": [f"cannot read JSON: {exc}"],
        }
    else:
        errors = validate_readiness(doc)
        if args.submission_ready and isinstance(doc, dict):
            if doc.get("delivery_mode") != "submission-ready":
                errors.append("--submission-ready requires delivery_mode=submission-ready")
            items = doc.get("deferred_items", [])
            if isinstance(items, list) and any(
                isinstance(item, dict) and item.get("status") == "pending" for item in items
            ):
                # validate_readiness also reports this for submission-ready mode; this
                # message covers preparation mode requested with the CLI flag.
                if "submission-ready mode cannot contain pending deferred_items" not in errors:
                    errors.append("--submission-ready cannot pass while any item is pending")
        report = build_report(doc, errors)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    try:
        if args.report_file:
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            with args.report_file.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        print(serialized, end="")
    except OSError as exc:
        print(json.dumps({"ok": False, "errors": [f"cannot write report: {exc}"]}, ensure_ascii=False))
        return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

