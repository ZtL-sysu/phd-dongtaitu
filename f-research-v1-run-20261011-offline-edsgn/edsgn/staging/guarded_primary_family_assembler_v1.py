"""Fail-closed, JSON-only assembly of root-accepted primary endpoint results.

This staging assembler does not read prediction arrays or make scientific
claims. It requires the complete 49-row P5 registry, all 49 producer records,
and a separate root-authored acceptance map pinning every producer artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe")
PIPELINE_VALIDATOR = ROOT / "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py"
REGISTRY_DEFINITIONS = ROOT / "protocols/primary-endpoint-registry-root-preparation-v2.json"
HOLM_PATH = ROOT / "edsgn/staging/typed_primary_family_holm_v3.py"
HOLM_ACCEPTANCE_PATH = ROOT / "reviews/typed-primary-family-holm-root-acceptance-v3.json"
PRODUCER_PATH = ROOT / "edsgn/staging/typed_endpoint_producer_v3.py"
PLAN_VALIDATOR_PATH = ROOT / "edsgn/staging/typed_endpoint_plan_validator_v6.py"
EXPECTED_FAMILIES = {"H1": 9, "H2": 16, "H3": 24}
SEEDS = [0, 1, 2, 3, 4]
ACCEPTANCE_SCOPE = "P5 root acceptance of all primary endpoint producer records"
PRODUCER_SCOPE = "typed endpoint sufficient-statistic/bootstrap producer; no family claim decision"
EXPECTED_RAW_P_KIND = {
    "forecast_mae_noninferiority": "centered_lower_tail_noninferiority",
    "forecast_mae_superiority": "centered_two_sided_p",
    "graph_tv_noise_superiority": "centered_two_sided_p",
    "graph_tv_response_superiority": "centered_two_sided_p",
    "paired_window_cost_superiority": "centered_two_sided_p",
}


class AssemblyError(RuntimeError):
    pass


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise AssemblyError(f"duplicate JSON key {key}: {path}")
            out[key] = value
        return out

    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(AssemblyError(f"nonfinite JSON constant {token}")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssemblyError(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, raw: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise AssemblyError("artifact path must be a nonempty string")
    path = Path(raw)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise AssemblyError(f"artifact escapes RUN: {raw}")
    return path


def _load_holm():
    spec = importlib.util.spec_from_file_location("_guarded_primary_holm_v3", HOLM_PATH)
    if spec is None or spec.loader is None:
        raise AssemblyError("cannot load the accepted typed Holm kernel")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_accepted_holm(root: Path) -> tuple[str, dict[str, str]]:
    accepted = _strict_json(HOLM_ACCEPTANCE_PATH)
    evidence = accepted.get("evidence_sha256")
    rel_holm = HOLM_PATH.relative_to(root).as_posix()
    if (accepted.get("accepted_by") != "root" or accepted.get("verdict") != "pass"
            or accepted.get("scope") != "engineering-staging-only"
            or accepted.get("actual_primary_inference_run") is not False
            or not isinstance(evidence, dict) or evidence.get(rel_holm) != sha256(HOLM_PATH)):
        raise AssemblyError("typed Holm v3 kernel lacks its exact root engineering acceptance")
    verified = {}
    for rel, digest in evidence.items():
        path = _resolve(root, rel)
        if not path.is_file() or sha256(path) != digest:
            raise AssemblyError(f"root-accepted Holm evidence changed: {rel}")
        verified[rel] = digest
    return sha256(HOLM_ACCEPTANCE_PATH), verified


def _official_p5_gate(root: Path, *, official: bool) -> tuple[dict, str]:
    state_path = root / "pipeline-state.json"
    state_bytes = state_path.read_bytes()
    state = _strict_json(state_path)
    stages = state.get("stages")
    if not isinstance(stages, list):
        raise AssemblyError("canonical stage rows are absent")
    ids = [row.get("id") if isinstance(row, dict) else None for row in stages]
    expected_order = ["P0", "P1", "P1A", "P1B", "P2", "P3", "P3A", "P4", "P5", "P5A"] + [f"P{i}" for i in range(6, 22)]
    if ids != expected_order:
        raise AssemblyError("canonical stage order is malformed")
    by_id = {row["id"]: row for row in stages}
    if any(by_id[name].get("status") != "passed" or by_id[name].get("gate_verdict") != "pass"
           for name in expected_order[:8]):
        raise AssemblyError("primary-family assembly requires P0-P4 passed/pass")
    if by_id["P5"].get("status") != "running":
        raise AssemblyError("primary-family assembly requires P5 running")
    if by_id["P5"].get("gate_verdict") not in (None, ""):
        raise AssemblyError("P5 must not already have a final gate verdict")
    if official:
        completed = subprocess.run(
            [str(PYTHON), str(PIPELINE_VALIDATOR), str(state_path), "--before", "P5"],
            cwd=root, capture_output=True, text=True, shell=False,
        )
        if completed.returncode:
            raise AssemblyError(f"official before-P5 validator rejected canonical state: {completed.stdout[-1200:]} {completed.stderr[-1200:]}")
    if state_path.read_bytes() != state_bytes:
        raise AssemblyError("canonical pipeline changed during P5 predecessor validation")
    return state, hashlib.sha256(state_bytes).hexdigest()


def _registry_rows(root: Path, registry_path: Path, definitions_path: Path):
    registry = _strict_json(registry_path)
    if (registry.get("schema_version") != 1 or registry.get("split") != "test"
            or registry.get("artifact_status") != "accepted_pretest_registry"):
        raise AssemblyError("registry must be the accepted pretest test registry")
    plans = registry.get("endpoint_plans")
    if not isinstance(plans, list) or len(plans) != 49:
        raise AssemblyError("accepted registry must bind exactly 49 primary endpoint plans")
    definitions = _strict_json(definitions_path)
    structural = definitions.get("primary_endpoints")
    if (definitions.get("artifact_status") != "root_structural_preparation_not_frozen_not_executed"
            or definitions.get("split") != "test" or not isinstance(structural, list) or len(structural) != 49):
        raise AssemblyError("root prospective 49-endpoint definition registry is malformed")
    definition_by_id = {}
    for row in structural:
        endpoint_id = row.get("endpoint_id")
        if not isinstance(endpoint_id, str) or endpoint_id in definition_by_id:
            raise AssemblyError("prospective endpoint definition IDs are missing or duplicated")
        definition_by_id[endpoint_id] = row
    if len(definition_by_id) != 49:
        raise AssemblyError("prospective endpoint definitions are not a 49-row unique registry")

    plan_rows = []
    seen = set()
    family_counts = {family: 0 for family in EXPECTED_FAMILIES}
    for item in plans:
        if not isinstance(item, dict) or set(item) != {"endpoint_id", "claim", "path", "sha256"}:
            raise AssemblyError("registry plan refs must contain exactly endpoint_id/claim/path/sha256")
        endpoint_id = item["endpoint_id"]
        if endpoint_id in seen or endpoint_id not in definition_by_id:
            raise AssemblyError(f"registry endpoint ID is duplicate or not preregistered: {endpoint_id}")
        seen.add(endpoint_id)
        claim = item["claim"]
        if claim not in EXPECTED_FAMILIES or claim != endpoint_id.split("-", 1)[0]:
            raise AssemblyError(f"registry claim does not match endpoint family: {endpoint_id}")
        path = _resolve(root, item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise AssemblyError(f"registered plan path/hash is stale: {item['path']}")
        plan = _strict_json(path)
        definition = definition_by_id[endpoint_id]
        if (plan.get("schema_version") != 2 or plan.get("endpoint_id") != endpoint_id
                or plan.get("claim") != claim or definition.get("claim") != claim
                or plan.get("outcome_type") != definition.get("outcome_type")
                or definition.get("primary") is not True
                or definition.get("holm_family") != claim):
            raise AssemblyError(f"plan differs from frozen primary endpoint identity/type: {endpoint_id}")
        if plan.get("seed_ids") != SEEDS:
            raise AssemblyError(f"plan does not bind seeds 0..4: {endpoint_id}")
        family_counts[claim] += 1
        plan_rows.append((item, path, plan, definition))
    if seen != set(definition_by_id) or family_counts != EXPECTED_FAMILIES:
        raise AssemblyError(f"registry family or endpoint coverage mismatch: {family_counts}")
    return registry, plan_rows, definitions


def _acceptance_records(root: Path, acceptance_path: Path, registry_path: Path, registry_sha: str,
                        definitions: dict, plan_rows: list):
    accepted = _strict_json(acceptance_path)
    required = {"schema_version", "status", "accepted_by", "verdict", "scope", "split",
                "registry_path", "registry_sha256", "test_feedback_used", "pipeline_advanced",
                "scientific_claim_decision", "records"}
    if set(accepted) != required:
        raise AssemblyError("root acceptance map has unexpected or missing fields")
    if (accepted.get("schema_version") != 1 or accepted.get("status") != "accepted"
            or accepted.get("accepted_by") != "root" or accepted.get("verdict") != "pass"
            or accepted.get("scope") != ACCEPTANCE_SCOPE or accepted.get("split") != "test"
            or accepted.get("registry_path") != registry_path.relative_to(root).as_posix()
            or accepted.get("registry_sha256") != registry_sha
            or accepted.get("test_feedback_used") is not False
            or accepted.get("pipeline_advanced") is not False
            or accepted.get("scientific_claim_decision") != "not_made"):
        raise AssemblyError("root acceptance does not attest exact non-advancing result review scope")
    rows = accepted.get("records")
    if not isinstance(rows, list) or len(rows) != 49:
        raise AssemblyError("root acceptance map must explicitly accept all 49 producer records")
    expected = {}
    definitions_by_id = {row["endpoint_id"]: row for row in definitions["primary_endpoints"]}
    for item, _path, _plan, _definition in plan_rows:
        endpoint_id = item["endpoint_id"]
        result_path = definitions_by_id[endpoint_id].get("result_path")
        if not isinstance(result_path, str) or not result_path:
            raise AssemblyError(f"preregistered result path is absent: {endpoint_id}")
        expected[endpoint_id] = (result_path, None)
    found = {}
    resolved_paths = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"endpoint_id", "path", "sha256", "accepted"}:
            raise AssemblyError("root acceptance record rows must bind endpoint/path/hash/accepted")
        endpoint_id, raw_path, digest = row["endpoint_id"], row["path"], row["sha256"]
        if endpoint_id not in expected or endpoint_id in found or row["accepted"] is not True:
            raise AssemblyError("root acceptance has duplicate, extra, or unaccepted producer record")
        path = _resolve(root, raw_path)
        expected_path = _resolve(root, expected[endpoint_id][0])
        if path != expected_path or path in resolved_paths:
            raise AssemblyError(f"root acceptance producer path differs from preregistered output: {endpoint_id}")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AssemblyError(f"root acceptance has malformed producer SHA: {endpoint_id}")
        if not path.is_file() or sha256(path) != digest:
            raise AssemblyError(f"root-accepted producer record is missing or stale: {endpoint_id}")
        found[endpoint_id] = (path, digest)
        resolved_paths.add(path)
    if set(found) != set(expected):
        raise AssemblyError("root acceptance omits preregistered endpoint producer records")
    return accepted, found


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AssemblyError(f"{label} must be finite numeric")
    return float(value)


def _map_producer_result(root: Path, path: Path, expected_plan_ref: dict, plan: dict,
                         registry_path: Path, registry_sha: str, producer_path: Path) -> dict[str, Any]:
    result = _strict_json(path)
    if (result.get("schema_version") != 1 or result.get("scope") != PRODUCER_SCOPE
            or result.get("split") != "test" or result.get("endpoint_id") != plan["endpoint_id"]
            or result.get("claim") != plan["claim"] or result.get("outcome_type") != plan["outcome_type"]
            or result.get("plan_path") != expected_plan_ref["path"]
            or result.get("plan_sha256") != expected_plan_ref["sha256"]
            or result.get("registry_path") != registry_path.relative_to(root).as_posix()
            or result.get("registry_sha256") != registry_sha):
        raise AssemblyError(f"producer output identity/schema differs from registered plan: {path}")
    if result.get("seed_ids") != SEEDS or not isinstance(result.get("n_full_origins"), int) or result["n_full_origins"] <= 0:
        raise AssemblyError(f"producer output lacks complete-seed/full-origin metadata: {path}")
    sources = result.get("producer_source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise AssemblyError(f"producer output lacks source hash map: {path}")
    for raw_source, digest in sources.items():
        source = _resolve(root, raw_source)
        if not source.is_file() or sha256(source) != digest:
            raise AssemblyError(f"producer source changed since endpoint computation: {raw_source}")
    if str(producer_path.resolve()) not in sources or sources[str(producer_path.resolve())] != sha256(producer_path):
        raise AssemblyError("producer output does not bind current producer source bytes")

    outcome = plan["outcome_type"]
    if outcome not in EXPECTED_RAW_P_KIND:
        raise AssemblyError(f"unsupported producer outcome type: {outcome}")
    identifiable = result.get("identifiable")
    if not isinstance(identifiable, bool):
        raise AssemblyError(f"producer identifiability must be boolean: {path}")
    mapped = {"endpoint_id": plan["endpoint_id"], "outcome_type": outcome, "identifiable": identifiable}
    if not identifiable:
        if result.get("raw_p_kind") not in (None, EXPECTED_RAW_P_KIND[outcome]):
            raise AssemblyError("unidentifiable producer result has an incompatible raw-p kind")
        if result.get("raw_p") is not None:
            raise AssemblyError("unidentifiable producer result must have raw_p=null")
        if any(result.get(k) is not None for k in ("effect_estimate", "percentile_95_interval", "upper_95", "epsilon_raw")):
            raise AssemblyError("unidentifiable producer result must not carry inferential quantities")
        mapped["raw_p"] = None
        return mapped

    if result.get("raw_p_kind") != EXPECTED_RAW_P_KIND[outcome]:
        raise AssemblyError(f"producer raw p-value kind is not the preregistered kind for {outcome}")
    raw_p = _finite(result.get("raw_p"), "producer raw_p")
    estimate = _finite(result.get("effect_estimate"), "producer effect_estimate")
    analysis = result.get("analysis")
    if not isinstance(analysis, dict) or _finite(analysis.get(result["raw_p_kind"]), "analysis raw p") != raw_p:
        raise AssemblyError("top-level raw p does not equal the saved producer analysis p")
    if _finite(analysis.get("estimate"), "analysis estimate") != estimate:
        raise AssemblyError("top-level estimate does not equal saved producer analysis estimate")
    mapped.update(raw_p=raw_p, estimate=estimate)
    if outcome == "forecast_mae_noninferiority":
        epsilon = _finite(result.get("epsilon_raw"), "producer epsilon_raw")
        upper = _finite(result.get("upper_95"), "producer upper_95")
        if epsilon <= 0 or _finite(analysis.get("upper_95"), "analysis upper_95") != upper:
            raise AssemblyError("NI epsilon or one-sided upper bound is invalid/inconsistent")
        mapped.update(epsilon=epsilon, one_sided95_upper=upper)
    else:
        interval = result.get("percentile_95_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise AssemblyError("superiority producer result lacks its percentile 95 interval")
        lower = _finite(interval[0], "producer percentile 95 lower")
        upper = _finite(interval[1], "producer percentile 95 upper")
        saved_interval = analysis.get("percentile_95_interval")
        if (not isinstance(saved_interval, list) or len(saved_interval) != 2
                or _finite(saved_interval[0], "analysis percentile 95 lower") != lower
                or _finite(saved_interval[1], "analysis percentile 95 upper") != upper
                or lower > upper):
            raise AssemblyError("top-level percentile upper does not match producer analysis")
        mapped["percentile95_upper"] = upper
    return mapped


def assemble_primary_families(registry_path: Path, acceptance_path: Path, output_path: Path,
                              *, root: Path = ROOT, official: bool = True,
                              _test_root: Path | None = None) -> dict[str, Any]:
    """Validate all byte bindings, apply Holm3, and exclusively serialize one result."""
    if _test_root is not None:
        root = Path(_test_root).resolve()
        official = False
    else:
        root = Path(root).resolve()
        if root != ROOT.resolve() or official is not True:
            raise AssemblyError("public assembler requires canonical RUN and official P5 validator")
    registry_path = Path(registry_path).resolve()
    acceptance_path = Path(acceptance_path).resolve()
    output_path = Path(output_path).resolve()
    for path in (registry_path, acceptance_path, output_path):
        if not path.is_relative_to(root):
            raise AssemblyError("registry, root acceptance, and output must remain within RUN")
    if output_path.exists():
        raise FileExistsError(output_path)

    source_paths = (Path(__file__).resolve(), HOLM_PATH, PRODUCER_PATH, PLAN_VALIDATOR_PATH,
                    PIPELINE_VALIDATOR, REGISTRY_DEFINITIONS, HOLM_ACCEPTANCE_PATH)
    source_before = {str(path.resolve()): sha256(path) for path in source_paths}
    holm_acceptance_sha = None
    holm_acceptance_evidence = {}
    if official:
        holm_acceptance_sha, holm_acceptance_evidence = _verify_accepted_holm(root)
    state, state_sha = _official_p5_gate(root, official=official)
    registry_sha = sha256(registry_path)
    definitions_sha = sha256(REGISTRY_DEFINITIONS)
    registry, plan_rows, definitions = _registry_rows(root, registry_path, REGISTRY_DEFINITIONS)
    accepted, records = _acceptance_records(root, acceptance_path, registry_path, registry_sha, definitions, plan_rows)
    acceptance_sha = sha256(acceptance_path)
    plan_hashes = {path.relative_to(root).as_posix(): ref["sha256"] for ref, path, _plan, _definition in plan_rows}

    registry_numeric_rows = []
    numeric_rows = []
    result_hashes = {}
    for ref, plan_path, plan, _definition in plan_rows:
        endpoint_id = ref["endpoint_id"]
        result_path, result_sha = records[endpoint_id]
        numeric_rows.append(_map_producer_result(root, result_path, ref, plan, registry_path,
                                                 registry_sha, PRODUCER_PATH))
        registry_numeric_rows.append({"endpoint_id": endpoint_id, "claim": plan["claim"],
                                      "outcome_type": plan["outcome_type"], "primary": True,
                                      "holm_family": plan["claim"]})
        result_hashes[result_path.relative_to(root).as_posix()] = result_sha

    holm = _load_holm()
    families = holm.assemble_primary_families(registry_numeric_rows, numeric_rows, alpha=definitions.get("alpha", 0.05))
    result = {
        "schema_version": 1,
        "scope": "JSON-only accepted primary-family Holm assembly; no arrays read and no pipeline decision",
        "split": "test",
        "registry_path": registry_path.relative_to(root).as_posix(),
        "registry_sha256": registry_sha,
        "root_acceptance_path": acceptance_path.relative_to(root).as_posix(),
        "root_acceptance_sha256": acceptance_sha,
        "definitions_path": REGISTRY_DEFINITIONS.relative_to(root).as_posix(),
        "definitions_sha256": definitions_sha,
        "pipeline_sha256": state_sha,
        "pipeline_stages": {row["id"]: row.get("status") for row in state["stages"] if row["id"] in {"P2", "P3", "P3A", "P4", "P5"}},
        "producer_record_sha256": result_hashes,
        "holm_kernel_sha256": source_before[str(HOLM_PATH.resolve())],
        "holm_root_acceptance_sha256": holm_acceptance_sha,
        "holm_root_acceptance_evidence": holm_acceptance_evidence,
        "families": families["families"],
        "scientific_claim_decision": "not_made",
        "pipeline_advanced": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        output_sha = sha256(output_path)
        _post_state, post_state_sha = _official_p5_gate(root, official=official)
        if post_state_sha != state_sha:
            raise AssemblyError("canonical P5 gate/state changed during family assembly")
        if sha256(registry_path) != registry_sha or sha256(acceptance_path) != acceptance_sha:
            raise AssemblyError("registry or root acceptance changed during assembly")
        if sha256(REGISTRY_DEFINITIONS) != definitions_sha:
            raise AssemblyError("prospective endpoint definitions changed during assembly")
        if official and (sha256(HOLM_ACCEPTANCE_PATH) != holm_acceptance_sha
                         or _verify_accepted_holm(root)[0] != holm_acceptance_sha):
            raise AssemblyError("root-accepted Holm kernel authority changed during assembly")
        for rel, digest in result_hashes.items():
            if sha256(_resolve(root, rel)) != digest:
                raise AssemblyError(f"accepted endpoint producer output changed during assembly: {rel}")
        for rel, digest in plan_hashes.items():
            if sha256(_resolve(root, rel)) != digest:
                raise AssemblyError(f"registered endpoint plan changed during assembly: {rel}")
        if {str(Path(path)): sha256(Path(path)) for path in source_before} != source_before:
            raise AssemblyError("assembler/kernel/producer/validator source changed during assembly")
        if sha256(output_path) != output_sha:
            raise AssemblyError("assembled output changed after exclusive serialization")
    except BaseException:
        # Preserve any exclusive partial/result artifact for audit. Never delete
        # evidence as a recovery shortcut.
        raise
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--root-acceptance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = assemble_primary_families(Path(args.registry), Path(args.root_acceptance), Path(args.output))
    print(json.dumps({"output": str(Path(args.output).resolve()), "family_sizes": {k: v["expected_size"] for k, v in result["families"].items()},
                      "pipeline_advanced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
