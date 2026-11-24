"""Prepare schema-4 prospective primary endpoint plans (metadata only).

This generator writes no accepted registry, launch binding, result, or future
artifact hashes. Domain preparation uses C06 metadata/mask/header-only APIs;
archive digests are byte commitments and do not decode value/target/oracle
members.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "protocols/primary-endpoint-plans-v5"
REGISTRY = ROOT / "protocols/primary-endpoint-registry-root-preparation-v2.json"
CATALOG = ROOT / "protocols/core-formal-default-candidate-catalog-v2.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_exclusive(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"existing artifact differs; refusing overwrite: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(payload.decode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def prepare_domains():
    sys.path.insert(0, str(ROOT / "edsgn/src"))
    sys.path.insert(0, str(ROOT / "edsgn/staging"))
    preflight = import_from(ROOT / "edsgn/staging/c06_domain_manifest_preflight_v4.py",
                            "_primary_plan_c06_preflight_v4")
    scenario_paths = {
        "noise-only": ROOT / "data/processed/synthetic-stress-n64-noise-only",
        "identity10": ROOT / "data/processed/synthetic-stress-n64-identity10",
        "identity30": ROOT / "data/processed/synthetic-n64-identity-change0.3-miss0",
        "weight10": ROOT / "data/processed/synthetic-stress-n64-weight10",
        "weight30": ROOT / "data/processed/synthetic-stress-n64-weight30",
        "email": ROOT / "data/processed/email/email.npz",
        "mooc": ROOT / "data/processed/mooc/mooc.npz",
    }
    accepted_refs = {
        "noise-only": ("protocols/synthetic-stress-domain-preparation-v1/n64_noise_only-test-domain.json",
                       "protocols/synthetic-stress-domain-preparation-v1/n64_noise_only-test-descriptor.json"),
        "identity10": ("protocols/synthetic-stress-domain-preparation-v1/n64_identity_change10-test-domain.json",
                       "protocols/synthetic-stress-domain-preparation-v1/n64_identity_change10-test-descriptor.json"),
        "identity30": ("protocols/synthetic-stress-domain-preparation-v1/n64_identity_change30_main_reuse-test-domain.json",
                       "protocols/synthetic-stress-domain-preparation-v1/n64_identity_change30_main_reuse-test-descriptor.json"),
        "weight10": ("protocols/synthetic-stress-domain-preparation-v1/n64_weight_change10-test-domain.json",
                     "protocols/synthetic-stress-domain-preparation-v1/n64_weight_change10-test-descriptor.json"),
        "weight30": ("protocols/synthetic-stress-domain-preparation-v1/n64_weight_change30-test-domain.json",
                     "protocols/synthetic-stress-domain-preparation-v1/n64_weight_change30-test-descriptor.json"),
        "email": ("protocols/c06-domain-preflight-root-v2/email-test-domain.json",
                  "protocols/c06-domain-preflight-root-v2/email-test-descriptor.json"),
        "mooc": ("protocols/c06-domain-preflight-root-v2/mooc-test-domain.json",
                 "protocols/c06-domain-preflight-root-v2/mooc-test-descriptor.json"),
    }
    domains = {}
    inventory = []
    for name, dataset_path in scenario_paths.items():
        paths, datasets, audit, recomputed_doc = preflight.domain_document(dataset_path, "test")
        domain_rel, descriptor_rel = accepted_refs[name]
        domain_source = ROOT / domain_rel
        descriptor_source = ROOT / descriptor_rel
        doc = load_json(domain_source)
        descriptor = load_json(descriptor_source)
        if doc != recomputed_doc:
            raise RuntimeError(f"accepted strict domain differs from C06 metadata-only recheck: {name}")
        descriptor_rows = descriptor.get("ordered_collection")
        expected_rows = [
            {"trajectory_index": i, "path": rel(Path(path)), "archive_sha256": sha(Path(path))}
            for i, path in enumerate(paths)
        ]
        if not isinstance(descriptor_rows, list):
            raise RuntimeError(f"accepted descriptor lacks ordered collection: {name}")
        projected = [{k: r.get(k) for k in ("trajectory_index", "path", "archive_sha256")}
                     for r in descriptor_rows]
        if projected != expected_rows:
            raise RuntimeError(f"accepted descriptor differs from metadata-only ordered bundles: {name}")
        domain_path = OUT / "domains" / f"{name}-test-domain.json"
        # Typed schema4 plan wrapper: preserve the accepted strict C06 domain
        # bytes/fields and add only the ordered hash-only bundle projection.
        doc = dict(doc)
        doc["ordered_collection"] = projected
        write_exclusive(domain_path, doc)
        collection_root = (dataset_path.parent if dataset_path.is_file() else dataset_path).resolve()
        all_files = sorted((p for p in collection_root.rglob("*") if p.is_file()),
                           key=lambda p: p.relative_to(ROOT).as_posix())
        domain_dataset_id = doc["dataset"]["source_name"]
        domains[name] = {
            "path": rel(domain_path), "sha256": sha(domain_path), "doc": doc,
            "accepted_domain": {"path": domain_rel, "sha256": sha(domain_source)},
            "accepted_descriptor": {"path": descriptor_rel, "sha256": sha(descriptor_source)},
            "collection": {"root": rel(collection_root), "files": [
                {"path": rel(p), "sha256": sha(p)} for p in all_files
            ]},
            "dataset_id": domain_dataset_id,
        }
        inventory.append({
            "scenario": name, "dataset_path": rel(dataset_path), "domain_path": rel(domain_path),
            "domain_sha256": sha(domain_path), "dataset_id": domain_dataset_id,
            "shape_tnf": doc["dataset"]["shape_tnf"], "history": doc["dataset"]["history_length"],
            "hmax": doc["dataset"]["hmax"], "split_bounds": [
                doc["dataset"]["split_bounds_start_inclusive"],
                doc["dataset"]["split_bounds_end_exclusive"]],
            "trajectory_count": len(paths),
            "full_test_origins": sum(t["origin_count"] for t in doc["trajectories"]),
            "collection_root": rel(collection_root), "collection_file_count": len(all_files),
            "decoded_npz_members": audit,
            "accepted_domain": {"path": domain_rel, "sha256": sha(domain_source)},
            "accepted_descriptor": {"path": descriptor_rel, "sha256": sha(descriptor_source)},
            "hash_only_members": ["values", "targets", "target_mask", "oracle members"],
        })
    return domains, inventory


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = load_json(REGISTRY)
    catalog = load_json(CATALOG)
    pools = {(row["dataset"], row["variant"]): row for row in catalog["pools"]}
    domains, domain_inventory = prepare_domains()
    helper_path = ROOT / "edsgn/staging/prospective_training_candidate_binding_v1.py"
    helper = import_from(helper_path, "_primary_plan_candidate_helper")
    # Validate only the existing pool/config/blueprint commitments. No selector
    # receipt, checkpoint, launch binding, or training result is created.
    prep_cache = {}
    for key, row in pools.items():
        prep = {"schema_version": 4, "candidate_pool": row["candidate_pool"],
                "comparison_id": row["comparison_id"]}
        for seed in range(5):
            refs = helper.validate_preparation(ROOT, prep, seed)
            if not refs:
                raise RuntimeError(f"candidate pool returned no source refs: {key}/{seed}")
            prep_cache[(key[0], key[1], seed)] = prep

    auth_index = load_json(ROOT / "protocols/synthetic-reporting-switch-authorities-v2/index.json")
    authority_by_id = {row["path"]: row["sha256"] for row in auth_index["authorities"]}
    authority_names = {
        "noise-only": "n64_noise_only-reporting-switch-authority.json",
        "identity10": "n64_identity_change10-reporting-switch-authority.json",
        "identity30": "n64_identity_change30_main_reuse-reporting-switch-authority.json",
        "weight10": "n64_weight_change10-reporting-switch-authority.json",
        "weight30": "n64_weight_change30-reporting-switch-authority.json",
    }

    common_sources = [
        "edsgn/staging/typed_endpoint_producer_v6.py",
        "edsgn/staging/typed_endpoint_plan_validator_v9.py",
        "edsgn/scripts/verified_frozen_dispatch_v10.py",
        "edsgn/scripts/bind_training_inputs.py",
        "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py",
        "edsgn/staging/endpoint_sufficient_statistics_v4.py",
        "edsgn/staging/paired_forecast_bootstrap_v1.py",
        "edsgn/staging/paired_scalar_bootstrap_v1.py",
        "edsgn/staging/prospective_training_candidate_binding_v1.py",
        "edsgn/staging/global_frozen_test_completion_v1.py",
    ]
    drivers = {
        "graph_tv_noise_superiority": "edsgn/staging/frozen_graph_state_diagnostic_v8.py",
        "graph_tv_response_superiority": "edsgn/staging/frozen_graph_state_diagnostic_v8.py",
        "forecast_mae_superiority": "edsgn/staging/frozen_prediction_guarded_v5.py",
        "forecast_mae_noninferiority": "edsgn/staging/frozen_prediction_guarded_v5.py",
        "paired_window_cost_superiority": "edsgn/staging/frozen_profile_guarded_v5.py",
    }
    blind_sources = ["edsgn/staging/blind_reference_cost_policy_v1.py",
                     "reviews/blind-reference-cost-policy-root-acceptance-v1.json"]

    def pool_variant(method, claim):
        if claim == "H3":
            return "V4"  # V7/V5/V6 are policy interventions on selected V4 weights.
        if method.startswith("V4-teacher-") or method.startswith("V4-negative-"):
            return "V4"
        return method

    def method_family(method):
        return "V4" if method.startswith("V4-") else method

    def prep_for(dataset, method, claim, seed):
        variant = pool_variant(method, claim)
        dataset_key = "synthetic" if dataset == "synthetic-N64" else dataset
        return prep_cache[(dataset_key, variant, seed)]

    def out_base(plan, domain_key, method, seed, device=None):
        family = method_family(method)
        if plan["outcome_type"] in {"graph_tv_noise_superiority", "graph_tv_response_superiority"}:
            return f"runs/primary-endpoint-v1/graph/{domain_key}/{family}/seed{seed}"
        if plan["outcome_type"] == "forecast_mae_superiority" and method.startswith("V4-teacher-"):
            return f"runs/primary-endpoint-v1/matched/{domain_key}/seed{seed}"
        if plan["claim"] == "H3":
            kind = "profile" if plan["outcome_type"] == "paired_window_cost_superiority" else "forecast"
            return f"runs/primary-endpoint-v1/H3/{domain_key}/{device}/{kind}/{family}/seed{seed}"
        return f"runs/primary-endpoint-v1/forecast/{domain_key}/{family}/seed{seed}"

    def artifacts(plan, domain_key, method, seed, device=None):
        outcome = plan["outcome_type"]
        base = out_base(plan, domain_key, method, seed, device)
        role_to_suffix = {
            "checkpoint": "checkpoint.pt", "freeze": "freeze.json",
            "input_binding": "input-binding.json", "driver_report": {
                "graph_tv_noise_superiority": "frozen-graph-state-diagnostic.json",
                "graph_tv_response_superiority": "frozen-graph-state-diagnostic.json",
                "forecast_mae_superiority": "frozen-prediction-report.json",
                "forecast_mae_noninferiority": "frozen-prediction-report.json",
                "paired_window_cost_superiority": "profile-report.json",
            }[outcome],
            "phase_receipt": "evidence/phase-receipt.json",
            "dispatch_receipt": "evidence/dispatch-receipt.json",
            "archive": "predictions.npz", "trace": "trace-rows.jsonl.gz",
            "csv": "profile-rows.csv", "validation_authority": "evidence/validation-authority.json",
            "policy": "policy.json", "queue_plan": "evidence/queue-plan.json",
            "matched_graph_proof": "matched-graph-proof.json",
        }
        roles = {
            "graph_tv_noise_superiority": ["checkpoint", "freeze", "input_binding", "driver_report", "phase_receipt", "dispatch_receipt", "trace", "validation_authority", "policy", "queue_plan"],
            "graph_tv_response_superiority": ["checkpoint", "freeze", "input_binding", "driver_report", "phase_receipt", "dispatch_receipt", "trace", "validation_authority", "policy", "queue_plan"],
            "forecast_mae_superiority": ["checkpoint", "freeze", "input_binding", "driver_report", "phase_receipt", "dispatch_receipt", "archive", "validation_authority", "policy", "queue_plan"],
            "forecast_mae_noninferiority": ["checkpoint", "freeze", "input_binding", "driver_report", "phase_receipt", "dispatch_receipt", "archive", "validation_authority", "policy", "queue_plan"],
            "paired_window_cost_superiority": ["checkpoint", "freeze", "input_binding", "driver_report", "phase_receipt", "dispatch_receipt", "csv", "validation_authority", "policy", "queue_plan"],
        }[outcome]
        if (plan["claim"] == "H2" and method.startswith("V4-teacher-")):
            roles.append("matched_graph_proof")
        paths = {role: f"{base}/{role_to_suffix[role]}" for role in roles}
        if (plan["claim"] == "H2" and method.startswith("V4-teacher-")):
            arm = {"V4-teacher-informative": "informative",
                   "V4-teacher-shuffled-c": "shuffled",
                   "V4-teacher-constant-g": "constant",
                   "V4-teacher-unconditional-neighbor-reset": "reset"}[method]
            paths["archive"] = f"{base}/predictions-{arm}.npz"
        if role_to_suffix.get("matched_graph_proof") and "matched_graph_proof" in paths:
            paths["matched_graph_proof"] = f"{base}/matched-graph-proof.json"
        return paths

    plan_inventory = []
    for row in registry["primary_endpoints"]:
        endpoint_id = row["endpoint_id"]
        domain_key = row["synthetic_case"] if row["claim"] in {"H1", "H2"} else row["domain"]
        domain = domains[domain_key]
        history = domain["doc"]["dataset"]["history_length"]
        hmax = domain["doc"]["dataset"]["hmax"]
        horizons = row["required_horizons"]
        outcome = row["outcome_type"]
        driver = drivers[outcome]
        source_paths = list(dict.fromkeys(common_sources + [driver,
            domain["accepted_domain"]["path"], domain["accepted_descriptor"]["path"]]))
        if row["claim"] == "H2" and "matched" in endpoint_id:
            source_paths.append("edsgn/staging/frozen_h2_matched_retention_v4.py")
        if row["claim"] == "H3" and ("V5" in row["left"] + row["right"] or "V6" in row["left"] + row["right"]):
            source_paths.extend(blind_sources)
        source_paths = list(dict.fromkeys(source_paths))
        producer_sources = [{"path": path, "sha256": sha(ROOT / path)} for path in source_paths]
        driver_sha = sha(ROOT / driver)
        side_methods = [row["left"], row["right"]]
        if outcome == "graph_tv_response_superiority":
            # One frozen V4 replay supports both paired temporal slices.
            side_methods = ["V4", "V4"]
        seed_sides = []
        for method in side_methods:
            seeds = []
            for seed in range(5):
                prep = prep_for(row["domain"], method, row["claim"], seed)
                paths = artifacts(row, domain_key, method, seed, row.get("device"))
                seeds.append({"seed_id": seed, "preparation": prep, "planned_paths": paths})
            seed_sides.append({"method_id": method, "producer_source_sha256": driver_sha, "seeds": seeds})

        acf_path = f"protocols/primary-endpoint-plans-v1/validation-calibration/{slug(domain['dataset_id'])}-L{history}-H{hmax}-acf.json"
        acf = {"mode": "validation_derived", "rule_id": "p3a-validation-acf-block-v1", "expected_record_path": acf_path}
        epsilon = None
        if outcome == "forecast_mae_noninferiority":
            eps_mode = "unidentifiable_by_rule" if row["domain"] == "email" else "validation_derived"
            epsilon = {"mode": eps_mode, "rule_id": "p3a-validation-raw-mae-epsilon-v1",
                       "expected_record_path": f"protocols/primary-endpoint-plans-v1/validation-calibration/{slug(domain['dataset_id'])}-H{'-'.join(map(str,horizons))}-epsilon.json",
                       "scope": "validation_only", "dataset_id": domain["dataset_id"],
                       "metric": "raw_mae", "horizons": horizons}

        authority = None
        if row["slice"] in {"full_post_switch", "post4L", "paired_pre_post4L"}:
            basename = authority_names[domain_key]
            authority_path = f"protocols/synthetic-reporting-switch-authorities-v2/{basename}"
            authority = {"path": authority_path, "sha256": sha(ROOT / authority_path)}

        bounds = []
        dataset_doc = domain["doc"]["dataset"]
        for t in domain["doc"]["trajectories"]:
            first, last, count = t["origin_first"], t["origin_last"], t["origin_count"]
            bounds.append({"trajectory_index": t["trajectory_index"],
                           "split_start": dataset_doc["split_bounds_start_inclusive"],
                           "split_stop": dataset_doc["split_bounds_end_exclusive"],
                           "first_origin": first, "last_origin": last, "origin_count": count})
        plan = {
            "schema_version": 4, "endpoint_id": endpoint_id, "claim": row["claim"],
            "outcome_type": outcome, "seed_ids": [0, 1, 2, 3, 4],
            "required_horizons": horizons, "history": history, "max_horizon": hmax,
            "block_length": None, "acf_dependence_length": None, "acf_evidence": acf,
            "epsilon_evidence": epsilon, "slice": row["slice"],
            "domain": {"dataset_id": domain["dataset_id"], "split": "test",
                       "path": domain["path"], "sha256": domain["sha256"], "bounds": bounds},
            "dataset_collection": domain["collection"], "producer_sources": producer_sources,
            "left": seed_sides[0], "right": seed_sides[1], "switch_indices": None,
            "execution_binding_path": f"runs/primary-endpoint-v1/execution-bindings/{endpoint_id}.json",
            "reporting_switch_authority": authority,
            "profile_device": row.get("device") if outcome == "paired_window_cost_superiority" else None,
            "global_test_campaign": {
                "queue_path": "protocols/global-frozen-test-job-queue-v1.json",
                "completion_path": "protocols/global-frozen-test-completion-v1.json",
            },
        }
        plan_path = OUT / "plans" / f"{endpoint_id}.json"
        write_exclusive(plan_path, plan)
        plan_inventory.append({"endpoint_id": endpoint_id, "claim": row["claim"],
                               "path": rel(plan_path), "sha256": sha(plan_path),
                               "outcome_type": outcome, "domain_id": domain["dataset_id"],
                               "history": history, "max_horizon": hmax,
                               "trajectory_count": len(bounds),
                               "full_test_origins": sum(x["origin_count"] for x in bounds),
                               "acf_status": "deferred-validation-derived",
                               "epsilon_status": ("not-applicable" if epsilon is None else
                                   "unidentifiable-by-rule" if epsilon["mode"] == "unidentifiable_by_rule" else "deferred-validation-derived"),
                               "device": plan["profile_device"]})

    if len(plan_inventory) != 49 or {claim: sum(x["claim"] == claim for x in plan_inventory)
                                     for claim in ("H1", "H2", "H3")} != {"H1": 9, "H2": 16, "H3": 24}:
        raise RuntimeError("primary plan count/family partition mismatch")
    capacity_review = ROOT / "reviews/graph-campaign-capacity-correction-addendum-v1.md"
    inventory = {
        "schema_version": 1, "artifact_status": "preparation-only-not-root-accepted-not-frozen-not-executed",
        "scope": "schema4 primary plan and metadata-only test domain preparation; no launch bindings, test arrays, oracle values, or outcomes",
        "root_preparation_registry": {"path": rel(REGISTRY), "sha256": sha(REGISTRY)},
        "root_operational_definitions": {"path": "protocols/primary-endpoint-operational-definitions-root-v1.md",
                                         "sha256": sha(ROOT / "protocols/primary-endpoint-operational-definitions-root-v1.md")},
        "candidate_catalog": {"path": rel(CATALOG), "sha256": sha(CATALOG), "status": catalog["status"],
                              "pool_count": len(pools)},
        "generator": {"path": rel(Path(__file__)), "sha256": sha(Path(__file__))},
        "metadata_preflight_source": {"path": "edsgn/staging/c06_domain_manifest_preflight_v4.py",
                                      "sha256": sha(ROOT / "edsgn/staging/c06_domain_manifest_preflight_v4.py")},
        "capacity_review": {"path": rel(capacity_review), "sha256": sha(capacity_review),
                            "h1_h2_planned_invocations": 130,
                            "components": {"H1_graph": 30, "H1_forecast_NI": 40,
                                           "H2_ordinary_V4_V3": 40, "H2_matched_replay": 20}},
        "plan_source_hashes": {path: sha(ROOT / path) for path in sorted(set(
            source["path"] for plan in plan_inventory for source in load_json(ROOT / plan["path"])["producer_sources"]))},
        "ordered_plan_sha256_manifest": hashlib.sha256(
            "\n".join(x["sha256"] for x in plan_inventory).encode("ascii")).hexdigest(),
        "future_global_queue_path": "protocols/global-frozen-test-job-queue-v1.json",
        "future_global_completion_path": "protocols/global-frozen-test-completion-v1.json",
        "plan_count": 49, "family_counts": {c: sum(x["claim"] == c for x in plan_inventory) for c in ("H1", "H2", "H3")},
        "method_input_rule": "H3 V7/V5/V6 reference the dataset's V4 candidate pool; post-training selection and byte-identical execution checkpoint copies remain pending root validation.",
        "domains": domain_inventory,
        "plans": plan_inventory,
        "execution_constraints": [
            "No candidate selection, launch binding, checkpoint bytes, policy receipt, epsilon value, or ACF/block result has been created.",
            "H3 V7/V5/V6 use V4 candidate pools as same-weight policy comparisons; future selected checkpoint copies and freezes must be byte-verified before binding.",
            "H3 email accuracy margin is retained as unidentifiable by the validation all-zero rule; it is not silently dropped.",
            "Global test queue and completion are future path references only; no hashes or completion claim are asserted.",
            "Test archives were byte-hashed as inventory but forecast/target/oracle payload members were not decoded; C06 preflight decoded only metadata_json, node_ids and input_mask and read the values NPY shape header.",
        ],
    }
    write_exclusive(OUT / "preparation-inventory.json", inventory)


if __name__ == "__main__":
    main()
