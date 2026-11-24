"""Prepare immutable logical secondary replay rows; never launch or load arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "protocols/secondary-replay-rows-v2"
METHODS = [f"V{i}" for i in range(10)] + ["GRU", "DGCRN", "SeqSNN"]
DOMAINS = ["synthetic", "email", "mooc", "email-episode"]
MAIN_DOMAINS = ["synthetic", "email", "mooc"]
SCENARIOS = {
    "identity30-main": "data/processed/synthetic-n64-identity-change0.3-miss0",
    "noise-only": "data/processed/synthetic-stress-n64-noise-only",
    "identity10": "data/processed/synthetic-stress-n64-identity10",
    "weight10": "data/processed/synthetic-stress-n64-weight10",
    "weight30": "data/processed/synthetic-stress-n64-weight30",
    "gradual-weight128-fixed-support": "data/processed/synthetic-stress-n64-gradual-weight128-fixed-support",
    "local-shock10-amp0.5-duration8": "data/processed/synthetic-stress-n64-local-shock",
    "nonlinear-identity30": "data/processed/synthetic-stress-n64-nonlinear",
}
SCENARIO_DESCRIPTORS = {
    "identity30-main": "n64_identity_change30_main_reuse",
    "noise-only": "n64_noise_only",
    "identity10": "n64_identity_change10",
    "weight10": "n64_weight_change10",
    "weight30": "n64_weight_change30",
    "gradual-weight128-fixed-support": "n64_gradual_weight128_fixed_support",
    "local-shock10-amp0.5-duration8": "n64_local_shock10_amp0.5_duration8",
    "nonlinear-identity30": "n64_nonlinear_identity30",
}
DATA_PATHS = {
    "synthetic": SCENARIOS["identity30-main"],
    "email": "data/processed/email/email.npz",
    "mooc": "data/processed/mooc/mooc.npz",
    "email-episode": "data/processed/email-episode-v1/email-episode.npz",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: str | None):
    if not path:
        return None
    p = ROOT / path
    return {"path": path, "sha256": sha(p)} if p.is_file() else None


def config_ref(dataset: str, method: str):
    """Return an existing config candidate, without claiming selection/admission."""
    candidates = []
    if method.startswith("V"):
        candidates += [f"configs/core-development-bound-v3/{dataset}-{method}-seed0.json"]
        if method in {"V0", "V1"}:
            candidates += [f"configs/core-formal-pending-v1/{dataset}-{method}-seed0.json"]
            candidates += [f"configs/p3-internal-bound-v4/{dataset}-{method}-seed0.json"]
        if method in {"V2", "V3", "V4", "V9"}:
            candidates += [f"configs/core-development-v1/{dataset}-{method}-seed0.json"]
    else:
        candidates += [f"configs/baselines-bound-v3/{dataset}-{method}-seed0.json"]
        candidates += [f"configs/p3-internal-bound-v4/{dataset}-{method}-seed0.json"]
        candidates += [f"configs/p3-strong-replication-v1/{dataset}-{method}-seed0.json"]
    for candidate in candidates:
        found = ref(candidate)
        if found:
            found["role"] = "existing_config_candidate_only"
            return found
    return None


def domain_ref(dataset: str, scenario: str | None = None):
    if dataset == "synthetic" and scenario in SCENARIO_DESCRIPTORS:
        stem = SCENARIO_DESCRIPTORS[scenario]
        path = f"protocols/synthetic-stress-domain-preparation-v1/{stem}-test-domain.json"
        return ref(path)
    return ref(f"protocols/c06-domain-preflight-root-v2/{dataset}-test-domain.json")


def domain_origin_count(domain_reference):
    if not domain_reference:
        return None
    path = ROOT / domain_reference["path"]
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    trajectories = doc.get("trajectories")
    if not isinstance(trajectories, list):
        return None
    counts = [row.get("origin_count") for row in trajectories]
    if any(type(count) is not int or count < 0 for count in counts):
        return None
    return sum(counts)


def checkpoint_source_method(method: str) -> str:
    if method in {"V5", "V6", "V7"}:
        return "V4"
    if method == "V8":
        return "V2"
    return method


def _grid_slug(grid):
    if grid is None:
        return None
    if "graph_fraction" in grid:
        gf = str(grid["graph_fraction"]).replace(".", "p")
        sf = str(grid["state_fraction"]).replace(".", "p")
        if "max_graph_age" in grid:
            return f"g{gf}-s{sf}-ag{grid['max_graph_age']}-as{grid['max_state_age']}-thr{grid['thresholds']}-seed{grid['recipe_seed']}"
        return f"g{gf}-s{sf}-age{grid['max_age']}"
    return "grid"


def run_row(category, dataset, method, *, recipe=None, scenario=None, control=None,
            action="prediction", device=None, policy=None, grid=None, template=None,
            checkpoint_method=None, extra=None):
    ident = "sec-" + "-".join(str(x) for x in (category, dataset, scenario, recipe, control,
                                                  method, grid, device, action) if x is not None)
    base = f"runs/secondary-replay-v1/{category}/{dataset}"
    path_pieces = [x for x in (scenario, recipe, control, method, _grid_slug(grid), device, action) if x is not None]
    output = base + "/" + "/".join(str(x) for x in path_pieces) + "/seed0"
    source_method = checkpoint_method or checkpoint_source_method(method) if method else None
    c_ref = config_ref(dataset, source_method) if source_method else None
    domain_reference = domain_ref(dataset, scenario)
    row = {
        "logical_id": ident,
        "category": category,
        "dataset": dataset,
        "scenario_id": scenario,
        "method_id": method,
        "model_seed": 0,
        "recipe_id": recipe,
        "recipe_sha256": None,
        "control_id": control,
        "grid": grid,
        "action": action,
        "device": device,
        "policy_id": policy,
        "policy_config_ref": None,
        "config_candidate_ref": c_ref,
        "checkpoint_source_method_id": source_method,
        "checkpoint_sha256": None,
        "freeze_sha256": None,
        "training_binding_sha256": None,
        "dataset_path": DATA_PATHS.get(dataset),
        "domain_manifest_ref": domain_reference,
        "expected_origin_count_from_metadata": domain_origin_count(domain_reference),
        "source_bundle_sha256": None,
        "output_path_prospective": output,
        "status": "planning_only_unadmitted_unexecuted",
        "gate_accepted": False,
        "test_accessed": False,
        "extra": extra or {},
    }
    if template:
        row["template_ref"] = ref(template)
    return row


def main():
    inventory_path = OUT / "logical-secondary-replay-inventory-v2.json"
    if inventory_path.exists():
        raise FileExistsError(f"refusing to overwrite prepared inventory: {inventory_path}")
    sys.path.insert(0, str(ROOT / "edsgn/src"))
    from edsgn.robustness import canonical_recipe, recipe_digest, standard_missingness_recipes

    OUT.mkdir(parents=True, exist_ok=True)
    recipes = {}
    for i, raw in enumerate(standard_missingness_recipes(seed=2027)):
        recipe = canonical_recipe(raw)
        key = f"missing-r{recipe['missing']['rate']:g}-{recipe['missing']['kind']}"
        recipes[key] = recipe
    recipes["count-noise1-r0-random"] = canonical_recipe({
        "schema_version": 1, "seed": 2027,
        "missing": {"rate": 0, "kind": "random"},
        "count_noise_max_abs": 1, "event_time_jitter_bins": 0})
    recipes["event-jitter1-r0-random"] = canonical_recipe({
        "schema_version": 1, "seed": 2027,
        "missing": {"rate": 0, "kind": "random"},
        "count_noise_max_abs": 0, "event_time_jitter_bins": 1})
    recipe_refs = {}
    for name, recipe in recipes.items():
        path = OUT / "recipes" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(recipe, sort_keys=True, indent=2) + "\n").encode("utf-8")
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError(f"existing recipe differs from frozen canonical recipe: {path}")
        else:
            with path.open("xb") as f:
                f.write(payload)
        recipe_refs[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "canonical_recipe_sha256": recipe_digest(recipe),
            "recipe": recipe,
        }

    rows = []
    # Standard 12-recipe robustness set: 12 × 4 domains × 13 methods.
    for recipe_id in list(recipes)[:12]:
        for dataset in DOMAINS:
            for method in METHODS:
                rows.append(run_row("missingness", dataset, method, recipe=recipe_id,
                    policy="frozen-clean-selected-policy-required", extra={
                        "replay": "C06", "batch_size": 1, "device": "cuda:0",
                        "precision": "fp32", "full_origin_replay": True,
                        "clean_targets_preserved": True,
                        "rate0_reuse": "not_assumed; needs exact authorization/contract/artifact proof"}))
    # Separate integer count noise and event-time jitter: 2 × 3 real count domains × 13.
    for recipe_id in ("count-noise1-r0-random", "event-jitter1-r0-random"):
        for dataset in ("email", "mooc", "email-episode"):
            for method in METHODS:
                rows.append(run_row("count-or-event-perturbation", dataset, method,
                    recipe=recipe_id, policy="frozen-clean-selected-policy-required", extra={
                        "replay": "C06", "batch_size": 1, "device": "cuda:0",
                        "precision": "fp32", "full_origin_replay": True,
                        "clean_targets_preserved": True,
                        "count_noise_signed_data": False}))
    # Eight N64 scenarios × 13 method contexts. These are contexts, not claimed reuses.
    for scenario, dataset_path in SCENARIOS.items():
        for method in METHODS:
            row = run_row("synthetic-clean-scenario", "synthetic", method,
                          scenario=scenario, policy="clean-input-policy-unresolved",
                          extra={"dataset_path": dataset_path, "full_origin_replay": True,
                                 "overlap_reuse": "requires exact checkpoint/policy/domain/source/origin/output proof"})
            if scenario == "identity30-main":
                row["config_candidate_ref"] = config_ref("synthetic", checkpoint_source_method(method))
            rows.append(row)
    # Four root-prepared controls × 3 main domains × 5 seeds, all unique identities.
    control_doc = json.loads((ROOT / "protocols/trained-state-control-matrix-root-preparation-v2.json").read_text(encoding="utf-8"))
    for item in control_doc["rows"]:
        dataset, control, seed = item["dataset"], item["control_id"], int(item["seed"])
        row = run_row("trained-state-control", dataset, "V4-control", control=control,
                      policy="full-clean-input", template=item.get("template_path"),
                      extra={"model_seed": seed, "planned_config_path": item.get("planned_config_path"),
                             "planned_output_dir": item.get("planned_output_dir"),
                             "source_matrix_status": item.get("status"),
                             "independently_trained_control": True})
        row["model_seed"] = seed
        row["logical_id"] = f"sec-trained-state-control-{dataset}-{control}-seed{seed}"
        row["output_path_prospective"] = item.get("planned_output_dir")
        rows.append(row)
    # Two count-loss predictions; MSE reference is explicit but no test reuse is asserted.
    countloss = json.loads((ROOT / "reviews/count-loss-config-blueprints-root-preparation-v3.json").read_text(encoding="utf-8"))
    for item in countloss["rows"]:
        dataset = item["dataset"]
        rows.append(run_row("count-loss-poisson", dataset, "V4-Poisson", control="masked-poisson",
            policy="full-clean-input", template=item["config"]["path"], extra={
                "blueprint_ref": item["blueprint"],
                "mse_reference_config": item.get("mse_reference_config"),
                "mse_reference_exact_reuse": "requires exact freeze/input/domain/origin proof",
                "root_preparation_acceptance_ref": "reviews/count-loss-config-blueprints-root-preparation-v3.json"}))
    # N256: six prediction policies and distinct CPU/CUDA full B1 profiles.
    n256_methods = ["V2", "V4", "V8", "V7", "V5", "V6"]
    for method in n256_methods:
        for action in ("prediction", "profile"):
            devices = ("cpu", "cuda:0") if action == "profile" else (None,)
            for device in devices:
                rows.append(run_row("synthetic-n256-clean", "synthetic", method,
                    scenario="identity30-n256", action=action, device=device,
                    policy="method-specific-validation-calibrated-policy-required",
                    extra={"dataset_path": "data/processed/synthetic-stress-n256-identity30-scale",
                           "batch_size": 1, "precision": "fp32" if action == "profile" else None,
                           "warmup_windows": 10 if action == "profile" else None,
                           "timed_repeats": 5 if action == "profile" else None,
                           "full_origin_replay": True,
                           "n64_checkpoint_reuse": False}))
    # Quota grid: 16 G/S fraction pairs × 3 ages × 2 datasets × 4 methods × 2 devices,
    # with a separate prediction and profile logical row per combination.
    methods = ["V7", "V8", "V5", "V6"]
    fractions = [0.25, 0.5, 0.75, 1.0]
    for dataset in ("email", "mooc"):
        for gf in fractions:
            for sf in fractions:
                for age in (2, 4, 8):
                    grid = {"graph_fraction": gf, "state_fraction": sf, "max_age": age}
                    for method in methods:
                        for device in ("cpu", "cuda:0"):
                            policy = f"quota-g{gf:g}-s{sf:g}-age{age}-{device}"
                            for action in ("prediction", "profile"):
                                rows.append(run_row("quota-age-grid", dataset, method,
                                    control="independent-quota-policy", grid=grid,
                                    action=action, device=device, policy=policy,
                                    extra={"batch_size": 1, "precision": "fp32",
                                           "warmup_windows": 10 if action == "profile" else None,
                                           "timed_repeats": 5 if action == "profile" else None,
                                           "full_origin_replay": True,
                                           "device_specific_policy_calibration": True,
                                           "cross_device_reuse_assumed": False}))
    # Explicit graph-only periodic reference contexts; root design does not settle
    # a model/checkpoint or quota/age tuple, so those fields remain null.
    for scenario, dataset_path in SCENARIOS.items():
        rows.append(run_row("graph-only-periodic-reference", "synthetic", None,
            scenario=scenario, control="graph-only-periodic", action="graph-diagnostic",
            policy="periodic-graph-only", grid={"graph_fraction": 0.25,
                "state_fraction": 1.0, "max_graph_age": 4, "max_state_age": 1,
                "thresholds": 0, "recipe_seed": 0}, extra={"dataset_path": dataset_path,
                               "model_method_id_unresolved": True,
                               "blind_cost_helper": "reviews/blind-reference-cost-policy-root-acceptance-v1.json",
                               "cost_table_and_budget_authority_pending": True,
                               "reference_job_required": True}))
    # Explicit unresolved simple-baseline clean rows and frozen causal case arms.
    for dataset in DOMAINS:
        for method in ("persistence", "linear"):
            rows.append(run_row("simple-baseline-clean", dataset, method,
                policy="clean-full-origin", extra={
                    "driver_route": None,
                    "route_status": "unfinished_guarded_no_neural_checkpoint_route",
                    "training_fit_only": True}))
    for dataset in DOMAINS:
        for arm in ("relationship-removal", "retention-removal"):
            rows.append(run_row("case-signal-sensitivity", dataset, "V4", control=arm,
                policy="frozen-clean-full-origin", extra={
                    "selection": "focal node training activity; neighbor selected from full validation V4 replay",
                    "selected_node_neighbor_ids": None,
                    "driver_source_sha256": None,
                    "controller_route": None,
                    "pretest_selection_binding": None,
                    "selection_status": "pending_validation_only_selection_and_frozen_driver",
                    "full_legal_origins": True}))

    ids = [r["logical_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("logical row ID collision")
    category_counts = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
        if row["recipe_id"]:
            row["recipe_sha256"] = recipe_refs[row["recipe_id"]]["canonical_recipe_sha256"]
    expected = {
        "missingness": 624,
        "count-or-event-perturbation": 78,
        "synthetic-clean-scenario": 104,
        "trained-state-control": 60,
        "count-loss-poisson": 2,
        "synthetic-n256-clean": 18,
        "quota-age-grid": 1536,
        "graph-only-periodic-reference": 8,
        "simple-baseline-clean": 8,
        "case-signal-sensitivity": 8,
    }
    if category_counts != expected:
        raise RuntimeError(f"category count mismatch: {category_counts}")
    design_paths = [
        "protocols/secondary-replay-rows-root-design-v1.md",
        "protocols/secondary-scope-and-count-controls-root-design-v1.md",
        "protocols/robustness-recipes.md",
        "edsgn/src/edsgn/robustness.py",
        "protocols/trained-state-control-matrix-root-preparation-v2.json",
        "reviews/count-loss-config-blueprints-root-preparation-v3.json",
        "protocols/c06-domain-preflight-root-v2/aggregate-report.json",
        "protocols/case-slices-and-signal-sensitivity-root-design-v1.md",
        "reviews/blind-reference-cost-policy-root-acceptance-v1.json",
    ]
    inputs = {p: sha(ROOT / p) for p in design_paths if (ROOT / p).is_file()}
    doc = {
        "schema_version": 2,
        "title": "prospective logical secondary replay inventory",
        "supersedes_preparation_attempt": "protocols/secondary-replay-rows-v1/logical-secondary-replay-inventory-v1.json",
        "supersession_reason": "v1 was an initial generator attempt whose quota output directory components used a dictionary representation; v2 emits portable stable path slugs and adds case-slice/signal-sensitivity requirements.",
        "scope": "planning_only_not_p3a_admission_not_launch_not_executed",
        "status": "prepared_unadmitted_unexecuted",
        "accepted_by_root": None,
        "pipeline_advanced": False,
        "test_accessed": False,
        "arrays_opened_or_hashed": False,
        "models_or_checkpoints_loaded": False,
        "optimizer_or_gpu_used": False,
        "model_seed": 0,
        "methods": METHODS,
        "domains": DOMAINS,
        "recipe_catalog": recipe_refs,
        "reporting_requirements": {
            "case_slice_source": "protocols/case-slices-and-signal-sensitivity-root-design-v1.md",
            "per_domain_slices": ["stable-input", "high-change-input", "worst-V4-raw-MAE", "long-inactive-gap", "incomplete-history"],
            "stable_high_thresholds": "training-history-only linear 25th/75th percentiles; first qualifying full L-origin block in ordered trajectory/origin order",
            "worst_case": "largest V4 seed0 required-offset pooled raw MAE; ties earliest trajectory/block; display only after global completion",
            "input_score": "history-only adjacent raw observation absolute difference divided by frozen train robust scale",
            "case_arms": ["relationship-removal", "retention-removal"],
            "case_arm_selection": "focal node by train activity; neighbor by V4 full validation mean valid neighbor weight; both pending",
            "no_posthoc_primary_selection": True,
        },
        "design_input_sha256": inputs,
        "category_counts": category_counts,
        "logical_row_count": len(rows),
        "rows": rows,
        "unresolved_execution_route_gaps": [
            "Persistence and pooled closed-form linear rows are not covered by this 13-method allocation and still need a guarded no-neural-checkpoint driver/config/output route.",
            "Quota rows have scientific fractions/ages from the root decision, but exact per-device cost tables, frozen policy configs/checkpoints and accepted full-origin outputs are not available; each row remains unadmitted.",
            "Graph-only periodic references are eight scenario-level placeholders because the design does not name a model/checkpoint or periodic quota/age tuple.",
            "SimpleBaselines and case-ablation execution routes remain separate unfinished items; this inventory does not imply their coverage.",
            "No row is selected by g00/m00 or any scheduler; candidate and policy authorities remain explicit null until root-approved.",
        ],
    }
    out_path = inventory_path
    with out_path.open("xb") as f:
        f.write((json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"path": out_path.relative_to(ROOT).as_posix(), "sha256": sha(out_path),
                      "rows": len(rows), "category_counts": category_counts,
                      "recipe_count": len(recipes), "status": doc["status"]}, indent=2))


if __name__ == "__main__":
    main()
