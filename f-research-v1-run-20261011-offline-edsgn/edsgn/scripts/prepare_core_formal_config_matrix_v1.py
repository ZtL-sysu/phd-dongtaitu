"""Prepare the 90-row core formal configuration matrix; never selects or launches."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V9")
DATASETS = ("synthetic", "email", "mooc")
SEEDS = (0, 1, 2, 3, 4)
CONCRETE_VARIANTS = {"V0", "V1"}
CONFIG_ROOT = ROOT / "configs/core-formal-pending-v1"
MATRIX_PATH = ROOT / "protocols/core-formal-config-matrix-preparation-v1.json"
CONTRACT_PATH = ROOT / "protocols/p3-training-resource-contract-v2.json"
ALLOCATION_PATH = ROOT / "protocols/p3-core-default-training-allocation-v1.json"
DECISION_PATH = ROOT / "protocols/formal-comparator-and-secondary-matrix-root-decision-v1.md"
FRESH_QUEUE = ROOT / "protocols/p3-fresh-remaining-bound-queue-v1.json"
FRESH_ACCEPTANCE = ROOT / "protocols/p3-fresh-remaining-bound-root-acceptance-v1.json"
INTERNAL_QUEUE = ROOT / "protocols/p3-internal-bound-queue-v4.json"
INTERNAL_ACCEPTANCE = ROOT / "protocols/p3-internal-bound-queue-root-acceptance-v4.json"
PILOT_QUEUE = ROOT / "protocols/p4-core-pilot-queue-v3.json"
PILOT_ACCEPTANCE = ROOT / "protocols/p4-core-pilot-queue-root-acceptance-v3.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def accepted_templates():
    sources = [
        (FRESH_QUEUE, FRESH_ACCEPTANCE, {"synthetic"}),
        (INTERNAL_QUEUE, INTERNAL_ACCEPTANCE, {"email", "mooc"}),
    ]
    templates = {}
    accepted_refs = []
    for queue_path, acceptance_path, wanted_datasets in sources:
        queue = read(queue_path)
        acceptance = read(acceptance_path)
        if acceptance.get("accepted") is not True and acceptance.get("verdict") != "pass":
            raise RuntimeError(f"P3 seed0 template queue is not root accepted: {acceptance_path}")
        if acceptance.get("queue_sha256") != sha(queue_path):
            raise RuntimeError(f"accepted seed0 queue hash mismatch: {queue_path}")
        pins = acceptance.get("file_inputs_sha256", {})
        if not isinstance(pins, dict):
            raise RuntimeError(f"accepted input pins missing: {acceptance_path}")
        for rel, expected in pins.items():
            if sha(ROOT / rel) != expected:
                raise RuntimeError(f"accepted P3 preparation pin changed: {rel}")
        for job in queue["jobs"]:
            config_path = Path(job.get("config_path") or job.get("config") or job["args"][job["args"].index("--config") + 1])
            config = read(config_path)
            dataset = "synthetic" if "synthetic" in config_path.name else (
                "email" if "email" in config_path.name and "episode" not in config_path.name else
                "mooc" if "mooc" in config_path.name else "email-episode"
            )
            variant = config.get("variant")
            if dataset not in wanted_datasets or variant not in CONCRETE_VARIANTS:
                continue
            key = (dataset, variant)
            if key in templates or config.get("seed") != 0:
                raise RuntimeError(f"duplicate/non-seed0 accepted template for {key}")
            rel_config = config_path.resolve().relative_to(ROOT).as_posix()
            pinned = pins.get(rel_config)
            for accepted_job in acceptance.get("jobs", []):
                accepted_config = accepted_job.get("config_path", accepted_job.get("config"))
                if accepted_config and Path(accepted_config).resolve() == config_path.resolve():
                    if accepted_job.get("config_sha256") != sha(config_path):
                        raise RuntimeError(f"accepted job config hash changed: {config_path}")
                    pinned = accepted_job["config_sha256"]
                    break
            if pinned != sha(config_path):
                raise RuntimeError(f"accepted root config pin changed: {config_path}")
            templates[key] = {"config_path": config_path, "config": config, "queue_path": queue_path,
                              "queue_sha256": sha(queue_path), "acceptance_path": acceptance_path,
                              "acceptance_sha256": sha(acceptance_path), "job_id": job["id"]}
        accepted_refs.append({"queue": str(queue_path.relative_to(ROOT)).replace("\\", "/"),
                              "queue_sha256": sha(queue_path),
                              "root_acceptance": str(acceptance_path.relative_to(ROOT)).replace("\\", "/"),
                              "root_acceptance_sha256": sha(acceptance_path),
                              "pinned_file_count": len(pins)})
    expected = {(d, v) for d in DATASETS for v in CONCRETE_VARIANTS}
    if set(templates) != expected:
        raise RuntimeError(f"accepted P3 templates incomplete: missing {sorted(expected-set(templates))}")
    return templates, accepted_refs


def accepted_candidate_templates():
    queue = read(PILOT_QUEUE)
    acceptance = read(PILOT_ACCEPTANCE)
    if (acceptance.get("accepted_by") != "root" or acceptance.get("accepted") is not True
            or acceptance.get("verdict") != "pass"
            or acceptance.get("scope") != "prepared-pilot-configuration-identity-only"
            or acceptance.get("queue_sha256") != sha(PILOT_QUEUE)
            or acceptance.get("training_started") is not False
            or acceptance.get("actual_launch_authorized_by_this_record") is not False
            or acceptance.get("dataset_arrays_loaded_or_hashed") is not False):
        raise RuntimeError("P4 pilot root acceptance is not an identity-only prepared record")
    accepted_jobs = {(row["dataset"], row["variant"]): row for row in acceptance.get("jobs", [])}
    templates = {}
    for job in queue.get("jobs", []):
        key = (job.get("dataset"), job.get("variant"))
        if key[0] in DATASETS and key[1] in {"V2", "V3", "V4", "V9"}:
            path = Path(job["config_path"])
            if key in templates or sha(path) != job.get("config_sha256"):
                raise RuntimeError(f"P4 candidate parameter template mismatch: {path}")
            accepted = accepted_jobs.get(key)
            if not accepted or accepted.get("config_sha256") != sha(path):
                raise RuntimeError(f"P4 root identity record does not pin candidate template: {path}")
            templates[key] = path
    expected = {(d, v) for d in DATASETS for v in ("V2", "V3", "V4", "V9")}
    if set(templates) != expected:
        raise RuntimeError("P4 pilot source does not provide all 12 primary candidate parameter templates")
    return templates, {"queue_path": str(PILOT_QUEUE.relative_to(ROOT)).replace("\\", "/"),
                       "queue_sha256": sha(PILOT_QUEUE),
                       "root_acceptance_path": str(PILOT_ACCEPTANCE.relative_to(ROOT)).replace("\\", "/"),
                       "root_acceptance_sha256": sha(PILOT_ACCEPTANCE),
                       "root_acceptance_scope": acceptance["scope"],
                       "root_acceptance_training_started": False,
                       "selected_candidate_ids": None}


def prepare():
    if (CONFIG_ROOT.exists() and any(CONFIG_ROOT.iterdir())) or MATRIX_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing formal config matrix/config directory")
    contract = read(CONTRACT_PATH)
    if contract.get("status") != "root-frozen-before-baseline-optimizer-launch":
        raise RuntimeError("training resource contract is not the accepted frozen contract")
    allocation = read(ALLOCATION_PATH)
    if allocation.get("status") != "root-frozen-before-core-seed0-optimizer-launch":
        raise RuntimeError("core default allocation is not the accepted frozen allocation")
    decision_text = DECISION_PATH.read_text(encoding="utf-8-sig")
    for phrase in ("V0/V1/V2/V3/V4/V9", "model seeds0–4", "g00/m00 is still the accepted default development pilot identity"):
        if phrase not in decision_text:
            raise RuntimeError(f"root formal decision missing expected scope statement: {phrase}")
    templates, accepted_refs = accepted_templates()
    candidate_templates, candidate_template_source = accepted_candidate_templates()
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    concrete_count = 0
    try:
        for variant in VARIANTS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    output = ROOT / "runs/core-formal-pending-v1" / dataset / variant / f"seed{seed}"
                    if output.exists():
                        raise FileExistsError(f"formal output path already exists: {output}")
                    row = {
                        "row_id": f"core-formal-{dataset}-{variant}-seed{seed}",
                        "dataset": dataset, "variant": variant, "model_seed": seed,
                        "primary_scope": "formal core primary regime",
                        "configuration_status": "concrete" if variant in CONCRETE_VARIANTS else "unresolved_candidate_selection",
                        "config_path": str((CONFIG_ROOT / f"{dataset}-{variant}-seed{seed}.json").relative_to(ROOT)).replace("\\", "/")
                            if variant in CONCRETE_VARIANTS else
                            str((CONFIG_ROOT / f"{dataset}-{variant}-seed{seed}.json").relative_to(ROOT)).replace("\\", "/"),
                        "config_sha256": None,
                        "template_config_path": None,
                        "template_config_sha256": None,
                        "output_dir": str(output),
                        "training_input_binding_sha256": None,
                        "training_input_binding_status": "pending-at-launch",
                        "launch_status": "planning_only",
                        "test_access_authorized": False,
                        "seed0_reuse_status": "requires exact immutable source/data/config/receipt/full-train-validation selection match and root acceptance; no reuse approved here",
                        "candidate_selection": {
                            "graph_candidate_id": None if variant not in CONCRETE_VARIANTS else "not-applicable-to-control-template",
                            "memory_candidate_id": None if variant not in CONCRETE_VARIANTS else "not-applicable-to-control-template",
                            "selection_status": "pending-post-pilot-formal-decision" if variant not in CONCRETE_VARIANTS else "fixed-by-accepted-control-template",
                            "default_g00_m00_selected": False,
                        },
                    }
                    if variant in CONCRETE_VARIANTS:
                        ref = templates[(dataset, variant)]
                        template_path, template = ref["config_path"], ref["config"]
                        if template.get("epochs") != allocation["max_epochs"] or template.get("patience") != allocation["early_stopping_patience"]:
                            raise RuntimeError(f"control template cap differs from root contract: {template_path}")
                        if template.get("batch_size") != contract["batch_sizes"][variant][dataset]:
                            raise RuntimeError(f"control template batch differs from root contract: {template_path}")
                        if template.get("device") != contract["devices"][variant][dataset] or template.get("threads") != allocation["threads"]:
                            raise RuntimeError(f"control template device/thread settings differ from root contract: {template_path}")
                        cfg = deepcopy(template)
                        cfg["seed"] = seed
                        cfg["output_dir"] = str(output)
                        cfg_path = ROOT / row["config_path"]
                        with cfg_path.open("x", encoding="utf-8", newline="\n") as stream:
                            json.dump(cfg, stream, ensure_ascii=False, indent=2)
                            stream.write("\n")
                        if {k: v for k, v in cfg.items() if k not in {"seed", "output_dir"}} != {
                            k: v for k, v in template.items() if k not in {"seed", "output_dir"}
                        }:
                            raise RuntimeError(f"control config changed beyond seed/output: {cfg_path}")
                        row.update(config_sha256=sha(cfg_path),
                                   template_config_path=str(template_path.relative_to(ROOT)).replace("\\", "/"),
                                   template_config_sha256=sha(template_path),
                                   training_parameters={k: cfg[k] for k in ("variant", "device", "threads", "batch_size", "lr", "weight_decay", "grad_clip", "epochs", "patience", "model") if k in cfg},
                                   source_queue_path=str(ref["queue_path"].relative_to(ROOT)).replace("\\", "/"),
                                   source_queue_sha256=ref["queue_sha256"],
                                   source_root_acceptance_path=str(ref["acceptance_path"].relative_to(ROOT)).replace("\\", "/"),
                                   source_root_acceptance_sha256=ref["acceptance_sha256"],
                                   config_diff_from_accepted_seed0=["seed", "output_dir"])
                        concrete_count += 1
                    else:
                        template_path = candidate_templates[(dataset, variant)]
                        template = read(template_path)
                        row.update(template_config_path=str(template_path.relative_to(ROOT)).replace("\\", "/"),
                                   template_config_sha256=sha(template_path),
                                   template_role="accepted P4 pilot config source for parameter-shape reference only; not a formal candidate selection",
                                   template_source_queue_path=candidate_template_source["queue_path"],
                                   template_source_queue_sha256=candidate_template_source["queue_sha256"],
                                   template_source_root_acceptance_path=candidate_template_source["root_acceptance_path"],
                                   template_source_root_acceptance_sha256=candidate_template_source["root_acceptance_sha256"],
                                   training_parameters={k: template[k] for k in ("variant", "device", "threads", "batch_size", "lr", "weight_decay", "grad_clip", "epochs", "patience", "model") if k in template},
                                   required_selection_placeholders={
                                       "graph_candidate_id": "required-root-selected-candidate-id; do not use g00 by implication",
                                       "memory_candidate_id": "required-root-selected-candidate-id; do not use m00 by implication",
                                       "selected_graph_parameter_values": None,
                                       "selected_memory_parameter_values": None,
                                       "root_selection_record_path": None,
                                       "root_selection_record_sha256": None,
                                       "selected_checkpoint_source_sha256": None,
                                       "selected_freeze_sha256": None,
                                   })
                    rows.append(row)
        if len(rows) != 90 or concrete_count != 30:
            raise RuntimeError(f"expected exactly 90 matrix rows and 30 concrete control configs; got {len(rows)}, {concrete_count}")
        matrix = {
            "schema_version": 1,
            "status": "planning_only",
            "scope": "P3/P4 formal core configuration matrix preparation; no execution authorization or scientific selection",
            "generator": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
                          "sha256": sha(Path(__file__).resolve()), "arrays_read": False, "training_started": False},
            "required_rows": 90,
            "row_dimensions": {"variants": list(VARIANTS), "datasets": list(DATASETS), "model_seeds": list(SEEDS)},
            "root_decision": {"path": str(DECISION_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(DECISION_PATH),
                              "scope": "V0/V1/V2/V3/V4/V9 × synthetic/email/MOOC × seeds0–4; V5/6/7 reuse same-seed V4 weights, V8 same-seed V2 weights"},
            "resource_contract": {"path": str(CONTRACT_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(CONTRACT_PATH),
                                  "status": contract["status"], "batch_sizes": contract["batch_sizes"],
                                  "devices": contract["devices"], "training_prior_cache": contract["training_prior_cache"],
                                  "training_prior_cache_bytes": contract["training_prior_cache_bytes"]},
            "core_allocation": {"path": str(ALLOCATION_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(ALLOCATION_PATH),
                                "status": allocation["status"], "max_epochs": allocation["max_epochs"],
                                "patience": allocation["early_stopping_patience"], "threads": allocation["threads"],
                                "optimizer": allocation["optimizer"], "selection": allocation["selection"]},
            "accepted_control_template_sources": accepted_refs,
            "candidate_parameter_template_source": candidate_template_source,
            "candidate_selection": {"status": "pending", "graph_candidate_id": None, "memory_candidate_id": None,
                                    "default_g00_m00_selected": False,
                                    "note": "g00/m00 remains a pilot identity; no formal candidate selected by this preparation"},
            "architecture_identity": {"ANN_controls": ["V0", "V1", "V2"],
                                      "genuine_SNN": ["V3", "V4", "V9"],
                                      "formal_mandatory": "V4; ANN controls do not replace it",
                                      "reuse_links": {"V5_V6_V7": "same-seed V4 weights", "V8": "same-seed V2 weights"},
                                      "verification_scope": "configuration/source provenance only; no forward pass was run"},
            "at_launch_bindings": {"status": "pending-at-launch", "config_source_hashes": "concrete configs hashed in rows; candidate configs unresolved",
                                   "training_input_binding_sha256": None, "must_bind_dataset_sources_and_source_files": True},
            "planning_constraints": ["No training, validation selection, checkpoint loading, or test access was performed.",
                                     "V0/V1 concrete configs are exact seed0-template copies except model_seed/output_dir.",
                                     "V2/V3/V4/V9 rows carry copied template parameters but are not executable until root selection values/source/checkpoint/freeze are supplied.",
                                     "No candidate search or default g00/m00 assignment is implied.",
                                     "N256 is not part of this matrix and cannot reuse N64 checkpoints without a separately specified and verified route."],
            "rows": rows,
        }
        with MATRIX_PATH.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(matrix, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return {"matrix_path": str(MATRIX_PATH), "matrix_sha256": sha(MATRIX_PATH),
                "concrete_configs": concrete_count, "unresolved_rows": len(rows)-concrete_count,
                "total_rows": len(rows), "status": "planning_only"}
    except BaseException:
        # Partial preparation is preserved for review; nothing is deleted or launched.
        raise


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
