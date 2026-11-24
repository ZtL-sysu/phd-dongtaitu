"""Prepare required matched state controls, without choosing configs or training."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def main():
    controls = {
        "tau-init-2-16": {"tau_fast": 2.0, "tau_slow": 16.0},
        "encoding-change-only": {"encoding_mode": "change_only"},
        "encoding-periodic-level-8": {"encoding_mode": "periodic_level", "level_refresh_interval": 8},
    }
    source = ROOT / "edsgn/src/edsgn/model.py"
    rows = []
    for dataset in ("synthetic", "email", "mooc"):
        template = ROOT / "configs/core-development-bound-v3" / f"{dataset}-V4-seed0.json"
        conf = json.loads(template.read_text(encoding="utf-8-sig"))
        for control, model_delta in controls.items():
            for seed in range(5):
                stem = f"{dataset}-{control}-seed{seed}"
                rows.append({
                    "dataset": dataset, "variant": "V4", "control_id": control, "seed": seed,
                    "status": "requires_selected_core_config_and_training", "model_delta": model_delta,
                    "reference": "same dataset/seed independently trained selected hybrid V4",
                    "template_path": template.relative_to(ROOT).as_posix(),
                    "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
                    "template_core_params": conf["model"],
                    "planned_config_path": f"configs/formal-state-controls-v1/{stem}.json",
                    "planned_output_dir": f"runs/formal-state-controls-v1/{dataset}/{control}/seed{seed}",
                    "config_sha256": None, "training_binding_sha256": None, "checkpoint_sha256": None,
                    "selection_authority": None, "actual_training_started": False,
                })
    target = ROOT / "protocols/trained-state-control-matrix-root-preparation-v1.json"
    doc = {
        "schema_version": 1, "scope": "secondary matched independently trained state controls; preparation only",
        "accepted_by": "root", "status": "prepared-unselected-unexecuted", "rows": rows, "row_count": len(rows),
        "source_path": source.relative_to(ROOT).as_posix(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selected_primary_configuration": None, "actual_training_started": False, "test_accessed": False,
        "pipeline_advanced": False, "scientific_hypothesis_accepted": False,
        "requirements": [
            "Use the validation-selected in-family V4 settings, changing only the declared model delta, seed and output directory.",
            "Train fresh weights under the same full train/validation windows, AdamW, width, graph, loss, optimizer cap and checkpoint selection rule.",
            "The tau comparison changes positive learned initial pairs (1,8) versus (2,16); it does not fix learned taus or claim different physical units.",
            "Change-only and periodic-level are state-encoder controls; retain graph input, masks, observation ages and all other mechanisms.",
            "Periodic-level uses window-local history step and interval8; report that convention, rather than a cross-window persistent clock.",
            "Do not relabel a hybrid V4 checkpoint as one of these trained controls, or mutate checkpoint config to create a result.",
            "These secondary reports do not alter the 49 primary endpoints; retain five seeds and negative outcomes.",
            "All actual jobs remain subject to the single cumulative72-hour ceiling; preparation creates no reservation or extra clock.",
        ],
    }
    if len(rows) != 45 or len({(r["dataset"], r["control_id"], r["seed"]) for r in rows}) != 45:
        raise AssertionError("state-control matrix must be3 regimes x3 controls x5 model seeds")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(doc, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": doc["status"], "rows": len(rows), "path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
