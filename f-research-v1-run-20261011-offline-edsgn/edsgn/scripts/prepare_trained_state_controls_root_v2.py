"""Add the explicitly required frozen-versus-learned tau control; no execution."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def main():
    prior = ROOT / "protocols/trained-state-control-matrix-root-preparation-v1.json"
    doc = json.loads(prior.read_text(encoding="utf-8-sig"))
    rows = deepcopy(doc["rows"])
    for dataset in ("synthetic", "email", "mooc"):
        for seed in range(5):
            base = next(r for r in rows if r["dataset"] == dataset and r["seed"] == seed and r["control_id"] == "tau-init-2-16")
            row = deepcopy(base)
            stem = f"{dataset}-tau-frozen-selected-init-seed{seed}"
            row.update({"control_id": "tau-frozen-selected-init", "model_delta": {"learn_tau": False},
                        "planned_config_path": f"configs/formal-state-controls-v2/{stem}.json",
                        "planned_output_dir": f"runs/formal-state-controls-v2/{dataset}/tau-frozen-selected-init/seed{seed}"})
            rows.append(row)
    doc.update({"schema_version": 2, "rows": rows, "row_count": len(rows),
                "supersedes_preparation_only": prior.relative_to(ROOT).as_posix(),
                "prior_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
                "correction": "v1 included the second RP tau initial pair but omitted the already-required frozen-versus-learned tau control; v2 adds it without erasing v1 or claiming execution"})
    doc["requirements"].append("Frozen-tau control fixes the validation-selected initial fast/slow pair by learn_tau=false for the whole fresh training. Its learned-tau reference starts from that identical initial pair; do not use a test-selected pair or freeze a learned checkpoint after training and relabel it as a trained control.")
    assert len(rows) == 60 and len({(r["dataset"], r["control_id"], r["seed"]) for r in rows}) == 60
    output = ROOT / "protocols/trained-state-control-matrix-root-preparation-v2.json"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(doc, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": doc["status"], "rows": len(rows), "path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
