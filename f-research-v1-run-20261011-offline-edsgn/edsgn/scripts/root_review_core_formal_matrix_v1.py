"""Root's read-only identity review of pending formal configs; no execution."""
from pathlib import Path
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))


def main():
    rel = "protocols/core-formal-config-matrix-preparation-v1.json"
    matrix = read(rel)
    assert digest(ROOT / rel) == "95a694c0837b73e8d1968a0a2431bff5ee288cde88ced704d82aaa117c901ede"
    assert matrix["status"] == "planning_only" and matrix["required_rows"] == 90
    rows = matrix["rows"]
    want = {(d, v, s) for d in ("synthetic", "email", "mooc") for v in ("V0", "V1", "V2", "V3", "V4", "V9") for s in range(5)}
    assert len(rows) == 90 and {(r["dataset"], r["variant"], r["model_seed"]) for r in rows} == want
    assert matrix["architecture_identity"]["ANN_controls"] == ["V0", "V1", "V2"]
    assert matrix["architecture_identity"]["genuine_SNN"] == ["V3", "V4", "V9"]
    assert matrix["candidate_selection"]["default_g00_m00_selected"] is False
    files = {rel, "edsgn/scripts/prepare_core_formal_config_matrix_v1.py", "edsgn/scripts/root_review_core_formal_matrix_v1.py"}
    for key in ("root_decision", "resource_contract", "core_allocation", "generator"):
        ref = matrix[key]
        assert digest(ROOT / ref["path"]) == ref["sha256"]
        files.add(ref["path"])
    n = 0
    for row in rows:
        assert not Path(row["output_dir"]).exists() and row["launch_status"] == "planning_only"
        assert row["training_input_binding_sha256"] is None and row["test_access_authorized"] is False
        assert row["candidate_selection"]["default_g00_m00_selected"] is False
        template = read(row["template_config_path"])
        assert digest(ROOT / row["template_config_path"]) == row["template_config_sha256"]
        files.add(row["template_config_path"])
        if row["variant"] in {"V0", "V1"}:
            conf = read(row["config_path"])
            assert digest(ROOT / row["config_path"]) == row["config_sha256"]
            assert conf["variant"] == row["variant"] and conf["seed"] == row["model_seed"] and conf["output_dir"] == row["output_dir"]
            assert {k: v for k, v in conf.items() if k not in {"seed", "output_dir"}} == {k: v for k, v in template.items() if k not in {"seed", "output_dir"}}
            assert conf["epochs"] == 100 and conf["patience"] == 10
            assert not {"smoke_train_windows", "smoke_val_windows", "resume", "max_run_seconds"} & set(conf)
            files.add(row["config_path"]); n += 1
        else:
            assert row["configuration_status"] == "unresolved_candidate_selection"
            assert row["config_sha256"] is None and not (ROOT / row["config_path"]).exists()
            assert row["required_selection_placeholders"]["root_selection_record_sha256"] is None
    assert n == 30
    for path in (ROOT / "edsgn/src/edsgn").rglob("*.py"):
        files.add(path.relative_to(ROOT).as_posix())
    write_record("core-formal-config-matrix-root-preparation-v1", """# Root identity review: formal core matrix preparation

Root independently checked all90 unique variant/regime/model-seed rows, all30
materialized V0/V1 configs against their hash-bound accepted templates, all60
unresolved proposed-model rows, absent output directories and current source
bytes. V0/V1/V2 are ANN controls; V3/V4/V9 are genuine SNN interventions. Every
concrete control config changes only seed and output directory. Proposed-model
selection, concrete configs, launch bindings, checkpoints and test evidence are
still pending. No g00/m00 selection or seed0 reuse is approved here.

This is preparation identity acceptance only. It permits no optimizer, test,
phase transition or scientific conclusion. Actual execution remains conditional
on the canonical gates, complete source/data/config bindings, validation-only
selection, independent numeric review and the existing cumulative72-hour clock.
""", sorted(files), "formal-core-configuration-identity-preparation-only",
                 total_rows=90, concrete_control_configs=30, unresolved_proposed_rows=60,
                 actual_training_started=False, arrays_decoded=False, actual_test_access=False,
                 default_g00_m00_selected=False, seed0_reuse_approved=False)
    print(json.dumps({"status": "pass-prepared-only", "rows": 90, "concrete_configs": 30, "pending_configs": 60,
                      "actual_training_started": False, "pipeline_advanced": False}))


if __name__ == "__main__":
    main()
