"""Root acceptance of the completed developmental email DGCRN validation run."""
import json
from pathlib import Path
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))


def main():
    numeric_rel = "logs/p3-bound-email-DGCRN-numeric-v3.json"
    review_rel = "reviews/p3-bound-email-DGCRN-independent-v3.json"
    phase_rel = "logs/offline-phase-wall/baseline-bound-v3-email-DGCRN-seed0.json"
    base = "runs/baselines-bound-v3/email/DGCRN/seed0"
    numeric, review, phase = read(numeric_rel), read(review_rel), read(phase_rel)
    assert digest(ROOT / review_rel) == "35b87a63660924ce5cd787e54283831be35fce8ce4b09a463d89cd9bb40c3fe6"
    assert numeric["status"] == "passed" and review["status"] == "passed"
    assert review["dataset"] == "email" and review["method"] == "DGCRN" and review["seed"] == 0
    assert review["provenance"]["test_origins_evaluated"] is False and review["provenance"]["checkpoint_deserialized"] is False
    assert phase["status"] == "completed" and phase["returncode"] == 0
    assert numeric["n_validation_origins"] == 74 and numeric["train_origins_recomputed_from_metadata"] == 528
    assert review["phase"]["epochs_executed"] == 44 and review["phase"]["best_epoch"] == 34
    summary = read(base + "/training_summary.json")
    assert summary["test_accessed"] is False and summary["stop_loss_reached"] is False
    assert abs(numeric["required_horizon_mean_selection_mae"] - 0.0018601083259029523) < 1e-12
    for row in phase["inputs"] + phase["outputs"]:
        path = Path(row["path"])
        path = path if path.is_absolute() else ROOT / path
        assert path.is_relative_to(ROOT) and digest(path) == row["sha256"]
    assert digest(ROOT / (base + "/checkpoint.pt")) == numeric["checkpoint_sha256"] == review["checkpoint_sha256"]
    files = [numeric_rel, review_rel, "reviews/p3-bound-email-DGCRN-independent-v3.md", phase_rel,
             base + "/input-binding-at-launch.json", base + "/training_summary.json",
             base + "/config_resolved.json", base + "/training_log.csv",
             "configs/baselines-bound-v3/email-DGCRN-seed0.json"]
    write_record("p3-bound-email-DGCRN-root-acceptance-v3", """# Root acceptance: completed developmental email DGCRN seed0

The full source-bound training/validation run completed44 epochs under the
100-epoch cap and patience10, selecting epoch34. Independent saved-array
reconstruction verified528 training windows and all74 ordered validation
origins561–634, raw targets/masks, h1/3/7 feature metrics, the training-only
normalization fit, checkpoint bytes,22 package sources,25 phase inputs and7
outputs. Root rechecked the completed zero-return receipt and every input/output
SHA. Mean raw MAE at the required horizons is0.0018601083259029523; pooled raw
RMSE is0.003594212141914529. Training time is6597.8611235 seconds and the managed
phase time6610.7296374 seconds; budget accounting uses their maximum once.

The primary email validation period is degenerate with zero targets. This small
MAE is developmental validation evidence, not useful predictive generalization
or authority for a positive H3 noninferiority margin. Main email's predeclared
unidentifiable margin remains explicit. The separate email episode cannot repair
or replace the primary result. Combined NPZ target/mask members may have been
materialized in memory; only validation indices were used by this audit. No
checkpoint was deserialized, model executed or test-origin metric evaluated.

This accepts one seed0 training/validation cell only. P3 remains running; no
five-seed inference, final test result, H1/H2/H3 acceptance or stage advancement
is implied. The next source-bound job started serially by the existing campaign.
""", files, "completed-development-seed-validation",
                 dataset="email", variant="DGCRN", model_seed=0,
                 required_horizon_mean_mae=numeric["required_horizon_mean_selection_mae"],
                 formal_five_seed_result=False, final_test_result=False,
                 main_email_margin_status="unidentifiable_by_predeclared_rule")
    print(json.dumps({"verdict": "pass", "dataset": "email", "method": "DGCRN", "seed": 0,
                      "scope": "development_train_validation_only", "pipeline_advanced": False}))


if __name__ == "__main__":
    main()
