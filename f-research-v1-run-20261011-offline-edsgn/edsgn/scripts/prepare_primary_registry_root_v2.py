"""Enumerate the root's prospective endpoints; no data/model/phase access."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = ROOT / "protocols/primary-endpoint-operational-definitions-root-v1.md"


def row(endpoint_id, claim, outcome, domain, left, right, slice_name,
        *, case=None, device=None, margin=False, note=None):
    return {
        "endpoint_id": endpoint_id,
        "claim": claim,
        "primary": True,
        "outcome_type": outcome,
        "domain": domain,
        "synthetic_case": case,
        "missing_rate": 0.0 if domain == "synthetic-N64" else None,
        "device": device,
        "left": left,
        "right": right,
        "direction": "left_minus_right_negative",
        "slice": slice_name,
        "seed_ids": [0, 1, 2, 3, 4],
        "required_horizons": [1, 6, 24] if domain == "mooc" else [1, 3, 7],
        "history": 24 if domain == "mooc" else 28,
        "margin_kind": "validation_raw_epsilon" if margin else "zero_superiority",
        "inference": "centered_lower_tail_noninferiority" if margin else "centered_two_sided_superiority",
        "directional_ci_gate": "one_sided95_upper_below_epsilon" if margin else "percentile95_upper_below_zero",
        "holm_family": claim,
        "result_path": f"runs/statistics/test/{claim}/{endpoint_id}.json",
        "status": "preparation_pending_bound_input_plan",
        "binding": {
            "input_plan_path": None, "input_plan_sha256": None,
            "validation_block_length": None, "acf_evidence": None,
            "epsilon_evidence": None if margin else "not_applicable",
            "producer_path": None, "producer_sha256": None,
        },
        "note": note,
    }


def build():
    rows = [row("H1-noise-only-tv-V4-V9", "H1", "graph_tv_noise_superiority",
                "synthetic-N64", "V4", "V9", "full_post_switch", case="noise-only")]
    cases = ["identity10", "identity30", "weight10", "weight30"]
    for case in cases:
        rows.append(row(f"H1-{case}-mae-ni-V4-V9", "H1", "forecast_mae_noninferiority",
                        "synthetic-N64", "V4", "V9", "post4L", case=case, margin=True))
        rows.append(row(f"H1-{case}-tv-response-V4", "H1", "graph_tv_response_superiority",
                        "synthetic-N64", "V4-negative-post-tv", "V4-negative-pre-tv",
                        "paired_pre_post4L", case=case,
                        note="Within-V4 operational response; does not isolate confidence weighting's response effect."))
    for case in cases:
        rows.append(row(f"H2-{case}-mae-V4-V3", "H2", "forecast_mae_superiority",
                        "synthetic-N64", "V4", "V3", "post4L", case=case))
        for control in ["shuffled-c", "constant-g", "unconditional-neighbor-reset"]:
            rows.append(row(f"H2-{case}-matched-mae-informative-{control}", "H2",
                            "forecast_mae_superiority", "synthetic-N64",
                            "V4-teacher-informative", f"V4-teacher-{control}", "post4L", case=case,
                            note="Identical V4 weights, prescribed teacher graph/actions and scorer inputs; informative parity required."))
    for domain in ["email", "mooc"]:
        for device in ["cpu", "cuda"]:
            for reference in ["V4", "V5", "V6"]:
                base = f"H3-{domain}-{device}-V7-{reference}"
                note = (
                    "Main email validation epsilon/selection remains unidentifiable; retain endpoint without p-value if unresolved."
                    if domain == "email" else None
                )
                rows.append(row(base + "-mae-ni", "H3", "forecast_mae_noninferiority",
                                domain, "V7", reference, "full_split", device=device,
                                margin=True, note=note))
                rows.append(row(base + "-window-cost", "H3", "paired_window_cost_superiority",
                                domain, "V7", reference, "full_split", device=device,
                                note="Median of exactly five full timed replays per seed/origin; bootstrap origins, not repeat IDs."))
    counts = Counter(x["claim"] for x in rows)
    if counts != {"H1": 9, "H2": 16, "H3": 24} or len({x["endpoint_id"] for x in rows}) != 49:
        raise RuntimeError("prospective primary family enumeration differs from root decisions")
    return {
        "schema_version": 2,
        "artifact_status": "root_structural_preparation_not_frozen_not_executed",
        "author": "root",
        "definitions": {
            "path": str(DEFINITIONS.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(DEFINITIONS.read_bytes()).hexdigest(),
        },
        "split": "test",
        "family_sizes": dict(counts),
        "alpha": 0.05,
        "bootstrap": {"reps": 2000, "random_seed": 1234,
                      "unit": "paired_circular_origin_blocks_within_trajectory",
                      "same_draws_across_seed_horizon_methods": True,
                      "block_status": "pending_validation_bound_P3A_freeze"},
        "multiplicity": "once_per_complete_H1_H2_H3_family_Holm",
        "unidentifiable_rule": "retain_member_null_p_block_family_never_reduce_denominator",
        "primary_endpoints": rows,
        "required_secondary_work_waived": False,
        "test_access_authorized": False,
        "scientific_inference_executed": False,
        "pending": ["accepted_P3A_registry_and_typed_plans", "validation_epsilon_and_dependence_blocks",
                    "selected_policy_and_checkpoint_bindings", "accepted_gated_producer_and_full_input_artifacts"],
    }


if __name__ == "__main__":
    registry = build()
    target = ROOT / "protocols/primary-endpoint-registry-root-preparation-v2.json"
    target.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(target), "sha256": sha256(target.read_bytes()).hexdigest(),
                      "family_sizes": registry["family_sizes"], "status": registry["artifact_status"]}))
