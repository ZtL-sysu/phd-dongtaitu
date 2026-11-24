import json
from pathlib import Path

import pytest

from typed_primary_family_holm_v1 import assemble_primary_families


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / "protocols/primary-endpoint-registry-root-preparation-v2.json").read_text(encoding="utf-8"))["primary_endpoints"]


def rows(*, p=0.001, estimate=-1.0, upper=-0.1):
    result = []
    for desc in REGISTRY:
        row = {"endpoint_id": desc["endpoint_id"], "outcome_type": desc["outcome_type"], "identifiable": True,
               "raw_p": p, "estimate": estimate, "percentile95_upper": upper}
        if desc["outcome_type"] == "forecast_mae_noninferiority":
            row.update(epsilon=0.5, one_sided95_upper=0.25)
            row.pop("percentile95_upper")
        result.append(row)
    return result


def test_exact_49_types_and_complete_holm_families():
    out = assemble_primary_families(REGISTRY, rows())
    assert {k: v["expected_size"] for k, v in out["families"].items()} == {"H1": 9, "H2": 16, "H3": 24}
    # Equal small p-values yield m*p, proving the denominator is complete.
    assert out["families"]["H3"]["members"][0]["holm_adjusted_p"] == pytest.approx(24 * 0.001)
    assert all(v["all_directional_gates_passed"] for v in out["families"].values())


def test_unidentifiable_member_blocks_family_without_reduced_holm():
    numeric = rows()
    target = next(x for x in numeric if x["outcome_type"] == "forecast_mae_noninferiority")
    target.update(identifiable=False, raw_p=None, epsilon=None, one_sided95_upper=None)
    family = assemble_primary_families(REGISTRY, numeric)["families"]["H1"]
    assert family["status"] == "inconclusive_unidentifiable_member"
    assert family["holm_denominator"] is None
    assert family["all_directional_gates_passed"] is None
    assert all(x["holm_adjusted_p"] is None and x["endpoint_gate_passed"] is None for x in family["members"])
    assert sum(x["raw_p"] is None for x in family["members"]) == 1


def test_identifiable_unsupported_is_not_inconclusive():
    numeric = rows()
    numeric[0]["supported"] = False
    family = assemble_primary_families(REGISTRY, numeric)["families"]["H1"]
    assert family["status"] == "unsupported_identifiable_member"
    assert family["holm_denominator"] == 9
    assert family["all_directional_gates_passed"] is False


@pytest.mark.parametrize("mutate", [
    lambda x: x.pop(),
    lambda x: x.append(dict(x[0])),
    lambda x: x[0].update(outcome_type="made_up"),
    lambda x: x[0].update(raw_p=True),
    lambda x: x[0].update(raw_p=float("nan")),
    lambda x: x[0].update(raw_p=1.1),
])
def test_rejects_incomplete_duplicate_wrong_type_and_invalid_p(mutate):
    numeric = rows()
    mutate(numeric)
    with pytest.raises(ValueError):
        assemble_primary_families(REGISTRY, numeric)


def test_noninferiority_requires_positive_margin_and_uses_one_sided_upper():
    numeric = rows()
    ni = next(x for x in numeric if x["outcome_type"] == "forecast_mae_noninferiority")
    ni["epsilon"] = None
    with pytest.raises(ValueError, match="null epsilon"):
        assemble_primary_families(REGISTRY, numeric)
    ni["epsilon"] = -0.2
    with pytest.raises(ValueError, match="positive"):
        assemble_primary_families(REGISTRY, numeric)
    ni["epsilon"] = 0.2
    ni["one_sided95_upper"] = 0.2
    out = assemble_primary_families(REGISTRY, numeric)
    member = next(x for x in out["families"]["H1"]["members"] if x["endpoint_id"] == ni["endpoint_id"])
    assert member["directional_gate_passed"] is False  # strict upper < epsilon


def test_superiority_requires_negative_estimate_and_ci_upper():
    numeric = rows(estimate=0.1, upper=-0.01)
    out = assemble_primary_families(REGISTRY, numeric)
    assert out["families"]["H2"]["all_directional_gates_passed"] is False

