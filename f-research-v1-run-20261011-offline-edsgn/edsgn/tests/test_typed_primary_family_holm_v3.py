import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn" / "staging"))
from typed_primary_family_holm_v3 import assemble_primary_families

REGISTRY = json.loads((ROOT / "protocols/primary-endpoint-registry-root-preparation-v2.json").read_text(encoding="utf-8"))["primary_endpoints"]

def numeric_rows(p_by_id=None):
    out=[]
    for desc in REGISTRY:
        row={"endpoint_id":desc["endpoint_id"], "outcome_type":desc["outcome_type"], "identifiable":True, "raw_p":(p_by_id or {}).get(desc["endpoint_id"], .5), "estimate":-.2, "percentile95_upper":-.01}
        if desc["outcome_type"] == "forecast_mae_noninferiority":
            row.pop("percentile95_upper"); row.update(epsilon=.5, one_sided95_upper=.4)
        out.append(row)
    return out

def test_structural_registry_is_exact_49_rows_and_expected_family_members():
    assert len(REGISTRY)==49
    assert {f:sum(r["endpoint_id"].startswith(f+"-") for r in REGISTRY) for f in ("H1","H2","H3")} == {"H1":9,"H2":16,"H3":24}
    out=assemble_primary_families(REGISTRY,numeric_rows())
    assert {k:v["expected_size"] for k,v in out["families"].items()} == {"H1":9,"H2":16,"H3":24}
    assert all(v["holm_denominator"]==v["expected_size"] for v in out["families"].values())

def test_non_equal_holm_matches_hand_calculation_and_retains_input_member_order():
    h1=[r["endpoint_id"] for r in REGISTRY if r["endpoint_id"].startswith("H1-")]
    ps=[.001,.004,.01,.02,.03,.04,.05,.06,.2]
    out=assemble_primary_families(REGISTRY,numeric_rows(dict(zip(h1,ps))))["families"]["H1"]
    got={m["endpoint_id"]:m["holm_adjusted_p"] for m in out["members"]}
    expected=[.009,.032,.07,.12,.15,.16,.16,.16,.2]
    assert [got[k] for k in h1] == pytest.approx(expected)

def test_false_claim_and_result_type_mismatch_rejected():
    bad=[dict(x) for x in REGISTRY]; bad[0]["claim"]="H3"
    with pytest.raises(ValueError,match="claim does not match"):
        assemble_primary_families(bad,numeric_rows())
    bad=[dict(x) for x in REGISTRY]; bad[0]["outcome_type"]="forecast_mae_superiority"
    vals=numeric_rows(); vals[0]["outcome_type"]="forecast_mae_superiority"; vals[0].pop("percentile95_upper",None)
    with pytest.raises(ValueError,match="not allowed for claim"):
        assemble_primary_families(bad,vals)
    vals=numeric_rows(); vals[0]["outcome_type"]="forecast_mae_superiority"
    with pytest.raises(ValueError,match="outcome_type mismatch"):
        assemble_primary_families(REGISTRY,vals)

def test_unidentifiable_ni_keeps_null_p_and_blocks_whole_family_without_shrinking():
    vals=numeric_rows(); ni=next(x for x in vals if x["outcome_type"]=="forecast_mae_noninferiority")
    ni.update(identifiable=False,raw_p=None,epsilon=None,one_sided95_upper=None)
    fam=assemble_primary_families(REGISTRY,vals)["families"]["H1"]
    assert fam["status"]=="inconclusive_unidentifiable_member" and fam["holm_denominator"] is None
    assert all(x["holm_adjusted_p"] is None and x["endpoint_gate_passed"] is None for x in fam["members"])
    vals=numeric_rows(); ni=next(x for x in vals if x["outcome_type"]=="forecast_mae_noninferiority"); ni["raw_p"]=None
    with pytest.raises(ValueError,match="finite number"):
        assemble_primary_families(REGISTRY,vals)

def test_identifiable_negative_support_is_not_unidentifiable_and_ni_bound_is_strict():
    vals=numeric_rows(); vals[0]["supported"]=False
    fam=assemble_primary_families(REGISTRY,vals)["families"]["H1"]
    assert fam["status"]=="unsupported_identifiable_member" and fam["holm_denominator"]==9
    assert fam["members"][0]["raw_p"]==.5 and fam["members"][0]["holm_adjusted_p"]==1.0
    ni=next(x for x in vals if x["outcome_type"]=="forecast_mae_noninferiority"); ni["one_sided95_upper"]=ni["epsilon"]
    fam=assemble_primary_families(REGISTRY,vals)["families"]["H1"]
    niout=next(x for x in fam["members"] if x["outcome_type"]=="forecast_mae_noninferiority")
    assert niout["directional_gate_passed"] is False

def test_superiority_requires_both_negative_point_and_upper_interval():
    vals=numeric_rows(); vals[0]["estimate"]=0.0
    fam=assemble_primary_families(REGISTRY,vals)["families"]["H1"]
    assert fam["members"][0]["directional_gate_passed"] is False
    vals=numeric_rows(); vals[0]["percentile95_upper"]=0.0
    fam=assemble_primary_families(REGISTRY,vals)["families"]["H1"]
    assert fam["members"][0]["directional_gate_passed"] is False

@pytest.mark.parametrize("field,value",[("primary",False),("primary",None),("holm_family","H3"),("holm_family",None)])
def test_registry_must_explicitly_mark_primary_and_exact_holm_family(field,value):
    bad=[dict(x) for x in REGISTRY]; bad[0][field]=value
    with pytest.raises(ValueError,match="explicitly primary|holm_family does not match"):
        assemble_primary_families(bad,numeric_rows())


@pytest.mark.parametrize("field",["primary","holm_family"])
def test_registry_missing_primary_family_metadata_is_rejected(field):
    bad=[dict(x) for x in REGISTRY]; bad[0].pop(field)
    with pytest.raises(ValueError,match="explicitly primary|holm_family does not match"):
        assemble_primary_families(bad,numeric_rows())
