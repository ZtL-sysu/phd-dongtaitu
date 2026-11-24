import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

assembler = load("guarded_primary_assembler_v2_test", ROOT / "edsgn/staging/guarded_primary_family_assembler_v2.py")
producer_tests = load("producer_fixture_for_assembler_v2", ROOT / "edsgn/tests/test_typed_endpoint_producer_v4.py")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def test_ni_raw_p_kind_matches_actual_v4_producer_contract():
    assert assembler.EXPECTED_RAW_P_KIND["forecast_mae_noninferiority"] == "centered_lower_tail_noninferiority_p"

def _localize_source_map(root, result_path, producer_path):
    doc = json.loads(result_path.read_text(encoding="utf-8"))
    localized = {}
    for source, digest in doc["producer_source_sha256"].items():
        original = Path(source)
        rel = original.relative_to(ROOT)
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(original.read_bytes())
        assert sha(dest) == digest
        localized[str(dest.resolve())] = digest
    doc["producer_source_sha256"] = localized
    result_path.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return doc

@pytest.mark.parametrize(("outcome", "expected"), [
    ("forecast_mae_superiority", "centered_two_sided_p"),
    ("forecast_mae_noninferiority", "centered_lower_tail_noninferiority_p"),
])
def test_public_v4_producer_record_maps_to_typed_family_row(tmp_path, monkeypatch, outcome, expected):
    fixture_module = producer_tests
    if outcome == "forecast_mae_noninferiority":
        # The older fixture helper mislabeled NI as H2; use the same public
        # producer fixture with its claim selector corrected to the legal H3.
        source = producer_tests.__file__
        text = Path(source).read_text(encoding="utf-8").replace(
            '.get(outcome,"H2")', '.get(outcome,"H3")')
        text = text.replace('''      plan["acf_evidence"]])
    put(root,"pipeline-state.json",state)''', '''      plan["acf_evidence"]])
    if outcome == "forecast_mae_noninferiority": p3a["inputs"].append(plan["epsilon_evidence"])
    put(root,"pipeline-state.json",state)''')
        text = text.replace('''        binding_runs.append(row)''', '''        if outcome == "forecast_mae_noninferiority": row.update(epsilon_status="accepted", epsilon_value_raw=0.2, epsilon_authority_path=plan["epsilon_evidence"]["path"])
        binding_runs.append(row)''')
        fixture_module = type(sys)("producer_fixture_ni_v2")
        fixture_module.__file__ = source
        sys.modules[fixture_module.__name__] = fixture_module
        exec(compile(text, source, "exec"), fixture_module.__dict__)
    root, plan_path, registry_path, manifest_path = fixture_module._public_forecast_fixture(
        tmp_path, monkeypatch, outcome=outcome)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    ref = next(row for row in registry["endpoint_plans"] if row["endpoint_id"] == plan["endpoint_id"])
    result_path = root / "reviews" / f"{outcome}.json"
    produced = fixture_module.producer.produce_endpoint(plan_path, registry_path, manifest_path, result_path, root=root)
    assert produced["identifiable"] is True
    assert produced["raw_p_kind"] == expected
    producer_path = root / "edsgn/staging/typed_endpoint_producer_v4.py"
    _localize_source_map(root, result_path, producer_path)
    produced = json.loads(result_path.read_text(encoding="utf-8"))
    validator = assembler._load_plan_validator()
    consumed = assembler._revalidate_producer_metadata(root, validator, registry_path, registry,
        sha(registry_path), ref, plan_path, plan, result_path, produced)
    assert consumed == produced["consumed_hashes"]
    mapped = assembler._map_producer_result(root, result_path, ref, plan, registry_path,
        sha(registry_path), producer_path)
    assert mapped["raw_p"] == produced["raw_p"]
    if outcome.endswith("noninferiority"):
        assert mapped["epsilon"] == produced["epsilon_raw"]
        assert mapped["one_sided95_upper"] == produced["upper_95"]
    else:
        assert mapped["percentile95_upper"] == produced["percentile_95_interval"][1]

def test_v4_noninferiority_result_field_mapping_is_exact():
    producer = producer_tests.producer
    analysis = {"scope": "paired_forecast_bootstrap", "estimate": -0.02,
                "upper_95": 0.01, "centered_lower_tail_noninferiority_p": 0.03}
    fields = producer._analysis_fields(analysis, noninferiority=True, epsilon=0.05)
    assert fields["raw_p_kind"] == assembler.EXPECTED_RAW_P_KIND["forecast_mae_noninferiority"]
    doc = {"schema_version": 1, "scope": assembler.PRODUCER_SCOPE, "split": "test",
           "endpoint_id": "H3-test-ni", "claim": "H3", "outcome_type": "forecast_mae_noninferiority",
           "plan_path": "plans/ni.json", "plan_sha256": "a"*64, "registry_path": "plans/reg.json",
           "registry_sha256": "b"*64, "seed_ids": [0,1,2,3,4], "n_full_origins": 4,
           "identifiable": True, "producer_source_sha256": {}, "raw_p_kind": fields["raw_p_kind"],
           "raw_p": fields["raw_p"], "effect_estimate": fields["effect_estimate"],
           "upper_95": fields["upper_95"], "epsilon_raw": .05, "analysis": analysis}
    assert doc["raw_p_kind"] == "centered_lower_tail_noninferiority_p"

def test_revalidation_rejects_consumed_input_mutation_after_producer_serialization(tmp_path, monkeypatch):
    root, plan_path, registry_path, manifest_path = producer_tests._public_forecast_fixture(
        tmp_path, monkeypatch, outcome="forecast_mae_superiority")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    ref = next(row for row in registry["endpoint_plans"] if row["endpoint_id"] == plan["endpoint_id"])
    result_path = root / "reviews/public-producer.json"
    produced = producer_tests.producer.produce_endpoint(plan_path, registry_path, manifest_path, result_path, root=root)
    consumed_path = root / next(iter(produced["consumed_hashes"]))
    consumed_path.write_bytes(consumed_path.read_bytes() + b"tamper")
    validator = assembler._load_plan_validator()
    with pytest.raises(assembler.AssemblyError):
        assembler._revalidate_producer_metadata(root, validator, registry_path, registry,
            sha(registry_path), ref, plan_path, plan, result_path, produced)

def test_public_assembler_exposes_no_test_root_or_nonofficial_bypass(tmp_path):
    assert "_test_root" not in __import__("inspect").signature(assembler.assemble_primary_families).parameters
    with pytest.raises(assembler.AssemblyError, match="canonical RUN"):
        assembler.assemble_primary_families(tmp_path / "registry.json", tmp_path / "accept.json",
            tmp_path / "out.json", root=tmp_path, official=False)

def test_root_holm_acceptance_is_current_and_explicitly_non_scientific():
    _, evidence = assembler._verify_accepted_holm(ROOT)
    accepted = json.loads(assembler.HOLM_ACCEPTANCE_PATH.read_text(encoding="utf-8"))
    assert evidence and accepted["pipeline_advanced"] is False
    assert accepted["scientific_hypothesis_accepted"] is False
    assert accepted["actual_primary_inference_run"] is False
