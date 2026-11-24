from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("continue_p3_strong_replication_serial_v2.py")
SPEC = importlib.util.spec_from_file_location("strong_continuation_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class Seed0AcceptanceSchemaTests(unittest.TestCase):
    def make_acceptance(self, root: Path, review_kind: str, *, typed=True):
        datasets = ("synthetic", "email", "mooc")
        methods = ("DGCRN", "SeqSNN")
        records, pins = [], {}
        for dataset in datasets:
            for method in methods:
                ident = f"{dataset}-{method}"
                if review_kind == "saved-json":
                    review_obj = {
                        "status": "passed",
                        "scope": "independent_saved_validation_numeric_and_binding_review",
                        "dataset": dataset, "method": method, "seed": 0,
                        "provenance": {"test_accessed": False},
                    }
                    review_rel = f"reviews/{ident}.json"
                    review_path = root / review_rel
                    review_path.parent.mkdir(parents=True, exist_ok=True)
                    review_path.write_text(json.dumps(review_obj), encoding="utf-8")
                elif review_kind == "verifier-json":
                    review_obj = {
                        "reviewer_role": "independent", "verdict": "pass",
                        "scope": "full_train_validation_numeric", "dataset": dataset,
                        "method": method, "seed": 0, "test_accessed": False,
                    }
                    review_rel = f"reviews/{ident}.json"
                    review_path = root / review_rel
                    review_path.parent.mkdir(parents=True, exist_ok=True)
                    review_path.write_text(json.dumps(review_obj), encoding="utf-8")
                else:
                    review_rel = f"reviews/{ident}.md"
                    review_path = root / review_rel
                    review_path.parent.mkdir(parents=True, exist_ok=True)
                    review_path.write_text("Root-attested independent train/validation review.\n", encoding="utf-8")
                paths = {"independent_review_path": review_rel}
                for name, path_key, hash_key in (
                    ("receipt", "phase_receipt_path", "phase_receipt_sha256"),
                    ("binding", "input_binding_path", "input_binding_sha256"),
                    ("config", "config_path", "config_sha256"),
                ):
                    rel = f"evidence/{ident}-{name}.json"
                    target = root / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("{}\n", encoding="utf-8")
                    paths[path_key] = rel
                    paths[hash_key] = MODULE.sha(target)
                    pins[rel] = paths[hash_key]
                paths["independent_review_sha256"] = MODULE.sha(review_path)
                pins[review_rel] = paths["independent_review_sha256"]
                records.append({
                    "dataset": dataset, "method": method, "seed": 0, "accepted": True,
                    **paths,
                    **({
                        "independent_review_verdict": "pass",
                        "independent_review_scope": "full_train_validation_numeric",
                        "independent_identity_verified_by": "root",
                    } if typed else {}),
                })
        acceptance_path = root / "acceptance.json"
        acceptance_path.write_text(json.dumps({
            "accepted_by": "root", "verdict": "pass",
            "scope": "P3 strong seed0 full train/validation independent numeric acceptance",
            "pipeline_advanced": False, "scientific_hypothesis_accepted": False,
            "test_accessed": False, "seed0_records": records, "evidence_sha256": pins,
        }), encoding="utf-8")
        return acceptance_path

    def run_acceptance(self, kind: str, *, typed=True):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_root = MODULE.ROOT
            MODULE.ROOT = root
            try:
                path = self.make_acceptance(root, kind, typed=typed)
                return MODULE.verify_seed0_numeric_acceptance(path, {"a": 1, "b": 2, "c": 3, "d": 4})
            finally:
                MODULE.ROOT = old_root

    def test_existing_saved_validation_json_schema_is_accepted(self):
        result = self.run_acceptance("saved-json")
        self.assertEqual(result["accepted_seed0_pairs"], 6)

    def test_standardized_verifier_json_schema_is_accepted(self):
        result = self.run_acceptance("verifier-json")
        self.assertEqual(result["accepted_seed0_pairs"], 6)

    def test_markdown_requires_typed_root_row_attestation_not_body_keywords(self):
        result = self.run_acceptance("markdown")
        self.assertEqual(result["accepted_seed0_pairs"], 6)
        with self.assertRaisesRegex(RuntimeError, "typed independent-review"):
            self.run_acceptance("markdown", typed=False)


if __name__ == "__main__":
    unittest.main()
