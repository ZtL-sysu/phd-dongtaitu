from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_completed_p3_campaign_v4.py"
_SPEC = importlib.util.spec_from_file_location("audit_completed_p3_campaign_v4", _SCRIPT)
batch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(batch)


def put(path: Path, data: bytes = b"x") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> str:
    return put(path, (json.dumps(value, sort_keys=True) + "\n").encode())


class CampaignAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = self.root / "data" / "toy"
        self.ds.mkdir(parents=True)
        meta = {"partition": "train", "history": 2, "hmax": 1, "splits": {"train": [0, 4]}}
        np.savez(self.ds / "train_seed0.npz", metadata_json=np.asarray(json.dumps(meta)))
        meta2 = {"partition": "validation", "history": 2, "hmax": 1, "splits": {"validation": [4, 8]}}
        np.savez(self.ds / "validation_seed0.npz", metadata_json=np.asarray(json.dumps(meta2)))
        self.config_path = self.root / "configs" / "toy.json"
        self.config = {"variant": "Toy", "seed": 0, "dataset": "email", "dataset_path": str(self.ds),
                       "history": 2, "hmax": 1}
        write_json(self.config_path, self.config)
        self.out = self.root / "runs" / "toy"
        self.out.mkdir(parents=True)
        self.job = {"id": "toy-job", "queue_key": "fixture", "queue_path": str(self.root / "queue.json"),
                    "queue_sha256": "a" * 64, "acceptance_path": str(self.root / "accept.json"),
                    "acceptance_sha256": "b" * 64, "config_path": str(self.config_path),
                    "config_sha256": batch.sha256(self.config_path), "config": self.config,
                    "output_dir": str(self.out), "dataset": "email", "variant": "Toy", "seed": 0}
        self._make_completed()

    def tearDown(self):
        self.tmp.cleanup()

    def _make_completed(self):
        for name in batch.TRAINING_PRINCIPAL:
            put(self.out / name, (name + "\n").encode())
        cp = self.out / "checkpoint.pt"
        cp_sha = batch.sha256(cp)
        write_json(self.out / "training_summary.json", {"scope": "full_protocol", "test_accessed": False,
                    "stop_loss_reached": False, "checkpoint_sha256": cp_sha, "epochs_executed": 2})
        write_json(self.out / "manifest.json", {"scope": "full_protocol", "train_windows": 3, "val_windows": 3,
                    "dataset_path": str(self.ds)})
        write_json(self.out / "config_resolved.json", self.config)
        self.binding_path = self.out / "input-binding-at-launch.json"
        self._write_binding()
        inp = self.root / "protocols" / "input.json"
        inp_sha = write_json(inp, {"fixed": True})
        outputs = [{"path": str(self.out / n), "sha256": batch.sha256(self.out / n)} for n in batch.TRAINING_PRINCIPAL]
        receipt = {"scope": "offline-experiment-phase", "id": "toy-job", "phase": "training", "status": "completed",
                   "returncode": 0, "queue_sha256": self.job["queue_sha256"], "output_dir": str(self.out),
                   "inputs": [{"path": str(inp), "sha256": inp_sha},
                              {"path": str(self.config_path), "sha256": batch.sha256(self.config_path)}],
                   "outputs": outputs}
        write_json(self.root / "logs" / "offline-phase-wall" / "toy-job.json", receipt)

    def _write_binding(self):
        write_json(self.binding_path, {"config_source": {"path": str(self.config_path),
                    "sha256": batch.sha256(self.config_path), "config": self.config},
                    "dataset_root": str(Path(self.config["dataset_path"]).resolve())})

    def numeric_stub(self, config_path, run_dir):
        source_path = Path(self.config["dataset_path"]).resolve()
        if source_path.is_dir():
            source_path = self.ds / "validation_seed0.npz"
            train_path = self.ds / "train_seed0.npz"
        else:
            train_path = source_path
        return {"status": "passed", "config_path": str(self.config_path), "run_dir": str(self.out),
                "dataset_path": str(Path(self.config["dataset_path"]).resolve()), "split": "validation",
                "seed": 0, "variant": "Toy", "checkpoint_sha256": batch.sha256(self.out / "checkpoint.pt"),
                "origins_complete_and_ordered": True, "saved_targets_and_masks_match_validation_source": True,
                "test_arrays_or_test_windows_read": False, "n_validation_origins": 3,
                "train_origins_recomputed_from_metadata": 3,
                "required_horizons": [1, 3, 7],
                "validation_source_files": [{"path": str(source_path), "sha256": batch.sha256(source_path)}],
                "training_metadata_source_files": [{"path": str(train_path), "sha256": batch.sha256(train_path)}],
                "required_horizon_metric_csv_checks": [{"matches_recomputed": True}],
                "training": {"epochs_executed": 2, "train_windows_per_epoch": 3, "val_windows": 3,
                             "summary_best_val_mae_matches_log_minimum_and_saved_predictions": True},
                "binding": {"status": "verified", "binding_path": str(self.binding_path)},
                "required_horizon_feature_rows": [{"horizon": 1, "feature": 0, "mae": 0.5}],
                "required_horizon_mean_selection_mae": 0.5,
                "required_horizon_pooled_sum_count": [{"horizon": 1, "mae_pooled_sum_over_count": 0.5}],
                "test_arrays_or_test_windows_read": False}

    @staticmethod
    def binding_verifier(path):
        return {"status": "verified", "path": str(path)}

    def test_completed_split_separated_fixture_audits_with_stub(self):
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub, binding_verifier=self.binding_verifier)
        self.assertEqual(row["status"], "audited")
        self.assertEqual(row["train_windows"], 3)
        self.assertEqual(row["val_windows"], 3)
        self.assertEqual(row["epochs"], 2)
        self.assertFalse(row["training_run_declared_test_accessed"])
        self.assertFalse(row["scored_test_origins"])
        self.assertFalse(row["source_array_may_materialize_other_partitions"])

    def test_missing_receipt_is_explicit_pending(self):
        (self.root / "logs" / "offline-phase-wall" / "toy-job.json").unlink()
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub, binding_verifier=self.binding_verifier)
        self.assertEqual(row["status"], "pending")

    def test_output_hash_mutation_fails_closed(self):
        receipt = self.root / "logs" / "offline-phase-wall" / "toy-job.json"
        data = json.loads(receipt.read_text())
        data["outputs"][0]["sha256"] = "0" * 64
        write_json(receipt, data)
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub, binding_verifier=self.binding_verifier)
        self.assertEqual(row["status"], "failed")

    def test_combined_chronological_npz_is_audited_with_disclosure(self):
        combined = self.root / "data" / "combined.npz"
        combined_meta = {"partition": "chronological", "history": 2, "hmax": 7,
                         "splits": {"train": [0, 4], "validation": [4, 8], "test": [8, 16]}}
        np.savez(combined, metadata_json=np.asarray(json.dumps(combined_meta)), targets=np.zeros((16, 2, 1)))
        self.job["config"]["dataset_path"] = str(combined)
        write_json(self.out / "config_resolved.json", self.job["config"])
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["dataset_path"] = str(combined)
        write_json(self.out / "manifest.json", manifest)
        write_json(self.config_path, self.job["config"])
        self.job["config_sha256"] = batch.sha256(self.config_path)
        self._write_binding()
        receipt_path = self.root / "logs" / "offline-phase-wall" / "toy-job.json"
        receipt = json.loads(receipt_path.read_text())
        for inp in receipt["inputs"]:
            if Path(inp["path"]) == self.config_path:
                inp["sha256"] = batch.sha256(self.config_path)
        for output in receipt["outputs"]:
            output_name = Path(output["path"]).name
            if output_name in ("config_resolved.json", "manifest.json"):
                output["sha256"] = batch.sha256(self.out / output_name)
        write_json(receipt_path, receipt)
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub, binding_verifier=self.binding_verifier)
        self.assertEqual(row["status"], "audited")
        self.assertFalse(row["scored_test_origins"])
        self.assertTrue(row["source_array_may_materialize_other_partitions"])
        self.assertEqual(row["numeric_source_mode"], "combined_chronological")

    def test_resolved_config_must_match_every_requested_field(self):
        resolved = dict(self.config)
        resolved["lr"] = 999.0
        write_json(self.out / "config_resolved.json", resolved)
        receipt_path = self.root / "logs" / "offline-phase-wall" / "toy-job.json"
        receipt = json.loads(receipt_path.read_text())
        for output in receipt["outputs"]:
            if Path(output["path"]).name == "config_resolved.json":
                output["sha256"] = batch.sha256(self.out / "config_resolved.json")
        write_json(receipt_path, receipt)
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub, binding_verifier=self.binding_verifier)
        self.assertEqual(row["status"], "failed")
        self.assertIn("complete requested config", row["reason"])

    def test_launch_binding_config_hash_must_match_requested_config(self):
        binding = json.loads(self.binding_path.read_text())
        binding["config_source"]["sha256"] = "0" * 64
        write_json(self.binding_path, binding)
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub, binding_verifier=self.binding_verifier)
        self.assertEqual(row["status"], "failed")
        self.assertIn("exact requested config", row["reason"])

    def test_phase_receipt_must_explicitly_bind_requested_config(self):
        receipt_path = self.root / "logs" / "offline-phase-wall" / "toy-job.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["inputs"] = [item for item in receipt["inputs"] if Path(item["path"]) != self.config_path]
        write_json(receipt_path, receipt)
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub, binding_verifier=self.binding_verifier)
        self.assertEqual(row["status"], "failed")
        self.assertIn("directly bind the exact requested config", row["reason"])

    def _set_dataset(self, dataset):
        self.config["dataset"] = dataset
        self.job["dataset"] = dataset
        write_json(self.config_path, self.config)
        self.job["config_sha256"] = batch.sha256(self.config_path)
        write_json(self.out / "config_resolved.json", self.config)
        self._write_binding()
        receipt_path = self.root / "logs" / "offline-phase-wall" / "toy-job.json"
        receipt = json.loads(receipt_path.read_text())
        for item in receipt["inputs"]:
            if Path(item["path"]) == self.config_path:
                item["sha256"] = batch.sha256(self.config_path)
        for item in receipt["outputs"]:
            name = Path(item["path"]).name
            if name == "config_resolved.json":
                item["sha256"] = batch.sha256(self.out / name)
        write_json(receipt_path, receipt)

    def test_mooc_horizon_report_is_reused_without_numeric_reaudit(self):
        self._set_dataset("mooc")
        report = self.numeric_stub(self.config_path, self.out)
        report["required_horizons"] = [1, 6, 24]
        reuse = self.root / "prior-numeric.json"
        write_json(reuse, report)

        def must_not_run(*args, **kwargs):
            raise AssertionError("valid MOOC report should be reused")

        row = batch.audit_job(self.root, self.job, numeric_fn=must_not_run,
                              reuse_path=reuse, binding_verifier=self.binding_verifier)
        self.assertEqual(row["status"], "reused")
        self.assertEqual(batch.REQUIRED_HORIZONS_BY_DATASET["mooc"], [1, 6, 24])

    def test_mooc_rejects_email_horizons_and_unknown_dataset(self):
        self._set_dataset("mooc")
        report = self.numeric_stub(self.config_path, self.out)
        completed = {"checkpoint_sha256": batch.sha256(self.out / "checkpoint.pt"),
                     "val_windows": 3, "train_windows": 3, "epochs": 2,
                     "binding_sha256": batch.sha256(self.binding_path)}
        self.assertFalse(batch._numeric_valid(report, self.job, completed,
                                              binding_verifier=self.binding_verifier))
        self.job["dataset"] = "unregistered"
        self.assertFalse(batch._numeric_valid(report, self.job, completed,
                                              binding_verifier=self.binding_verifier))


if __name__ == "__main__":
    unittest.main()
