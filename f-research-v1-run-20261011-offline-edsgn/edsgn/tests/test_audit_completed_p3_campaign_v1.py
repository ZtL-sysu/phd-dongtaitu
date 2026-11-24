from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_completed_p3_campaign_v1.py"
_SPEC = importlib.util.spec_from_file_location("audit_completed_p3_campaign_v1", _SCRIPT)
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
        self.config = {"variant": "Toy", "seed": 0, "dataset": "toy", "dataset_path": str(self.ds),
                       "history": 2, "hmax": 1}
        write_json(self.config_path, self.config)
        self.out = self.root / "runs" / "toy"
        self.out.mkdir(parents=True)
        self.job = {"id": "toy-job", "queue_key": "fixture", "queue_path": str(self.root / "queue.json"),
                    "queue_sha256": "a" * 64, "acceptance_path": str(self.root / "accept.json"),
                    "acceptance_sha256": "b" * 64, "config_path": str(self.config_path),
                    "config_sha256": batch.sha256(self.config_path), "config": self.config,
                    "output_dir": str(self.out), "dataset": "toy", "variant": "Toy", "seed": 0}
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
        inp = self.root / "protocols" / "input.json"
        inp_sha = write_json(inp, {"fixed": True})
        outputs = [{"path": str(self.out / n), "sha256": batch.sha256(self.out / n)} for n in batch.TRAINING_PRINCIPAL]
        receipt = {"scope": "offline-experiment-phase", "id": "toy-job", "phase": "training", "status": "completed",
                   "returncode": 0, "queue_sha256": self.job["queue_sha256"], "output_dir": str(self.out),
                   "inputs": [{"path": str(inp), "sha256": inp_sha}], "outputs": outputs}
        write_json(self.root / "logs" / "offline-phase-wall" / "toy-job.json", receipt)

    def numeric_stub(self, config_path, run_dir):
        return {"status": "passed", "required_horizon_feature_rows": [{"horizon": 1, "feature": 0, "mae": 0.5}],
                "required_horizon_mean_selection_mae": 0.5,
                "required_horizon_pooled_sum_count": [{"horizon": 1, "mae_pooled_sum_over_count": 0.5}],
                "test_arrays_or_test_windows_read": False}

    def test_completed_split_separated_fixture_audits_with_stub(self):
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub)
        self.assertEqual(row["status"], "audited")
        self.assertEqual(row["train_windows"], 3)
        self.assertEqual(row["val_windows"], 3)
        self.assertEqual(row["epochs"], 2)
        self.assertFalse(row["test_accessed"])

    def test_missing_receipt_is_explicit_pending(self):
        (self.root / "logs" / "offline-phase-wall" / "toy-job.json").unlink()
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub)
        self.assertEqual(row["status"], "pending")

    def test_output_hash_mutation_fails_closed(self):
        receipt = self.root / "logs" / "offline-phase-wall" / "toy-job.json"
        data = json.loads(receipt.read_text())
        data["outputs"][0]["sha256"] = "0" * 64
        write_json(receipt, data)
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub)
        self.assertEqual(row["status"], "failed")

    def test_combined_npz_is_rejected_before_new_numeric_audit(self):
        combined = self.root / "data" / "combined.npz"
        np.savez(combined, metadata_json=np.asarray("{}"), targets=np.zeros((2, 3, 1)))
        self.job["config"]["dataset_path"] = str(combined)
        # Keep resolved metadata aligned so the new-data policy is the rejecting gate.
        write_json(self.out / "config_resolved.json", self.job["config"])
        write_json(self.config_path, self.job["config"])
        self.job["config_sha256"] = batch.sha256(self.config_path)
        receipt_path = self.root / "logs" / "offline-phase-wall" / "toy-job.json"
        receipt = json.loads(receipt_path.read_text())
        for output in receipt["outputs"]:
            if Path(output["path"]).name == "config_resolved.json":
                output["sha256"] = batch.sha256(self.out / "config_resolved.json")
        write_json(receipt_path, receipt)
        row = batch.audit_job(self.root, self.job, numeric_fn=self.numeric_stub)
        self.assertEqual(row["status"], "failed")
        self.assertIn("combined/file NPZ", row["reason"])


if __name__ == "__main__":
    unittest.main()
