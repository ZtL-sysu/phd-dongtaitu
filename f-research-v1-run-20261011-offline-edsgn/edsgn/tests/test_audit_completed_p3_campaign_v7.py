from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_completed_p3_campaign_v7.py"
_SPEC = importlib.util.spec_from_file_location("audit_completed_p3_campaign_v7", _SCRIPT)
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

    def test_preflight_normalization_preserves_phase_and_scope_for_recovery_map(self):
        data=self.root/"data"/"email.npz";data.parent.mkdir(parents=True,exist_ok=True);data.write_bytes(b"opaque fixture")
        cfg=self.root/"configs"/"job.json"
        write_json(cfg,{"dataset_path":str(data),"dataset":"email","variant":"V0","seed":0})
        out=self.root/"runs"/"job"
        queue=self.root/"protocols"/"queue.json"
        rows=[{"id":f"job-{i}","phase":"training","scope":"development",
            "args":["--config",str(cfg)],"output_dir":str(out),"dataset":"email","variant":"V0","seed":0}
            for i in range(80)]
        qsha=write_json(queue,{"jobs":rows})
        acceptance=self.root/"protocols"/"acceptance.json"
        write_json(acceptance,{"accepted_by":"root","verdict":"pass","queue_path":str(queue),
            "queue_sha256":qsha,"jobs":rows})
        old_registry=batch.REGISTRY
        batch.REGISTRY=(("fixture","protocols/queue.json","protocols/acceptance.json",80),)
        try:
            result=batch.preflight_campaign(self.root)
        finally:
            batch.REGISTRY=old_registry
        self.assertEqual(result[0]["jobs"][0]["phase"],"training")
        self.assertEqual(result[0]["jobs"][0]["scope"],"development")

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

    def _campaign_queue(self):
        queue_path = self.root / "protocols" / "toy-campaign-queue.json"
        acceptance_path = self.root / "protocols" / "toy-campaign-acceptance.json"
        qsha = write_json(queue_path, {"scope": "toy normalized queue", "jobs": ["toy-job", "pending-job"]})
        acceptance_sha = write_json(acceptance_path, {"status": "accepted", "queue_sha256": qsha})
        self.job["queue_path"] = str(queue_path)
        self.job["queue_sha256"] = qsha
        self.job["acceptance_path"] = str(acceptance_path)
        self.job["acceptance_sha256"] = acceptance_sha
        receipt_path = self.root / "logs" / "offline-phase-wall" / "toy-job.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["queue_sha256"] = qsha
        write_json(receipt_path, receipt)
        pending = dict(self.job, id="pending-job", output_dir=str(self.root / "runs" / "pending"), seed=1)
        return {"key": "fixture", "path": str(queue_path), "sha256": qsha,
                "acceptance": {"path": str(acceptance_path), "sha256": acceptance_sha, "scope": "accepted"},
                "jobs": [self.job, pending]}

    def _run_campaign(self, tmp_output, queue, reuse_path, *, mutation=None):
        original_audit = batch.audit_job
        def audit_with_fixture_verifier(root, job, numeric_fn=None, reuse_path=None):
            return original_audit(root, job, numeric_fn=numeric_fn, reuse_path=reuse_path,
                                  binding_verifier=self.binding_verifier)
        batch.audit_job = audit_with_fixture_verifier
        def reuse(job):
            if mutation == "queue" and job["id"] == "toy-job":
                Path(queue["path"]).write_text('{"mutated":true}\n', encoding="utf-8")
            if mutation == "acceptance" and job["id"] == "toy-job":
                Path(queue["acceptance"]["path"]).write_text('{"mutated":true}\n', encoding="utf-8")
            if mutation == "config" and job["id"] == "toy-job":
                Path(job["config_path"]).write_text('{"changed":true}\n', encoding="utf-8")
            if mutation == "source" and job["id"] == "toy-job":
                Path(batch.__file__).write_bytes(Path(batch.__file__).read_bytes() + b"\n# mutated fixture\n")
            return reuse_path if job["id"] == "toy-job" else None
        try:
            batch.preflight_campaign = lambda root: [queue]
            impossible_numeric = lambda *_: (_ for _ in ()).throw(
                AssertionError("completed fixture must reuse its saved validation report"))
            return batch.run_campaign(self.root, tmp_output, numeric_fn=impossible_numeric,
                                      reuse_resolver=reuse)
        finally:
            batch.audit_job = original_audit

    def test_public_run_campaign_uses_nested_acceptance_and_persists_completed_plus_pending(self):
        queue = self._campaign_queue()
        reuse_path = self.root / "logs" / "saved-validation-numeric.json"
        write_json(reuse_path, self.numeric_stub(self.config_path, self.out))
        result_path = self.root / "logs" / "campaign-audit.json"
        result = self._run_campaign(result_path, queue, reuse_path)
        self.assertEqual(result["status"], "completed_with_pending_or_failed_rows")
        self.assertEqual(result["counts"], {"reused": 1, "audited": 0, "pending": 1, "failed": 0})
        self.assertEqual([row["status"] for row in result["rows"]], ["reused", "pending"])
        saved = json.loads(result_path.read_text())
        self.assertEqual(saved["counts"], result["counts"])
        self.assertFalse(saved["scored_test_origins"])

    def test_public_run_campaign_rechecks_queue_acceptance_config_and_source(self):
        for mutation in ("queue", "acceptance", "config", "source"):
            with self.subTest(mutation=mutation):
                queue = self._campaign_queue()
                reuse_path = self.root / f"logs/saved-{mutation}-numeric.json"
                write_json(reuse_path, self.numeric_stub(self.config_path, self.out))
                output = self.root / f"logs/campaign-{mutation}.json"
                old_file, old_numeric = batch.__file__, batch.NUMERIC_AUDITOR
                temp_source = self.root / f"fixture-auditor-{mutation}.py"
                temp_numeric = self.root / f"fixture-numeric-{mutation}.py"
                temp_source.write_bytes(Path(old_file).read_bytes())
                temp_numeric.write_bytes(Path(old_numeric).read_bytes())
                batch.__file__, batch.NUMERIC_AUDITOR = str(temp_source), temp_numeric
                try:
                    with self.assertRaises(batch.CampaignAuditError):
                        self._run_campaign(output, queue, reuse_path, mutation=mutation)
                finally:
                    batch.__file__, batch.NUMERIC_AUDITOR = old_file, old_numeric
                self.assertFalse(output.exists())

    def test_public_run_campaign_refuses_existing_output(self):
        queue = self._campaign_queue()
        reuse_path = self.root / "logs" / "saved-output-numeric.json"
        write_json(reuse_path, self.numeric_stub(self.config_path, self.out))
        output = self.root / "logs" / "campaign-existing.json"
        self._run_campaign(output, queue, reuse_path)
        before = output.read_bytes()
        with self.assertRaises(batch.CampaignAuditError):
            self._run_campaign(output, queue, reuse_path)
        self.assertEqual(output.read_bytes(), before)

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

    def test_host_recovery_projection_is_exact_80_logical_rows_12_plus_68(self):
        original=[];recovery=[]
        logical_ids=[f"accepted-{i:02d}" for i in range(12)]+["internal-bound-v4-email-V0-seed0"]+[f"logical-{i:02d}" for i in range(13,80)]
        accepted=set(logical_ids[:12])
        for i in range(80):
            logical=logical_ids[i]
            old_cfg=self.root/"configs"/f"old-{i}.json";out=self.root/"runs"/f"old-{i}"
            config={"variant":"V0","seed":0,"lr":0.001,"output_dir":str(out)}
            write_json(old_cfg,config)
            original.append({"id":logical,"queue_key":"source","queue_path":str(self.root/"source.json"),
                "queue_sha256":"a"*64,"acceptance_path":str(self.root/"accept.json"),"acceptance_sha256":"b"*64,
                "config_path":str(old_cfg),"config_sha256":batch.sha256(old_cfg),"config":config,
                "output_dir":str(out),"dataset":"email","variant":"V0","seed":0,"phase":"training","scope":"development"})
            if logical in accepted:continue
            new_cfg=self.root/"configs"/f"new-{i}.json";new_out=self.root/"runs"/f"new-{i}"
            new_config={**config,"output_dir":str(new_out)};write_json(new_cfg,new_config)
            physical=("host-recovery-v2-internal-bound-v4-email-V0-seed0" if i==12
                      else "coverage-recovery-v1-"+logical)
            recovery.append({"id":physical,"phase":"training","scope":"development","args":["--config",str(new_cfg)],
                "output_dir":str(new_out),"inputs":[],"_queue_sha256":"c"*64,"_acceptance_sha256":"d"*64})
        mapped,pairs=batch.project_recovery_jobs(original,recovery,accepted,self.root)
        self.assertEqual(len(mapped),68);self.assertEqual(len(pairs),68)
        self.assertEqual({r["logical_job_id"] for r in mapped},set(logical_ids[12:]))
        self.assertEqual(mapped[0]["id"],"host-recovery-v2-internal-bound-v4-email-V0-seed0")
        recovery[0]["id"]="coverage-recovery-v1-internal-bound-v4-email-V0-seed0"
        with self.assertRaises(batch.CampaignAuditError) as raised:
            batch.project_recovery_jobs(original,recovery,accepted,self.root)
        self.assertIn("frozen recovery mapping",str(raised.exception))

    def test_host_recovery_projection_rejects_stale_alias_or_config_drift(self):
        old_cfg=self.root/"old.json";new_cfg=self.root/"new.json";oldout=self.root/"runs"/"old";newout=self.root/"runs"/"new"
        base={"variant":"V0","seed":0,"lr":0.001,"output_dir":str(oldout)}
        write_json(old_cfg,base);changed={**base,"lr":0.2,"output_dir":str(newout)};write_json(new_cfg,changed)
        original={"id":"internal-bound-v4-email-V0-seed0","queue_key":"source","queue_path":str(self.root/"source.json"),
          "queue_sha256":"a"*64,"acceptance_path":str(self.root/"accept.json"),"acceptance_sha256":"b"*64,
          "config_path":str(old_cfg),"config_sha256":batch.sha256(old_cfg),"config":base,"output_dir":str(oldout),
          "dataset":"email","variant":"V0","seed":0,"phase":"training","scope":"development"}
        row={"id":"coverage-recovery-v1-internal-bound-v4-email-V0-seed0","phase":"training","scope":"development",
          "args":["--config",str(new_cfg)],"output_dir":str(newout),"inputs":[],"_queue_sha256":"c"*64,"_acceptance_sha256":"d"*64}
        with self.assertRaises(batch.CampaignAuditError):
            batch.project_recovery_jobs([original],[row],set(),self.root,require_full_campaign=False)

    def test_campaign_result_rows_follow_original_logical_order(self):
        originals=[{"id":"first"},{"id":"second"},{"id":"third"}]
        by_id={"third":{"logical_job_id":"third"},"first":{"logical_job_id":"first"},
               "second":{"logical_job_id":"second"}}
        self.assertEqual([r["logical_job_id"] for r in batch.order_logical_rows(originals,by_id)],
                         ["first","second","third"])
        with self.assertRaises(batch.CampaignAuditError):
            batch.order_logical_rows(originals,{"first":by_id["first"]})

    def _historical_ledger_fixture(self):
        ledger_rel="protocols/training-budget-manual-ledger.json"
        frozen_rel="protocols/training-budget-manual-ledger-before-host-repair-v2.json"
        prior_rel=batch.PREVIOUS_RECOVERY_ACCEPTANCE_REL
        live_sha=put(self.root/ledger_rel,b"current ledger v2")
        old_bytes=b"old ledger v1 exact bytes"
        old_sha=hashlib.sha256(old_bytes).hexdigest()
        put(self.root/frozen_rel,old_bytes)
        other_rel="protocols/prior-other-evidence.json"
        other_sha=write_json(self.root/other_rel,{"immutable":True})
        prior={"accepted_by":"root","verdict":"pass","scope":"P3 baseline coverage recovery admission",
               "pipeline_advanced":False,"scientific_hypothesis_accepted":False,"test_accessed":False,
               "test_access_authorized":False,"evidence_sha256":{ledger_rel:old_sha,other_rel:other_sha}}
        prior_sha=write_json(self.root/prior_rel,prior)
        queue_path=self.root/batch.RECOVERY_QUEUE_REL
        queue_sha=write_json(queue_path,{"jobs":[{"id":f"job-{i}"} for i in range(68)]})
        host={"accepted_by":"root","verdict":"pass","scope":"P3 baseline host recovery admission",
              "queue_path":batch.RECOVERY_QUEUE_REL,"queue_sha256":queue_sha,"job_count":68,
              "test_access_authorized":False,"evidence_sha256":{
                  prior_rel:prior_sha,frozen_rel:old_sha,ledger_rel:live_sha,other_rel:other_sha}}
        host_path=self.root/batch.RECOVERY_ACCEPTANCE_REL
        write_json(host_path,host)
        return queue_path,host_path,prior_rel,frozen_rel,other_rel

    def test_historical_ledger_uses_only_host_accepted_exact_snapshot(self):
        queue,host,prior,frozen,_=self._historical_ledger_fixture()
        authority=batch._verified_historical_ledger_authority(self.root,queue,host,batch.sha256(queue))
        self.assertEqual(authority["historical_ledger_path"],frozen)
        record=batch._verify_root_evidence_record(self.root,prior,
            expected_scope="P3 baseline coverage recovery admission",
            historical_ledger_authority=authority)
        self.assertEqual(record["path"],str((self.root/prior).resolve()))

    def test_historical_ledger_rejects_wrong_snapshot_mapping(self):
        queue,host,prior,_,_=self._historical_ledger_fixture()
        doc=json.loads(host.read_text())
        doc["evidence_sha256"]["protocols/training-budget-manual-ledger-before-host-repair-v2.json"]="0"*64
        write_json(host,doc)
        with self.assertRaises(batch.CampaignAuditError):
            batch._verified_historical_ledger_authority(self.root,queue,host,batch.sha256(queue))

    def test_historical_ledger_does_not_waive_any_other_old_pin(self):
        queue,host,prior,_,other=self._historical_ledger_fixture()
        authority=batch._verified_historical_ledger_authority(self.root,queue,host,batch.sha256(queue))
        put(self.root/other,b"changed unrelated evidence")
        with self.assertRaises(batch.CampaignAuditError):
            batch._verify_root_evidence_record(self.root,prior,
                expected_scope="P3 baseline coverage recovery admission",
                historical_ledger_authority=authority)


if __name__ == "__main__":
    unittest.main()
