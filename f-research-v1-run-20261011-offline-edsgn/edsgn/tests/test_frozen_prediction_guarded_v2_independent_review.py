from pathlib import Path
import importlib.util,json
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("writer_v2_tests",ROOT/"edsgn/tests/test_frozen_prediction_guarded_v2.py")
writer_tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer_tests)
bridge=writer_tests.bridge
engine=writer_tests.engine

def test_registry_exact_families_and_design_axes():
 doc=json.loads((ROOT/"protocols/primary-endpoint-registry-root-preparation-v2.json").read_text(encoding="utf-8"))
 rows=doc["primary_endpoints"]
 assert doc["family_sizes"]=={"H1":9,"H2":16,"H3":24}
 assert len(rows)==49 and len({r["endpoint_id"] for r in rows})==49
 assert all(r["primary"] is True for r in rows)
 assert doc["split"]=="test"
 for claim,n in (("H1",9),("H2",16),("H3",24)):
  group=[r for r in rows if r["claim"]==claim]
  assert len(group)==n and all(r["holm_family"]==claim for r in group)
  assert all(r["seed_ids"]==[0,1,2,3,4] for r in group)
  assert all(r["binding"]["input_plan_path"] is None for r in group)
 h1=[r for r in rows if r["claim"]=="H1"]
 assert sum(r["outcome_type"]=="graph_tv_noise_superiority" for r in h1)==1
 assert sum(r["outcome_type"]=="forecast_mae_noninferiority" for r in h1)==4
 assert sum(r["outcome_type"]=="graph_tv_response_superiority" for r in h1)==4
 assert {r["synthetic_case"] for r in h1}=={"noise-only","identity10","identity30","weight10","weight30"}
 h2=[r for r in rows if r["claim"]=="H2"]
 assert {r["synthetic_case"] for r in h2}=={"identity10","identity30","weight10","weight30"}
 assert all(r["outcome_type"]=="forecast_mae_superiority" and r["required_horizons"]==[1,3,7] and r["slice"]=="post4L" for r in h2)
 h3=[r for r in rows if r["claim"]=="H3"]
 assert len([r for r in h3 if r["domain"]=="email"])==12 and len([r for r in h3 if r["domain"]=="mooc"])==12
 for domain,horizons in (("email",[1,3,7]),("mooc",[1,6,24])):
  for device in ("cpu","cuda"):
   cell=[r for r in h3 if r["domain"]==domain and r["device"]==device]
   assert len(cell)==6 and {r["right"] for r in cell}=={"V4","V5","V6"}
   assert all(r["required_horizons"]==horizons and r["slice"]=="full_split" for r in cell)
 assert doc["artifact_status"]=="root_structural_preparation_not_frozen_not_executed"
 assert doc["bootstrap"]["reps"]==2000 and doc["bootstrap"]["random_seed"]==1234

def test_real_cpu_engine_then_acceptance_mutation_fails_and_releases_leases(tmp_path,monkeypatch):
 root,data,ckpt,cfg,binding,policy,payload,policy_cfg=writer_tests._fixture(tmp_path,monkeypatch,"periodic")
 original=engine.evaluate_run
 acceptance=root/"reviews/blind-reference-cost-policy-root-acceptance-v1.json"
 def mutate_after_engine(*args,**kwargs):
  result=original(*args,**kwargs)
  doc=json.loads(acceptance.read_text(encoding="utf8"));doc["scope"]="stale-after-prediction"
  acceptance.write_text(json.dumps(doc),encoding="utf8")
  return result
 monkeypatch.setattr(engine,"evaluate_run",mutate_after_engine)
 output=root/"runs/independent-mutation"
 with pytest.raises(RuntimeError,match="Bound source/input hashes changed"):
  bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=output,split="test",policy_kind="periodic",policy_config_path=policy_cfg,device="cpu",root=root,budget_fn=writer_tests._budget,process_scan=lambda:[])
 receipt=json.loads(next((root/"logs/offline-phase-wall").glob("frozen-prediction-*.json")).read_text())
 assert receipt["status"]=="failed" and receipt["test_accessed"] is False  # observed receipt underreports test access after engine execution
 assert not (root/"logs/model-resource.lock").exists()
 assert not (root/"logs/offline-phase-wall/campaign.lock").exists()
 assert (output/"predictions.npz").is_file()  # failure remains auditable; never relabeled completed


def test_cpu_engine_output_mutation_after_export_is_not_bound_by_engine_manifest(tmp_path,monkeypatch):
 root,data,ckpt,cfg,binding,policy,payload,policy_cfg=writer_tests._fixture(tmp_path,monkeypatch,"full")
 original=engine.evaluate_run
 def mutate_predictions_after_export(*args,**kwargs):
  result=original(*args,**kwargs)
  path=root/"runs/mutable/predictions.npz"
  with np.load(path,allow_pickle=False) as z:
   arrays={k:z[k].copy() for k in z.files}
  arrays["predictions"]+=123.0
  np.savez_compressed(path,**arrays)
  return result
 import numpy as np
 monkeypatch.setattr(engine,"evaluate_run",mutate_predictions_after_export)
 out=root/"runs/mutable"
 bridge.run_prediction(checkpoint=ckpt,dataset_path=data,binding_path=binding,output_dir=out,split="test",policy_kind="full",device="cpu",root=root,budget_fn=writer_tests._budget,process_scan=lambda:[])
 receipt=json.loads(next((root/"logs/offline-phase-wall").glob("frozen-prediction-*.json")).read_text())
 with np.load(out/"predictions.npz",allow_pickle=False) as archive:
  assert np.max(archive["predictions"])>100
 assert receipt["status"]=="completed"  # current bridge has no freeze/manifest digest for prediction bytes


def test_root_acceptance_validator_does_not_enforce_non_scientific_status_flags(tmp_path,monkeypatch):
 root,data,ckpt,cfg,binding,policy,payload,policy_cfg=writer_tests._fixture(tmp_path,monkeypatch,"periodic")
 path=root/"reviews/blind-reference-cost-policy-root-acceptance-v1.json"
 doc=json.loads(path.read_text(encoding="utf8"));doc.update(scientific_hypothesis_accepted=True,pipeline_advanced=True,c07_experiment_closed=True)
 path.write_text(json.dumps(doc),encoding="utf8")
 accepted=bridge._validate_blind_acceptance(root,root/"edsgn/staging/blind_reference_cost_policy_v1.py")
 assert str(path.resolve()) in accepted  # current validator accepts flags contradicting engineering-only scope
