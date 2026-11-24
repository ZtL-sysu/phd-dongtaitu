from __future__ import annotations
import hashlib, importlib.util, json, os, sys, types
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
STAGING=ROOT/'edsgn'/'staging';SRC=ROOT/'edsgn'/'src'
PRE_BUNDLE_SOURCE=ROOT/'snapshots'/'robustness-before-bundle-integration-v1.py'
sys.path[:0]=[str(STAGING),str(SRC)]
import integrate_robustness_bundle_v1 as integration
import extend_robustness_freeze_bundle_v1 as bundle
from edsgn.robustness import standard_missingness_recipes, recipe_digest

class EmptyPsutil:
  NoSuchProcess=type('NSP',(Exception,),{});AccessDenied=type('AD',(Exception,),{})
  @staticmethod
  def process_iter(_):return []


def fixture_root(tmp_path):
  target=tmp_path/integration.TARGET_REL;target.parent.mkdir(parents=True);source=PRE_BUNDLE_SOURCE.read_text(encoding='utf-8');target.write_text(source,encoding='utf-8')
  snapshot=tmp_path/integration.SNAPSHOT_REL;snapshot.parent.mkdir(parents=True);snapshot.write_text(source,encoding='utf-8')
  (tmp_path/'logs'/'offline-phase-wall').mkdir(parents=True)
  state={'stages':[{'id':'P1B','status':'passed'},{'id':'P2','status':'pending'},{'id':'P3','status':'pending'}]}
  (tmp_path/'pipeline-state.json').write_text(json.dumps(state),encoding='utf-8')
  return target,source


def generated_module(source):
  name='edsgn.robustness_generated_bundle_test';module=types.ModuleType(name);module.__package__='edsgn';module.__file__=str(SRC/'edsgn'/'robustness.py')
  exec(compile(source,module.__file__,'exec'),module.__dict__)
  return module


def test_plan_preflight_accepts_only_required_pending_state_and_is_nonmutating(tmp_path):
  target,source=fixture_root(tmp_path);before=hashlib.sha256(target.read_bytes()).hexdigest()
  plan=integration.build_plan(tmp_path,EmptyPsutil)
  assert plan['original_sha256']==before and plan['snapshot_sha256']==before
  assert plan['planned_sha256']==hashlib.sha256(plan['planned_source'].encode()).hexdigest()
  assert 'allowed_robustness_recipe_digests' in plan['planned_source']
  assert 'frozen_allowed_robustness_recipe_digests' in plan['planned_source']
  assert target.read_text(encoding='utf-8')==source


def test_preflight_rejects_wrong_gate_and_existing_resource_lock(tmp_path):
  fixture_root(tmp_path)
  state=json.loads((tmp_path/'pipeline-state.json').read_text());state['stages'][1]['status']='passed';(tmp_path/'pipeline-state.json').write_text(json.dumps(state))
  with pytest.raises(integration.PreflightError,match='P2 must be pending'):integration.preflight(tmp_path,EmptyPsutil)
  state['stages'][1]['status']='pending';(tmp_path/'pipeline-state.json').write_text(json.dumps(state))
  lock=tmp_path/'logs'/'model-resource.lock';lock.write_text('active')
  with pytest.raises(integration.PreflightError,match='resource/campaign lock'):integration.preflight(tmp_path,EmptyPsutil)


def test_process_preflight_finds_relative_runner_from_root_cwd(tmp_path,monkeypatch):
  monkeypatch.setattr(integration.os,'getpid',lambda:77)
  class Proc:
    def __init__(self,pid,argv):self.pid=pid;self.info={'pid':pid,'cmdline':argv,'cwd':str(tmp_path),'exe':'C:/python.exe'}
  class Fake(EmptyPsutil):
    @staticmethod
    def process_iter(_):return [Proc(77,['python','-m','edsgn.train']),Proc(78,['python','edsgn/scripts/baseline_campaign.py'])]
  found=integration.owned_model_processes(tmp_path,Fake)
  assert [item['pid'] for item in found]==[78]


def test_generated_validator_accepts_bundle_and_legacy_single(monkeypatch,tmp_path):
  source=PRE_BUNDLE_SOURCE.read_text(encoding='utf-8');helper=integration.HELPER.read_text(encoding='utf-8')
  module=generated_module(integration.generate_source(source,helper));import edsgn.engine as engine
  monkeypatch.setattr(engine,'validate_freeze',lambda *args:{'extra':freeze['extra']})
  recipes=standard_missingness_recipes();freeze={'extra':bundle.robustness_freeze_bundle_extras(recipes)}
  for recipe in recipes:assert module.validate_robustness_freeze('ckpt',object(),{},None,recipe)['extra']==freeze['extra']
  freeze['extra']=bundle.robustness_freeze_extras(recipes[0])
  assert module.validate_robustness_freeze('ckpt',object(),{},None,recipes[0])['extra']==freeze['extra']
  with pytest.raises(RuntimeError,match='does not authorize'):
    module.validate_robustness_freeze('ckpt',object(),{},None,recipes[1])


def test_generated_evaluator_manifest_keeps_selected_legacy_field_and_full_bundle(monkeypatch,tmp_path):
  import torch
  from edsgn import engine, common
  source=PRE_BUNDLE_SOURCE.read_text(encoding='utf-8');helper=integration.HELPER.read_text(encoding='utf-8')
  module=generated_module(integration.generate_source(source,helper));recipes=standard_missingness_recipes();recipe=recipes[5]
  freeze={'extra':bundle.robustness_freeze_bundle_extras(recipes)}
  class Model(torch.nn.Module):pass
  class Data:horizons=(1,)
  class Wrapped:
    def __init__(self,data,recipe,**kwargs):self.perturbation_summary={'mocked':True}
  module.RobustnessPreparedData=Wrapped
  model=Model();ckpt={'config':{'dataset_path':'mock-do-not-read','variant':'V0','seed':9}}
  monkeypatch.setattr(engine,'load_checkpoint',lambda *a:(model,ckpt))
  monkeypatch.setattr(engine,'validate_freeze',lambda *a:freeze)
  monkeypatch.setattr(engine,'datasets_at',lambda *a:[Data()])
  def fake_evaluate(model,wrapped,split,device,**kwargs):
    assert split=='test' and isinstance(wrapped[0],Wrapped)
    return {'predictions':np.zeros((1,1,1,1),np.float32),'targets':np.ones((1,1,1,1),np.float32),'target_mask':np.ones((1,1,1,1),bool),'pairs':[(0,0)],'counter_rows':[]}
  monkeypatch.setattr(engine,'evaluate_batches',fake_evaluate);monkeypatch.setattr(engine,'metric_rows',lambda *a:[])
  monkeypatch.setattr(common,'parameter_hash',lambda model:'fixed-weight-hash')
  ckpt_path=tmp_path/'checkpoint.pt';ckpt_path.write_bytes(b'mocked-checkpoint');(tmp_path/'test-freeze.json').write_text('{}')
  out=tmp_path/'run';module.evaluate_frozen_robustness(ckpt_path,'test',out,recipe,device_name='cpu',batch_size=1)
  manifest=json.loads((out/'perturbation-manifest.json').read_text())
  full=sorted({recipe_digest(r) for r in recipes});selected=recipe_digest(recipe)
  assert manifest['allowed_robustness_recipe_digest']==selected
  assert manifest['selected_authorized_robustness_recipe_digest']==selected
  assert manifest['frozen_allowed_robustness_recipe_digests']==full and len(full)==12
  assert manifest['allowed_robustness_implementation_sha256']==freeze['extra']['allowed_robustness_implementation_sha256']
