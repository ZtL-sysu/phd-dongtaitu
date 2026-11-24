from __future__ import annotations
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'edsgn'/'scripts';STAGING=ROOT/'edsgn'/'staging'
sys.path[:0]=[str(STAGING),str(ROOT/'edsgn'/'src'),str(SCRIPT)]
import extend_robustness_freeze_bundle_v1 as bundle
from edsgn.robustness import standard_missingness_recipes, recipe_digest, robustness_freeze_extras


def test_predeclared_twelve_recipes_are_authorized_as_one_bundle(monkeypatch):
    monkeypatch.setattr(bundle,'robustness_implementation_sha256',lambda:'a'*64)
    recipes=standard_missingness_recipes(seed=2027)
    extra=bundle.robustness_freeze_bundle_extras(recipes)
    expected=sorted({recipe_digest(r) for r in recipes})
    assert extra['allowed_robustness_recipe_digests']==expected and len(expected)==12
    assert extra['allowed_robustness_implementation_sha256']=='a'*64
    for recipe in recipes:
        fields=bundle.robustness_manifest_authorization_fields(extra,recipe,implementation_sha256='a'*64)
        assert fields['selected_authorized_robustness_recipe_digest']==recipe_digest(recipe)
        assert fields['frozen_allowed_robustness_recipe_digests']==expected


def test_bundle_deduplicates_sorts_and_rejects_nonmember_seed_or_recipe(monkeypatch):
    monkeypatch.setattr(bundle,'robustness_implementation_sha256',lambda:'b'*64)
    recipes=standard_missingness_recipes(seed=11)
    extra=bundle.robustness_freeze_bundle_extras([recipes[4],recipes[1],recipes[4]])
    assert extra['allowed_robustness_recipe_digests']==sorted({recipe_digest(recipes[1]),recipe_digest(recipes[4])})
    with pytest.raises(RuntimeError,match='does not authorize'):
        bundle.validate_robustness_bundle_extra(extra,standard_missingness_recipes(seed=12)[4],implementation_sha256='b'*64)
    valid_extra=dict(recipes[0],event_time_jitter_bins=1)
    with pytest.raises(RuntimeError,match='does not authorize'):
        bundle.validate_robustness_bundle_extra(extra,valid_extra,implementation_sha256='b'*64)
    outsider=dict(recipes[0]);outsider['missing']=dict(recipes[0]['missing'],rate=0.2)
    with pytest.raises(ValueError):bundle.robustness_freeze_bundle_extras([outsider])


def test_bundle_validation_rejects_ambiguous_empty_and_malformed_authorization(monkeypatch):
    monkeypatch.setattr(bundle,'robustness_implementation_sha256',lambda:'c'*64)
    recipe=standard_missingness_recipes()[0];one=recipe_digest(recipe)
    base={'allowed_robustness_implementation_sha256':'c'*64}
    malformed=[
      {**base,'allowed_robustness_recipe_digest':one,'allowed_robustness_recipe_digests':[one]},
      {**base,'allowed_robustness_recipe_digests':[]},
      {**base,'allowed_robustness_recipe_digests':[1]},
      {**base,'allowed_robustness_recipe_digests':['not-a-sha']},
      {**base,'allowed_robustness_recipe_digests':[one,one]},
      {**base,'allowed_robustness_recipe_digests':[one],'allowed_robustness_recipe_digest':None},
    ]
    for extra in malformed:
      with pytest.raises(RuntimeError):bundle.validate_robustness_bundle_extra(extra,recipe,implementation_sha256='c'*64)
    with pytest.raises(RuntimeError,match='implementation SHA-256'):
      bundle.validate_robustness_bundle_extra({**base,'allowed_robustness_recipe_digests':[one]},recipe,implementation_sha256='d'*64)
    with pytest.raises(RuntimeError,match='implementation digest'):
      bundle.validate_robustness_bundle_extra({'allowed_robustness_recipe_digests':[one]},recipe,implementation_sha256='c'*64)


def test_legacy_single_recipe_freeze_remains_compatible(monkeypatch):
    monkeypatch.setattr(bundle,'robustness_implementation_sha256',lambda:'e'*64)
    recipe=standard_missingness_recipes()[3]
    legacy=bundle.robustness_freeze_extras(recipe)
    assert set(legacy)=={'allowed_robustness_recipe_digest','allowed_robustness_implementation_sha256'}
    assert legacy['allowed_robustness_recipe_digest']==recipe_digest(recipe)
    assert legacy['allowed_robustness_implementation_sha256']=='e'*64
    fields=bundle.validate_robustness_bundle_extra(legacy,recipe,implementation_sha256='e'*64)
    assert fields['selected_authorized_robustness_recipe_digest']==recipe_digest(recipe)
    assert fields['frozen_allowed_robustness_recipe_digests']==[recipe_digest(recipe)]
    with pytest.raises(RuntimeError,match='does not authorize'):
        bundle.validate_robustness_bundle_extra(legacy,standard_missingness_recipes()[4],implementation_sha256='e'*64)
