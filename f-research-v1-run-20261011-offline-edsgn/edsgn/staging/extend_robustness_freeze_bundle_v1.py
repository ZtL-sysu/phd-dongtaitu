"""Staging helpers for one immutable robustness freeze authorizing a recipe bundle.

Not imported by edsgn.robustness; root integration must explicitly review the delta.
"""
from __future__ import annotations
import re
from typing import Any, Sequence

from edsgn.robustness import canonical_recipe, recipe_digest, robustness_implementation_sha256

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"Frozen robustness {label} must be a lowercase SHA-256 digest")
    return value


def robustness_freeze_bundle_extras(recipes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Freeze a nonempty recipe bundle and the current perturbation implementation."""
    if isinstance(recipes, (str, bytes, dict)) or not isinstance(recipes, (list, tuple)) or not recipes:
        raise ValueError("robustness recipe bundle must be a nonempty list or tuple")
    digests = sorted({recipe_digest(canonical_recipe(recipe)) for recipe in recipes})
    return {
        "allowed_robustness_recipe_digests": digests,
        "allowed_robustness_implementation_sha256": robustness_implementation_sha256(),
    }


def validate_robustness_bundle_extra(
    extra: dict[str, Any], recipe: dict[str, Any], *, implementation_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate plural bundle or legacy single-recipe authorization, rejecting ambiguity."""
    if not isinstance(extra, dict):
        raise RuntimeError("Frozen manifest lacks robustness freeze extras")
    single_key = "allowed_robustness_recipe_digest"
    bundle_key = "allowed_robustness_recipe_digests"
    has_single = single_key in extra
    has_bundle = bundle_key in extra
    if has_single == has_bundle:
        raise RuntimeError("Freeze must contain exactly one single or recipe-bundle authorization")
    expected = recipe_digest(canonical_recipe(recipe))
    if has_single:
        allowed = [_digest(extra[single_key], "single recipe digest")]
        if allowed[0] != expected:
            raise RuntimeError("Frozen manifest does not authorize this exact robustness recipe digest")
    else:
        raw = extra[bundle_key]
        if not isinstance(raw, list) or not raw:
            raise RuntimeError("Frozen robustness recipe digest bundle must be a nonempty list")
        allowed = [_digest(item, "recipe digest bundle entry") for item in raw]
        if allowed != sorted(set(allowed)):
            raise RuntimeError("Frozen robustness recipe digest bundle must be sorted and duplicate-free")
        if expected not in allowed:
            raise RuntimeError("Frozen manifest does not authorize this exact robustness recipe digest")
    frozen_impl = _digest(extra.get("allowed_robustness_implementation_sha256"), "implementation digest")
    current_impl = implementation_sha256 if implementation_sha256 is not None else robustness_implementation_sha256()
    _digest(current_impl, "current implementation digest")
    if frozen_impl != current_impl:
        raise RuntimeError("Frozen manifest does not authorize this exact robustness.py implementation SHA-256")
    return {
        "selected_authorized_robustness_recipe_digest": expected,
        "frozen_allowed_robustness_recipe_digests": allowed,
        "allowed_robustness_implementation_sha256": frozen_impl,
    }


def robustness_manifest_authorization_fields(
    extra: dict[str, Any], recipe: dict[str, Any], *, implementation_sha256: str | None = None,
) -> dict[str, Any]:
    """Fields to merge into each output manifest after freeze validation."""
    return validate_robustness_bundle_extra(extra, recipe, implementation_sha256=implementation_sha256)


def robustness_freeze_extras(recipe: dict[str, Any]) -> dict[str, str]:
    """Legacy single-recipe schema retained verbatim for existing callers."""
    return {
        "allowed_robustness_recipe_digest": recipe_digest(canonical_recipe(recipe)),
        "allowed_robustness_implementation_sha256": robustness_implementation_sha256(),
    }
