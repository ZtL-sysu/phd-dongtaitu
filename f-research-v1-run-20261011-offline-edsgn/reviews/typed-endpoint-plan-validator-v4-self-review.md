# Typed endpoint validator v4 — writer self-review

Scope: metadata-only staging validation. This is an implementation self-review, not an independent audit and not authorization to inspect held-out outcomes.

v4 preserves v3 and adds four fail-closed conditions: (1) `post4L` must fit inside each trajectory's legal origin support at both the switch lower bound and the four-history upper bound; (2) pretest and postrun block length and ACF dependence length must be positive; (3) all pretest execution-binding artifact hashes are included in the final consumed-file rehash barrier; and (4) each postrun dispatch receipt must prove that its BEFORE snapshot bound the exact pretest binding path and SHA, and bind the original completed child phase receipt plus stable alias.

Validation: `python -m pytest -q edsgn/tests/test_typed_endpoint_plan_validator_v4.py` — 14 passed. The postrun fixture now includes accepted dispatcher-v3 receipt fields and a targeted tamper case that updates its external manifest hashes yet still fails because the BEFORE snapshot omitted the binding. No actual endpoint data, predictions, or model outputs were read. Semantic producer checks and any scientific result interpretation remain external to this metadata validator.
