# Independent narrow review: claim-family Holm v2

Scope: changed gate/hash/path branches only; the v1 review of Holm semantics and registry completeness remains applicable. No actual reports, predictions, heldout arrays, or model artifacts were read.

The v2 delta closes the v1 defect. In the validation branch, P3A status `passed` now requires `gate_verdict=pass` and P4 running; both P3A `running` and `passed` states rehash every ledger input/evidence entry. `_resolve` rejects consumed artifacts outside RUN after canonical resolution, and the artifact plan/report paths use it. This closes the prior acceptance of failed P3A verdicts and stale evidence and prevents path traversal out of RUN.

Targeted Conda CPU tests: `pytest -q test_claim_family_holm_v2.py -k 'validation_gate or consumed_paths'` — 5 passed, 7 deselected. They cover the failed verdict, stale evidence under both statuses, and plan/report paths escaping RUN. No actual global assembly was run. Verdict: v2 changes pass narrow staging review.

Hashes: source `4c3330cb553814d4b8f3307665bce9169928f79019e950bae813913da8fc13dc`; tests `307ece4709b5cc85b593b6fd334c397f7e5fb48b0446d6c137005a2019dca8ff`.
