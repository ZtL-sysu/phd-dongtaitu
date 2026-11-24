# Independent review: global claim-family Holm staging tool

Scope: read-only source/protocol review and CPU toy tests. No real report, prediction, test/oracle arrays, model, optimizer, or GPU was accessed. No reviewed source was changed.

The assembler correctly uses only the raw registered forecast composite p-value pointer, groups values by H1/H2/H3, and calls Holm once per claim family. It excludes already adjusted stratum p-values from the family. It labels the output as not making a scientific claim. Its tests cover missing/extra/duplicate/reordered reports, duplicate endpoints, unsupported outcome types, adjusted pointers, invalid p-values/seeds, and TOCTOU report mutation. Endpoint completeness is checked against the frozen registry: every endpoint must have one matching report row, and every claim must have at least one endpoint. P3A must hash-bind the registry and all its stratum plans. This is not an independent check that the registry semantically contains every mechanism gate in the protocol; the output correctly makes no claim and says mechanism gates remain separate.

**Blocking gate defect:** `edsgn/staging/claim_family_holm_v1.py` lines 70–76 allow validation assembly when P3A is `passed` and P4 is `running` without requiring `P3A.gate_verdict == "pass"`. Unlike the test branch at lines 63–69, the validation branch does not check the verdict. A temp-only fixture with canonical P2/P3 pass, P3A `status="passed"` / `gate_verdict="fail"`, valid hash-bound registry/plans and the normal validation artifact fixture was accepted and returned H1/H2/H3 families. Require the passed verdict before post-P3A validation assembly; validation while P3A is still `running` can remain separately authorized.

Hashes: runner `5c4dbff461b1d70ac9593e0a69713ed4ef8c2686c7a765dcc4df6be39c9fdc72`; tests `bab0aee6bb73ba83aced2e64e378f8f9f568b15900277724e1e5e82a136e5107`.

Verification: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest -q edsgn/staging/test_claim_family_holm_v1.py` — 7 passed. This is staging-only; no global statistical assembly or P3A acceptance occurred.

