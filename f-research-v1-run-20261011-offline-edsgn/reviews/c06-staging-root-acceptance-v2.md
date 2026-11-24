# C06 split-history wrapper: root staging acceptance

The isolated v2 successor fixes the six independent v1 review findings:
weighted complete origin-history coverage, mask-dependent QC counts, actual
executing wrapper source binding, transitive robustness.py source binding,
strict domain identity/schema checks and strict trajectory-index typing.
Root inspected the weighted sums/domain reconstruction, and core_semantic_tests
independently reviewed v2 and reran the12 CPU toy checks successfully.

Independent evidence is reviews/split-history-robustness-external-review-v2.md.
The file named split-history-robustness-independent-v2.md is the v2 implementer's
self-assessment, rather than an independent review; its role is recorded here
explicitly. V1 implementation/reviews/failures remain preserved.

Accept edsgn/staging/split_history_robustness_v2.py as engineering preparation
only. Before actual C06 replay, create and review a real immutable domain
manifest using the same ordered collection throughout freeze, construction and
validation; bind complete original data/preprocessing fingerprints as well as
the domain/mask identities. Integrate these validations into a separately
gated successor evaluator with frozen weights/policies/recipes, complete origins,
serial resource ownership and the same cumulative72-hour budget.

C06 actual evidence repair remains open. No real perturbed replay, robustness
result, P3A/P4 gate or scientific acceptance is inferred from these toy passes.
The active baseline package/controller/input hashes are unchanged.
