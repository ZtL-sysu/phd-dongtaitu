# P3 campaign auditor v3 — implementation self-review

This is the writer's narrow successor self-review; v2 remains unchanged. V3 adds a fail-closed check that every completed training phase receipt directly lists the exact requested config path and SHA among its input hashes. A queue declaration or launch binding alone no longer substitutes for explicit phase-receipt membership. This is in addition to full resolved-config equality and the launch-binding path/content/hash checks introduced in v2.

The regression fixture removes the config from an otherwise valid completed phase receipt and confirms the audit rejects it. All seven focused CPU tests pass. Metadata preflight still verifies the six accepted queue snapshots and 80 unique jobs. The live campaign remains unexecuted; there is no claim of numeric acceptance, P3 completion, or scientific result.

Evidence: [preflight](../logs/p3-campaign-auditor-v3-preflight.json), [tests](../logs/p3-campaign-auditor-v3-tests.log). Source SHA-256: `a03ba0994df6eb1d81383827d51d52b44bfa6a4a0fe7f9466d26211c83593479`; test SHA-256: `c6b70aa6f467515888846c698774a21cedc7d782ae043359ca957f3cd5c5cc4f`.
