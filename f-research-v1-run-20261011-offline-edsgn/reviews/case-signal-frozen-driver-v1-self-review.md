# Case-signal frozen driver v1 — writer engineering self-review

This is an implementation self-review, not an independent review or scientific acceptance. The driver is a staging CPU-toy prototype for the case-signal sensitivity in `protocols/case-slices-and-signal-sensitivity-root-design-v1.md`.

The prototype runs the accepted case-signal kernel v2 with the accepted H2 graph-retention helper, preserving the frozen full-policy teacher and two intervention arms (relationship removal and focal retention removal). It writes three prediction archives (teacher, relationship-removal, focal-retention-removal), per-horizon/per-feature metrics, a semantic proof JSONL covering every origin/history step, an evaluation manifest, report, and output-closure receipt. The kernel receives only x, mask, age, raw, and event priors; target arrays are collected outside the kernel. It verifies fixed parameter bytes, informative-teacher parity, exact retention graph/actions, full-origin coverage, output hashes, and cumulative-budget guards.

Validation: the six focused CPU toy tests pass (`logs/case-signal-frozen-driver-v1-tests.log`). The end-to-end toy uses the actual `PreparedData`, engine checkpoint/freeze path, and accepted numerical kernel. Tests also reject an unaccepted selection, a node-ID/source mismatch, public launch without the managed inventory, and a budget that expires at the final guard. No real collection, checkpoint, model run, GPU, optimizer, or test outcome was accessed.

The public CLI intentionally fails closed. The required selection document currently uses a proposed schema (`root_accepted_case_signal_validation_selection`) and is not yet integrated with an accepted producer. More importantly, the complete managed dispatcher/global case-job inventory and its global completion-before-oracle closure are unfinished. Therefore this driver is not launchable for real data and does not establish P3A readiness, case selection acceptance, scientific results, or a hypothesis claim. Dataset coverage beyond the CPU fixture is also pending.

Source pins: driver SHA-256 `869daf224e2cdafbf74ed14b60855776b0a09dcf37cbb378a51495e1a14ffacd`; case kernel v2 `a49f823c...c68b5`; H2 helper v2 `2bb4d80f...35d91`; root engineering receipt `54c0282e...84065`. Focused tests: 6 passed.

