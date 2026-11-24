# P3 campaign auditor v7 engineering self-review

Scope is metadata-only engineering, not independent review or experiment acceptance.

The v7 successor makes one exact historical substitution: the v1 coverage-recovery acceptance’s old `protocols/training-budget-manual-ledger.json` pin is satisfied only by the v2 root-accepted preserved snapshot with the same SHA. The v2 acceptance is fully revalidated against the current queue and all current evidence, including the current ledger; every other prior evidence pin is still rehashed. A regression test rejects a wrong snapshot mapping and a changed unrelated old pin. The recovery projection also retains original phase/scope fields so the frozen recovery row can be matched exactly.

A metadata-only preflight now constructs 80 canonical logical rows (12 prior accepted + 68 recovery mappings) and verifies two distinct stopped attempts separately: the first recorded budget stop remains returncode 1 / 6331.6217393 seconds; the second host-interrupted attempt remains returncode/finish null with 67.3734519 observed seconds plus the root-recorded 90.6265481 conservative supplement. No status or duration is inferred.

Conda CPU test suite: 19 passed plus 4 subtests; whole command 4.806565 s. The metadata preflight completed in 3.0259775 s and did not invoke numeric audit. Source SHA256 `d7bae9727e7027e0ecc0f7beebc44cbe2df9065e87ba6322474f6e514b749893`; tests SHA256 `729ba5314d704ac0e14ee1d610e7e6ab0b687ad4b29985f00873f2efe39fa1ac`. Detailed hashes and receipt pins are in the companion JSON.

No actual validation numeric audit, model/checkpoint load, target-array decode, GPU run, or scientific acceptance occurred.
