# Independent review: typed primary-family Holm v2

## Verdict

**Revise before using this kernel for claim assembly.** Its Holm arithmetic, family counts, endpoint claim/type checks, null handling, and directional gates passed independent pure-data checks. One structural field-validation gap remains: `assemble_primary_families` does not require `primary is True` or `holm_family == endpoint_id` prefix. I mutated a registered H1 endpoint to `primary: false, holm_family: H3`; the kernel still silently included it in H1. This permits a non-primary row or a misdeclared registered family to be treated as a primary endpoint. Require both fields and exact matches (or make this an explicit, separately hash-bound caller precondition that is independently enforced before invocation).

## Evidence

The root registry v2 has 49 unique primary rows, with H1/H2/H3 sizes 9/16/24. The operational contract calls for a single Holm adjustment over every registered member in its named claim family. Independent tests exercised those exact rows and verified complete denominators. A non-equal H1 example with raw p-values `.001,.004,.01,.02,.03,.04,.05,.06,.2` produced hand-calculated adjusted values `.009,.032,.07,.12,.15,.16,.16,.16,.2` in member order. False claim, disallowed H1 forecast-superiority type, and numeric/registry outcome-type mismatch were rejected.

An unidentifiable noninferiority endpoint with `raw_p: null` remains in the family, yields a null family denominator/decision and does not shrink adjustment. Identifiable rows require positive finite epsilon; the strict NI gate is `upper < epsilon`. An identifiable negative-support member remains distinguishable from unidentifiability and retains its p-value/adjustment. Superiority correctly requires both a negative estimate and a confidence upper bound strictly below zero.

The 12 existing writer tests passed separately; 7 independent tests passed. Logs: `logs/typed-primary-holm-v2-writer-tests.log` and `logs/typed-primary-holm-v2-independent-tests.log`. No production data, arrays, models, or phase outputs were read.

## Calibration v9 source-only check

`edsgn/staging/frozen_validation_calibration_v9.py` appears to enforce the intended chronology: canonical P2/P3/P3A passed/pass and P4 running are checked; P3A binds the queue, validation-only epsilon authority, and baseline-evidence hashes; the P3A authority rejects checkpoint/config/binding/freeze fields; only afterward must a distinct root-accepted `post-training-calibration-launch-binding` bind the trained checkpoint, config, queue, input binding, authority, and candidate IDs. It derives test-origin support from metadata only and compares that map to P3A, while calibration loads validation split objects and computes epsilon selection from validation. For the explicitly unidentifiable all-zero email path, it requires `primary_epsilon: null`, descriptive fallback exactly zero plus a reason; it subsequently verifies all valid validation targets are zero, retains preregistered `s00`, sets quality/noninferiority authorization false, and does not issue a primary epsilon. For a combined NPZ it candidly records `test_accessed: true` because the loader may materialize the bundle, while declaring no test origins or target/window index were evaluated; partitioned files select validation only. The final input/gate/checkpoint fences and stop-loss check precede completed status.

This is source inspection only, not a calibration v9 runtime verification or authorization to launch. I did not run its writer suite or read any actual validation/test arrays.

## Hashes

- Kernel `edsgn/staging/typed_primary_family_holm_v2.py`: `005EEF360B88864CBA1DA13A893AD817F5B7CD1A54BFAA791D673CFA715BC0CD`
- Root registry v2: `D2686C9BF4F1F0574A60774284A77E033B9E25D973F691C2B3E822381FDB476F`
- Endpoint definitions: `0F1BF0EC944701518882011964BC3AA42D9E136A7632059C7A319A03266C819E`
- Family assembly requirement: `C4D54BF1FEE9AB37260E480A9195D54DEDA5B8B449F7DDB8A7DA88571EE39EB5`
- Calibration v9 source inspected: `975505E0BF7CBFC80392D0C8259B7C8A1F79DE2B67284C32C63CF6F6DFB07554`
