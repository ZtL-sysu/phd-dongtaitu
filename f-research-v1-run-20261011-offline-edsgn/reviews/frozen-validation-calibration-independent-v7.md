# Independent narrow review: frozen validation calibration v7

**Verdict: engineering-only pass, with stated boundaries.** This does not accept P3A/P4 or any scientific result.

I inspected `edsgn/staging/frozen_validation_calibration_v7.py` and independently ran its CPU fixture suite with the dispatcher suite: **20 passed in 11.98s** under the local `dyg-cold` Conda Python, `PYTHONPATH=edsgn/src;edsgn/scripts;edsgn/staging`, OMP/MKL threads=2. The v7 tests include the real `calibrate_run` + `ForecastModel` + `PreparedData` path on a tiny generated bundle, with only the launch lease/receipt service mocked. The final cases inject cumulative-budget exhaustion on the third (finally) hash pass and an input-hash mismatch; both preserve artifacts as failed receipts and raise. The pass case checks all six validation origins, targets, and output schema. No real archive, checkpoint, test split, GPU, or optimizer was used.

The relevant ordering is sound: final report/prediction bytes are written, input hashes and canonical gate are rechecked, checkpoint bytes are checked, then a final budget check runs before the function marks success. In `finally`, another complete input-hash pass runs; `_finalize_phase` then performs the budget guard immediately before writing/augmenting the receipt and releases the guard. There is no later long hash after that guard. The test that fails the third hash pass verifies the failure path rather than relying on source-order inference alone.

Source SHA256: `D0212737ED0F02E6F9342F788EFDA73C8C3BBA53D12662F15E734B96CB6D2D9E`. Test SHA256: `BFC01D7F96A0B91825CF0416C6B3644A87D4932CE24475A74A17C45CD828AD8B`.

Limitations: fixtures do not execute the shared real lease implementation or real-data IO. The wrapper rehashes and verifies P3A input/evidence but does not itself independently rehash every P2/P3 archive; the managed P4 dispatcher/controller must retain that predecessor binding. Combined-bundle calibration materializes the archive arrays, including the test suffix, while indexed evaluation is constrained to validation origins; the report correctly discloses this as `test_accessed=true` and says no test origin/target was used. Calibration remains a technical validation procedure, not a scientific acceptance.

# Independent narrow review: verified frozen dispatcher v2

**Verdict: engineering-only pass with one hardening gap and one scientific caveat.** The 9 dispatcher mocks plus 11 calibration tests passed together (**20 passed in 11.98s**). This is source/schema and CPU-mock evidence only; no child phase was launched.

The dispatcher enforces per-driver required-flag subsets while allowing only enumerated optional flags; rejects duplicate/extra flags, unsafe paths, existing output directories, and missing principal artifacts; snapshots the pipeline, passed-stage inputs/evidence, declared sources, driver, validator, and argument files; runs the official before/after-P4 validator; and accepts only one new completed phase receipt associated with the exact output directory. Source inspection confirms the tested child fields are produced by the actual drivers: graph `origin_count` (`complete_phase_receipt`), H2 `origin_count`/`report_sha256` (`run_with_guards`), calibration input/parameter hash fields (`_augment_phase_receipt`), and C06 `origins_completed`/`outputs` (final receipt update). The tests' receipt documents are handcrafted mocks, not serialized by those producers; the field-to-source inspection closes that gap for required fields, but does not demonstrate a full child/dispatcher run.

Hardening gap: C06 receipt validation checks every hash present in `outputs`, but does not require that this map include every principal artifact. A child could omit one principal name from the receipt map while the artifact exists and passes the separate nonempty-file check. The real C06 writer currently exports all files, so this is not an observed producer mismatch; require the principal-artifact set to be a subset of receipt `outputs` and compare each digest before production use.

Scientific caveat: this dispatcher permits arbitrary driver policy/config arguments after allowlisting flag names. It does not establish that periodic/random controls are action-blind. Current scheduler cost-fitting can re-rank offered actions using graph/state scores; until the separate frozen blind-control adapter is source-bound in the job and used, such runs must not be described as pure blind controls.

Source SHA256: `BC0EDA83133173F1C1D3EB28C2E75D6A4AF3ED8207E40CF12C40637306BF37FB`. Test SHA256: `362F6B23439DEA3AC20EA97C06D6A1C536F410A788929523D82FD65589F592A6`.

Validation SHA256s: graph v6 `7C91A01F237D64C7F2DB383FC31A823265EDD7E42B4DAA456744E755716302A7`; H2 v4 `6C4FF121436990B8AAF97993525ED9D995F2DED40E252022C78518A49643C578`; C06 v4 `15F8996B90E1BB6C01D139E6B9484B1D6272E4CADE67B373C37446F97E08C107`.

The scripts and existing tests remain unchanged. These reviews do not confer actual phase acceptance or validation of scientific outcomes.
