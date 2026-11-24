# Guarded primary-family assembler v2 — writer evidence

Scope: engineering-only staging implementation and temporary CPU fixtures. No canonical pipeline advancement, real archives, model inference, GPU, or scientific test result was used.

Implemented in `edsgn/staging/guarded_primary_family_assembler_v2.py`:

- Uses typed producer v4 and plan validator v7; NI raw p kind is `centered_lower_tail_noninferiority_p`.
- Revalidates plan and postrun manifest with validator v7's pure metadata functions and checks each manifest-consumed file SHA before mapping and again after output serialization.
- Requires root Holm acceptance to affirm `pipeline_advanced=false`, `scientific_hypothesis_accepted=false`, and `actual_primary_inference_run=false` in addition to the accepted source evidence map.
- Removed the `_test_root`/`official=False` public assembly bypass; only the canonical RUN with official P5 validation is accepted.

CPU evidence: `logs/guarded-primary-family-assembler-v2-cpu.log`; 7 passed. Public producer v4 forecast superiority and noninferiority records were generated using the existing small temporary fixture; metadata revalidation confirmed the consumed hash set, and mutation of one consumed input was rejected. NI value mapping was checked against the live v4 `_analysis_fields` contract. This does not assemble an actual completed 49-record result set; that remains pending until producer records and root result-acceptance artifacts exist.

Source SHA-256: `4807140339919F8867D530CAA1B91C47AD97C64F9C1583E69520E80316839373`
Test SHA-256: `A521F90F2FBA344F5C15925188524AAD4F7735F5958F095411194E0BFE0E503B`
