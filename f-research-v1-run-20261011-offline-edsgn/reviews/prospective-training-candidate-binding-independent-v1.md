# Independent review: prospective training candidate binding v1

**Verdict:** pass for static blueprint and selected-launch linkage as an engineering helper, subject to the caller prerequisites below. This is a source-only review. No candidates were selected or trained, and no target/test/oracle arrays were read. Root reports its 11 CPU toy tests passed; I did not rerun them.

The helper follows the P3A design’s prospective/actual split. Before P4 it freezes config/canonical config, dataset file set, fixed-prior files, package/author code files, pool identity, and expected future binding/selection paths without inventing future output hashes. `validate_preparation` recomputes that identity and rejects changed bytes. After selection, `validate_selected_training` checks the root-only validation receipt, exact selected config and expected binding path, calls the unchanged `bind_training_inputs.verify_binding`, and compares the actual launch record to the blueprint while keeping `recorded_utc` and at-launch status as distinct actual-only fields.

The deterministic ranker is bounded to 1–12 unique joint graph/memory candidates. Its accepted receipt path uses only synthetic and MOOC V4 model-seed-0 validation horizons with positive GRU seed-0 denominators, equal-weight normalized MAE, then validation wall time and candidate ID. Failed/unsupported choices remain explicit. This is metadata validation; the helper does not certify numeric evidence or issue scientific acceptance.

Caller responsibilities: enforce canonical phase gates and prove the timestamp/binding was captured before training; do not call `allow_postrun_outputs=True` before independently verifying the phase; map `dataset_id` to the actual dataset root in the P3A plan; and validate matched scientific configuration/evidence semantics for any future multi-candidate pool. The current singleton default pool does not demonstrate optimality.

Helper SHA-256: `3dce0d58bf59dcb16b86699c35e7ac6664f4357ce6cfe6390982b1b0ebc998d8`
Binder SHA-256: `028750be3b3f4ca9feba9c93fde176fb7a212aa3e2cb9c0e830c38ba94c4aa59`
Protocol SHA-256: `fb5b1e7e1c2fbf53aa78176a900d615c5846baa6d12583b3a10e089cb807e230`
Evidence JSON: `prospective-training-candidate-binding-independent-v1.json`.
