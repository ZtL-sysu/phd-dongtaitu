# Schema-4 public outcome integration fixtures (engineering only)

The temporary canonical-run fixtures now exercise four `produce_endpoint` paths through the public producer, actual schema-8 plan validator, dispatcher/completion receipt validators, candidate-binding helper, and official pipeline-state gate: ordinary forecast, matched H2, oracle-free graph-v8 H1, and profile/cost H3. The H1 fixture obtains the public graph-v8 route by setting schema 4 before driver resolution. Matched H2 checks the four arm archives and paired graph proof; graph-v8 checks the oracle-deferred report contract; H3 uses CPU-labelled five-repeat timing rows. All fixture payloads are generated toy arrays/receipts in pytest temporary directories.

Focused run: `python -m pytest -q edsgn/tests/test_typed_endpoint_producer_v5.py -k 'schema4_public_forecast_queue_candidate_and_global_completion_path or schema4_public_h1_graph8_queue_candidate_and_global_completion or schema4_public_h2_matched_queue_candidate_and_global_completion or schema4_public_h3_profile_queue_candidate_and_global_completion'` using the local `dyg-cold` Conda interpreter. Result: **4 passed, 22 deselected** (88.78 s); raw log is [typed-endpoint-schema4-public-outcomes-v6-tests.log](../logs/typed-endpoint-schema4-public-outcomes-v6-tests.log).

Current source pins observed during this run:

- Producer v5: `afbe2cb083b6fc7ffe62cfab1e9a164191225d3b96e6459884ac4afd3510e3d5`
- Plan validator v8: `f3a33630d0787920d8bd5b964702a7cb5a94c8be6f6381c33a4c6884f8149d73`
- Dispatcher v8: `b946169932699f5efad8ae187e5c42cf2e247ccf0c97c895c9103c21d7748998`
- Global completion verifier: `59b3d529efa76dc475448b3eccf88529a5ab12cc7ade53913f66d46b7a3850af`
- Global oracle reporter: `6a42146ddbce3c1520a89cc502a3167835f4261fe9f49f56a1c8a1714e51a5c2`
- Fixture tests: `b2fa3fbcf82a8572cb46bed7b53bfb7b58e265c7dd7b94d06388b823d225e29f`
- Test log: `e05c5488864660901adea2813db60bc3d8edd0eaaf2f7c00c261a619424b0bd9`

These fixtures establish API/schema compatibility only. They do not establish actual P3A/P4 acceptance, global test-matrix completeness, model prediction validity, statistical claims, hardware cost, or test-set evidence; the test outputs are synthetic toy artifacts. Dispatcher-8 scope remains limited to its registered primary endpoint jobs and cannot by itself certify that every ordinary/secondary test job has completed.
