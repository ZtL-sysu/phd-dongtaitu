# Serial resource queue driver v2 addendum

The final validation pass added two additional fail-closed checks to v1:

- Queue config digests are resolved from the top-level frozen `configs[]` map; `queue[]` rows may omit `config_sha256`. If a row does include one, it must match. The first live config's actual SHA-256 matches its `configs[0].config_sha256` value exactly.
- Every queue output/config/dataset path is constrained to the runroot's external-resource logs, frozen external-resource configs, and prepared-data directories. All existing attempts/outputs across the full queue are checked before the first child launches, then completed reuse is reverified after lock acquisition.

The final mock-only local Conda suite is recorded in `logs/serial-resource-queue-tests-v2.log` (**5 passed**). No production queue command was run. The live queue status is `proposed_unlaunched_unaccepted`; explicit acceptance remains a launch prerequisite.
