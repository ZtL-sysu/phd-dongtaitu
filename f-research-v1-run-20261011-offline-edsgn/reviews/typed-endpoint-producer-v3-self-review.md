# Typed endpoint producer v3 / metadata validator v6: writer self-review

Role: writer self-review, not an independent review or scientific acceptance. Producer v2 and metadata validator v5 remain preserved.

The metadata plan schema now freezes `profile_device` only for H3 cost endpoints (`cpu` or `cuda`) and binds a separate report-only switch-authority descriptor. Metadata validation hash-checks the authority and its source files without decoding switch values or adding them to the model-readable collection. After all ten semantic receipts pass, the producer reads only each external NPZ's scalar `switch_index`, checks it against the frozen plan map, then supplies converter v4's required `prepared_frozen_switch_mapping` proof with source and mapping hashes. This proof format was tested against the unchanged converter.

Same-seed/same-role output sharing is permitted for matched H2, and for H1 V4-vs-V4 graph response only when both sides bind the same producer source. Cross-seed and cross-role reuse remains rejected; H2 arm archives must remain distinct. Cost device handling follows converter v4 canonicalization, including `cuda` and `cuda:0` equivalence. Producer source binding now names `verified_frozen_dispatch_v7.py` in both source inventory and required helper checks.

Validation used the local `dyg-cold` Conda environment. The focused suite passed **40 tests**. Four full public producer integrations use temporary canonical RUN fixtures, unmodified producer numeric/semantic routines, metadata validator v6, and the official pipeline validator:

- Ordinary V4-vs-V3 full-origin forecast with five-seed archives.
- Matched H2 with shared checkpoint, report, and graph proof, distinct informative/control prediction archives, and arm-specific report hashes.
- H1 V4 response with shared checkpoint/report/trace and separately bound toy switch authority; the mapping reaches unchanged converter v4.
- H3 paired profiling with identical frozen checkpoint bytes, full five-repeat CSVs, and both CPU and CUDA (`cuda:0`) device paths.

Mutation fixtures also reject a changed dispatch before/after snapshot, an H3 device mismatch, and a non-scalar report-only switch before trace numerical processing. All arrays and oracle files used here were generated toy fixtures; no real dataset, checkpoint, test archive, GPU, optimizer, or scientific endpoint was run.

Evidence log: `logs/typed-endpoint-v3-v6-focused-tests.log` (40 passed).

SHA-256:
- producer v3: `C77A3EBB15B0C6BA959EA7C5BF069E850837C87954960411E4DA0ECFC5063F86`
- validator v6: `4A2AE2FFD66D32449E6A90B881967E9701FC3D000C1CFB038D6AC3390F8BA110`
- producer tests: `F088805FB9FB05B6FA4E5B97741EC24902980A2C3FAE1529200557130D01F7A8`
- validator tests: `D4D10C605892A6F123C2336D54AFE1F5F0F5EBE651C272EE3F027A55A2B2CDBF`
- dispatcher v7: `1B95E9E9AF3C6A28E009937148FD688EC361344C7865A7D857F7A3EFEC845F62`
- test log: `82DE5271EBBFC09C79C145D57CA2F93FB58032C57EDD26D4D060E181221E1ECA`
