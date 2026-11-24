# Independent review: frozen graph/state diagnostic v8

**Verdict:** engineering source and CPU-toy review passed, with an output-receipt integration limitation. This is not an actual test replay, model evaluation, scientific result, or oracle analysis.

The local Conda test command ran 25 tests successfully (3.61 s). The real-schema toy test covers full B1-origin replay, frozen train normalization, oracle-load spying, source/gate mutation and final serialization budget failures. Source inspection confirms the full origin order is asserted and oracle bytes are only included in a hash fence; the driver does not decode oracle NPZ members. Retention interpretation correctly distinguishes same-step graph matching from whole-window graph identity.

One integration item remains for managed acceptance: the driver receipt contains `output_dir`/origin count but no explicit per-file output path/SHA list. A dispatcher/collector should bind every output artifact before marking the phase accepted. The driver’s own local gate rechecks P2/P3 status/verdict and P3A evidence hashes; the full P2/P3 predecessor snapshot must remain enforced by root-managed dispatch.

Driver SHA-256: `4d54488130639e3504d5da307d523249e3b18e08e768dfb01c6bc8680c8d8be4`
Test SHA-256: `f485975e49fe954228d45aa872f0e82c2c5c8ae1ae22b4ff2dd7d79462852fe8`
Evidence: `frozen-graph-state-diagnostic-v8-independent-review-v1.json`.
