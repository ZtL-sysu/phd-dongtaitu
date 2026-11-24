# Frozen graph/state diagnostic v3 review

Scope: isolated successor review and CPU toy verification only. No source-package, controller, data, checkpoint, prediction, GPU, replay, or optimizer execution was performed. Existing v1/v2 files remain intact. This v3 is preparation evidence only; it is not authorized for actual integration/replay until root accepts its new frozen extras and stage-gate contract.

The v3 successor fixes the six requested blockers:

- `predictions.npz` writes scalar `seed_id` from checkpoint `payload["config"]["seed"]`; per-window source trajectory IDs are in `trajectory_seed_id`. A test writes the canonical NPZ names and loads it through `edsgn.statistics._load_prediction`.
- Clock snapshots index `runtime.spike_pending[nodes]`, retaining every selected node, fast/slow layer, and channel. The toy capture verifies `[2,2,2,4]` snapshots.
- Compact rows retain core step-trace quality/quality-change, residual/residual-smooth/validity, graph/state IDs, mandatory and quota/budget fields, model component host timings, input cutoff/global bin, and q/r summaries. `c` is passed through as null on no graph action. Detailed support and membrane/spike snapshots remain restricted to frozen sampled origins/nodes.
- Freeze extras bind the `frozen_synthetic_stress.py` normalization helper SHA directly and as a resource helper; replay verifies before normalization and after oracle-dependent processing.
- Prediction storage preflight uses checkpoint `hmax`; the compact allowance was increased for added trace fields. The writer counts uncompressed trace bytes and rejects allowance overflow; the completed output tree is checked against the frozen total artifact cap.
- Stage validation requires P2/P3 pass verdicts, P3A `passed` with `gate_verdict=pass`, P4 running, and all P3A inputs/evidence to exist and match recorded SHA-256.

Verification: `edsgn/tests/test_frozen_graph_state_diagnostic_v3.py` — 15 passed; `py_compile` passed. Log: `logs/frozen-graph-state-diagnostic-toy-v3.log`. V3 source SHA-256: `7d72b85b1b23031b490a99a3c0cdf1f3a20fb477e334349c859da8468046c00d`.

Remaining limit: tests exercise toy serialization contracts, not a complete prepared/frozen collection, canonical live pipeline ledger, or end-to-end replay. Re-freeze must include the v3 diagnostic hash, normalization helper hash, updated selection version, and unchanged artifact cap; subsequent integration review remains required. Instrumented component and wall timings are diagnostic overhead, not performance/energy evidence.
