# Global completion v3: seven-driver mixed-fixture check

The global-v3 completion verifier passed a mixed, single-queue closure containing all seven required driver contracts: graph v8, prediction v5, H2 v4, profile v5, C06 v5, simple v3, and case v3. The test fixture used opaque output bytes and a `numpy.load` spy; it checked queue/job/binding/receipt ownership, report scope and origin count, principal artifacts, closed output hashes, and the per-driver phase-output schema. The unknown mandatory-driver case was rejected.

The Conda CPU run combined the existing 31 tests with 30 new mixed-driver tests: **61 passed** in 13.56 seconds; measured whole-command wall time was 16.772269 seconds. Source SHA-256: `6B4979738256B0FFE86DD30B94DB6543C7CED3E578F88CA70410C0507D06A1C5`. New test SHA-256: `77407CD81D862237CEA01E5B60A580E5A3B66BA4FB4B5E97FCE6E1942DC86FCB`.

This is engineering-only metadata/hash validation. It does not load real archives, checkpoints, or models; it does not launch or replay any phase, and it is not evidence of scientific acceptance or actual global completion. The case path remains closed to real admission until the dedicated case admission helper and a reviewed dispatcher that actually supports `case_v3` are present and bound into the frozen queue.
