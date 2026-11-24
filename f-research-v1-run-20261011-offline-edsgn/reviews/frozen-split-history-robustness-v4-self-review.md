# C06 frozen robustness v4 author self-review

This is writer self-evidence only, not independent acceptance. v4 adds the missing pre-write cumulative stop-loss check after replay integrity validation and immediately before output-directory creation and first prediction serialization. The existing same-callback post-serialization/final-integrity budget check and completion-receipt ordering remain in place. Tests trigger the two boundaries from distinct state predicates: before-write requires all origins complete while output does not exist; after-write requires both prediction and final perturbation manifest files to exist. Both cases leave a failed receipt; the former has no output tree, the latter retains already-written artifacts.

The v3 preflight descriptor binds the v4 evaluator, while domain manifest generation remains delegated to the unchanged v1 helper/schema to preserve manifest identity. Existing v2/v3 files and generated real domain artifacts/descriptors were not changed.

Focused local Conda CPU toy suite: 8 passed in 4.36s. No actual archive/checkpoint, optimizer, GPU, or replay was used. See `logs/c06-frozen-robustness-v4-cpu.json` for source hashes. Independent review and gated scientific evaluation remain pending.
