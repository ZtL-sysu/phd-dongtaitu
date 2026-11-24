# Count-loss adapter v2 — implementation self-review

This is a staging-only, prospective matched-loss control. The adapter copies the existing `engine.train_run` function globals and replaces only the loss function, dataset guard, and source-manifest function in that private namespace. It does not mutate the engine module or authorize a training run.

The Poisson loss is the mean elementwise `PoissonNLLLoss(log_input=False, full=False)` over valid target cells. It rejects invalid valid-cell values and negative count means, safely ignores masked padding, and clamps valid means at the declared `1e-8` floor. The adapter requires the original V4 seed-0 email/MOOC configuration and a byte-hashed canonical MSE reference under `configs/core-development-bound-v3`. After excluding only `output_dir`, `training_loss`, and `mse_reference_config`, the candidate configuration must equal that reference exactly; the output must be a distinct path under `RUN/runs`. The reference bytes and adapter source are added to the private training source manifest.

Validation used local Conda CPU tensors and temporary JSON fixtures: `py_compile` passed and the focused suite reports 23 passed. Fixtures cover scalar loss arithmetic, masking and gradient behavior, strict recipe types, canonical reference/hash checks, effective-config mismatches, count-dataset enforcement, and private-global nonmutation.

No real archive, model checkpoint, optimizer, GPU, or training run was accessed. The helper does not bind the reference into a phase queue, validate a root training admission, or alter the active controller. Those bindings and any execution remain pending separate acceptance.
