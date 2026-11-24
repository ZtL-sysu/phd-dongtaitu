# Bound phase campaign: independent v3 review

The focused mock suite passes after the input-binding repair. Command-line file paths are now collected by `command_input_paths()` and included with explicit `job.inputs`; a changed checkpoint omitted from the queue list is still recorded and fails post-run hash verification. Robustness uses its callable standalone wrapper, empty-success output is rejected, phase output hashes are checked before reuse, queue hash is rechecked after the child, and budget-stop mocks confirm that only the child process owned by this campaign is terminated.

Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest -q edsgn/tests/test_bound_phase_campaign.py` with `PYTHONPATH=edsgn/src;edsgn/scripts`. Result: **8 passed**. Tests used temporary files and mocked children, clocks, process lists, and budget responses; no actual child/model job, optimizer, GPU, or dataset was run.

The driver still fails closed on an existing campaign lock but has no automated stale-lock recovery. A crash outside Python `finally` can leave a dead-owner lock requiring a separately verified manual recovery. Before implementing recovery, require proof the recorded PID is absent and retain the old lock contents in recovery evidence.

The queue now hashes file-valued paths parsed from each phase’s native CLI parser plus explicit inputs. This makes phase parser definitions part of the integrity boundary: a newly added file-valued flag must be declared by the phase’s input-path registry and covered by a regression test. Keep checking parser/registry completeness when phase CLIs change.

Two schema/provenance checks merit review before broader launches: (1) for non-training jobs, ensure `parameters_before/after` or weight hashes are not merely equal nonempty strings but are tied to the checkpoint listed and hashed in the receipt; calibration should bind its `checkpoint_sha256`, selected policy digest, and cost-table metadata to that same frozen input; (2) training output validation checks full scope, checkpoint hash, epoch count, and manifest train-window coverage, but add a focused positive/negative fixture for configured full validation-window coverage and manifest/config consistency. The budget sidecar path should also remain integrated with `aggregate()` exactly once, with active output excluded and elapsed time added during execution.

No campaign phase was launched or accepted by this review; passing mock tests establish orchestration behavior only.
