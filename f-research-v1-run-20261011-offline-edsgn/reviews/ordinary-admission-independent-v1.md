# Independent review: ordinary admission helper

**Verdict: pass for the bounded metadata-only admission path, when paired with dispatcher9's pre-child snapshot recheck.** This is engineering evidence only; it does not accept or complete the prospective test matrix.

I checked `ordinary_frozen_test_admission_v1.py` against the ordinary-admission and byte-identical replay design, then ran five focused Conda CPU fixtures covering static blueprint identity, checkpoint-copy admission, serial jobs with prior outputs, and rejection of existing output/alias for the current job. The helper's official temporary-RUN fixture also validates the actual canonical P2/P3/P3A/P4 state and global queue helper. Dispatcher9's current public fixture passes the actual helper path; its two new pre-child mutation tests reject a newly added dataset sidecar or modified source after the official pre-P4 validator, before the driver is called.

The static route derives the P3A blueprint again from the config and current dataset/code/prior inventories; it requires `comparison_id == method_id`, exact variant/seed/config/output-binding path, and equality with the at-launch binding projection. `verify_binding` checks current dataset membership and hashes, config, code-file set and bytes. The helper returns these consumed refs, which dispatcher9 adds to its before/after byte and directory-membership snapshot.

For checkpoint-copy rows, the matrix binds the execution-copy path before its bytes exist. After training, admission requires the phase-owned original at `config.output_dir/checkpoint.pt`, an execution copy under the frozen-replay root in a distinct parent, equal SHA-256, an adjacent `test-freeze.json` bound to the copied checkpoint, and an explicit original checkpoint ref. The training phase receipt, summary, launch binding and validation-only numeric audit are checked against the original training run; the root acceptance must cite exact config, binding, phase, summary and checkpoint hashes. The copy and its adjacent freeze are both included in consumed refs.

The matrix requires unique IDs, output paths, aliases and checkpoint-copy parents. Admission checks absence only for the current job's output and alias, so completed earlier serial jobs do not block the next row. The job argv and sources must match the P3A row, and global queue membership binds the actual job path, alias and output. The binding points back to the already frozen queue; the queue carries path-only future job/binding entries, so this reference direction avoids a hash cycle. The return map includes the matrix, queue, training provenance, actual binding, code/data/prior inputs, selected driver and all declared job inputs.

The source inventory still relies on the dispatcher consumer to close the last filesystem race between initial admission and child start. In the current dispatcher9 integration, `_snapshot_matches` is called again after the official pre-P4 validator and immediately before child launch; the two mutation fixtures establish child-call count zero. Consumers that call this helper without that post-validator pre-child fence would not get the same race protection.

Focused independent tests: **5 passed** (`logs/ordinary-admission-independent-check-v1.log`). Combined helper/dispatcher integration after the pre-child fix: **32 passed** (`logs/dispatcher9-ordinary-admission-v3-tests.log`). All fixtures use temporary synthetic files and mocked execution; no real dataset arrays, checkpoint, model, optimizer, or GPU was read or run.

Hashes:

- helper source: `7e265b84d33a98c16032ed152ab6f3668fd1d6ddb7a6d0193ab5de0eada24b48`
- helper tests: `fcb99304d03db80ccaa6758c517b4332a5c97e0b3f8bb08290870609905f6457`
- dispatcher9 source: `34ab18bc9f5da5134615e204af314828c444ef041d0efa85847f5b936b039a19`
- dispatcher9 tests: `1085213be6ec523d15ea93140a47edcfbde61ebcae5ba26651fc0477677940a`
- independent focused log: `6e54aa20c7943afd05cbd991bec97e9f91da18837bc047dd805f2c3f81d43030`
- combined integration log: `b2c1538a9ae4aba4e3099e2195ad85ab6dec7595e66ab75fdaf4213937872bd5`

