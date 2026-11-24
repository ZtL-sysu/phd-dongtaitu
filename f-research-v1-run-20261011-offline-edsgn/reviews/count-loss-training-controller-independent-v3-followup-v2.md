# Independent scanner follow-up v2: controller v3 blocker

This follow-up **supersedes the pass verdict** in `count-loss-training-controller-independent-v3.md` for launch safety. The controller and original tests remain unchanged.

## Reproduced findings

A fresh CPU-only fixture called the actual `_owned_processes` on RUN-owned Python processes. The scanner returned no active process for all three valid prefixed commands:

- `python -u -m edsgn.train --config <RUN>/...`
- `python -B -X dev -m edsgn.profile --split test`
- `python -u <RUN>/edsgn/staging/frozen_prediction_guarded_v5.py --plan ...`

The unprefixed control `python -m edsgn.train ...` was detected. The reason is `job_identity` only interprets `argv[1]` as `-m` or as the script path; Python accepts interpreter options before either. These are valid real command forms, so the pre-child owned-process barrier can miss an active RUN model/frozen job.

A second fixture supplied the documented psutil `process_iter(attrs=..., ad_value=None)` shape for an inaccessible process: `info` succeeds but `cmdline`, `cwd`, `exe`, and `name` are `None`; `Process.name()` identifies `python.exe`. The controller returned no active process and did not refuse. Local psutil 7.2.2 source confirms `process_iter(attrs=...)` calls `Process.as_dict(attrs=attrs, ad_value=ad_value)`, and `as_dict` substitutes `ad_value` (default `None`) on `AccessDenied`/`ZombieProcess` per attribute. The current scanner's exception handler therefore does not cover this ordinary placeholder case; it sees no Python identity and no RUN ownership, then skips it. An inaccessible process cannot safely be assumed unrelated to RUN when the required ownership fields are unavailable.

These are two concrete fail-open paths in the process gate, not merely missing test cases. The original controller/test hashes are unchanged: controller `53decda89ed7548b0bbe2c37e65eff1a257d32450b9cf4b0691cdd26c4a26033`; tests `3c559e4b8e009da3ef772c74abe951f014daf405fc2c560de531c93f2eb0b9b8`.

## Evidence pins

- Previous independent pass report, superseded: `reviews/count-loss-training-controller-independent-v3.md`, SHA256 `2de84fece7726c606ef0105745cade3a1c6dc274ce0c18361c4942ac4649a85c`.
- Original CPU suite still passes: `logs/count-loss-controller-independent-v3-tests.log`, SHA256 `51b7dfac9fac0ce8362282155daedce70f68a48c0e0989d7243be58cff096c41`, 17 passed in 34.67s. This suite does not cover prefixed Python interpreter options or psutil's per-field `None` placeholders.
- Attack fixture log: `logs/count-loss-controller-independent-v3-scanner-attacks-v2.log`, SHA256 `7c131907c423857709fd8b7738d748e97c49d5971e3a5df34f62f64a3b0ff1af`.
- Design: `protocols/count-loss-controller-root-design-v1.md`, SHA256 `3962fcffb7cf13efe5de206e3acf6a7be11321e56acc31ee75d30823af2e90a3`.
- Source line anchors: `_owned_processes` 392–495; `job_identity` 440–450; current `argv[1]` dispatch 444–454; psutil `info` handling 462–485; bound runner scanner comparator `edsgn/scripts/bound_phase_campaign.py` lines 38–53 scans `-m` across argv positions.

## Disposition

**Blocker: process barrier is not fail-closed for valid Python prefixes or psutil `None` placeholders.** Do not rely on this controller's scanner as the sole exclusion barrier until a successor parses interpreter options without interpreting arbitrary command text as a script and treats unavailable process identity/ownership fields conservatively. No source fix was made in this review. No process was launched; no data/model/GPU access occurred.
