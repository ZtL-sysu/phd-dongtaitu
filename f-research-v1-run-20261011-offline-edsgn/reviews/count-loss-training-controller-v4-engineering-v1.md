# Count-loss training controller v4 — writer engineering evidence

V4 preserves v3 queue validation, canonical P2/P3/P3A and official before-P4 checks, exact adapter command, private unchanged bound runner, complete job input/output fencing, resource/campaign leases, preserved failures, and the cumulative 72-hour budget. It only changes process-command parsing and handling of unavailable psutil identity fields.

The scanner parses Python interpreter prefixes before the actual `-m` or script target, including `-u`, `-B`, `-O`/`-OO`, `-X dev`, and `-W ignore`. It does not search arbitrary later argument text or `-c` code for a script name. A RUN-owned `-c` process is refused because its executed code cannot be identified as a script target. It falls back to `Process.name()` when `process_iter(attrs=...)` fields are `None`; a known non-Python process is skipped, a disappeared process is skipped, and an unclassifiable or potentially RUN-owned Python process fails closed.

Local Conda CPU test run: **26 passed in 23.82 s**. This includes all 17 inherited v3 public-queue and scanner cases, four valid prefixed invocation forms, arbitrary script-like argument text non-detection, `-c` refusal without parsing code text, psutil `ad_value=None` Python fail-closed and non-Python skip cases, and ProcessGone handling. The public successful mocked-child test stubs the process iterator barrier to isolate that path from this machine's unrelated unclassifiable OS process; direct scanner fixtures test the barrier itself.

The live local scanner also encountered PID 188 whose `cmdline`, `cwd`, and `exe` are AccessDenied and whose direct process name is empty. V4 correctly refuses because the process cannot be classified or ownership ruled out. No real job was started. A root/operator must resolve this host-specific refusal from trustworthy process metadata before any future execution; the controller must not be weakened to skip it without proof.

Hashes: source `11399b3e30e31b7a7b1252deB617d1ae92f53b62a9fda3e6fbc328ec702862bd`; tests `2ff19cb4e7383048b4a10621db22b883a04adfb6334f5fbc59e074dc319f9bd5`; log `1702b4c88f1a2de0a5239b8375fa8d0c691834a96b74544e47eb948c7504b10b`.

This is implementation evidence only. No real data/model/checkpoint, optimizer, GPU, campaign, canonical state, or actual execution was used or authorized.
