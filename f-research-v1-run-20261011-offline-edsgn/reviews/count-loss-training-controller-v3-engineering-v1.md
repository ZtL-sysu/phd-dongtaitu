# Count-loss training controller v3 — writer engineering evidence

V3 preserves the v2 queue, candidate/numeric validation, official pre-P4 checks, exact adapter command, input snapshots, resource lease, and cumulative budget gates. The launch process barrier now retains the bound runner’s model-module, baseline, frozen-driver, and resource-probe coverage, while adding the count-loss controller/adapter version patterns.

The scanner considers Python/PyPy processes only, requires ownership by RUN-local cwd or an actual RUN-local argv path, and recognizes a script argument or explicit `-m edsgn.*` training/calibration/evaluation/profile module. It skips its own PID and launcher ancestors, so PowerShell command text and parent-controller command lines do not create false matches. An owned Python process with no usable argv, or a Python process whose details raise AccessDenied/OS errors, refuses launch. A public mocked-run test confirms the second process barrier blocks before child creation and releases the resource lease.

Local `dyg-cold` CPU tests: **17 passed in 34.18 s**. The suite retains the five v2 public temporary-root paths (email/MOOC validation, two mocked child jobs, missing input, official-validator mutation before spawn, and unexpected argv) and adds process classification for `edsgn.train`, calibration, baseline/frozen drivers, resource probes, and count-loss processes; self/ancestor and PowerShell-text exclusion; inaccessible/no-argv refusal; and refusal before a mocked child. No real job, data archive, model, optimizer, checkpoint, or GPU was used. The canonical gate remains separately authoritative.

Hashes: controller `53decda89ed7548b0bbe2c37e65eff1a257d32450b9cf4b0691cdd26c4a26033`; tests `3c559e4b8e009da3ef772c74abe951f014daf405fc2c560de531c93f2eb0b9b8`; log `722a5d2a72a497dbc48657e6e651e92264a78ca015b63ffc43379a02d3f1ad67`. V2 remains unchanged at `bff6e993ead515f281716c8c4591a8a33e14067988aa84ebb7acac8ca2033250`.

This is writer-side engineering evidence only; root source acceptance and any later execution authorization remain pending.
