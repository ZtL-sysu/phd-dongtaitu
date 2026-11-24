# Root acceptance: failed v3 WAIT and narrow v4 status-write recovery

The v3 WAIT process exited at11:41UTC with WinError5 during atomic replacement
of its mutable status JSON. Its final exception snapshot remains in the original
temporary file; the last published JSON is stale and must not be treated as a
live process. Root observed PID27672 absent and session15568 exited1. Neither
record reports replica training; all24 accepted replica output paths are absent.
The actual baseline GPU campaign and the seed0/GRU coordinators continue.

Root reviewed the v3-to-v4 diff: only atomic status persistence and the new log
path change. A unique same-directory PID/UUID temporary file is fsynced, then
os.replace is retried for at most15seconds on Windows sharing/access errors5,
32 or33; other errors fail immediately, exhausted temporary evidence remains.
Scientific subprocesses are never retried by this change. Eight Conda CPU
fixtures passed, including real replacement after one simulated lock, bounded
exhaustion and rejection of non-transient errors. Root rechecked unchanged112
preparation pins,24 template/config identities, output absence, coordinator
sources and the official before-P3 gate.

Root authorizes exactly one v4 replacement WAIT process and the same conditional
24-job continuation already accepted for v3. All56 exact predecessor receipts,
six numeric seed0 acceptances, resources, unchanged preparation and the same
cumulative72-hour budget remain required. No replica/test training or stage
advance is represented by this recovery acceptance. V3 files are preserved.
