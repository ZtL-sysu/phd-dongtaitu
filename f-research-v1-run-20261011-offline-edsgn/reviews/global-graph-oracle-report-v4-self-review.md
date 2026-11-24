# Global graph oracle report v4 — implementation self-review

Writer self-review only; not an independent/scientific review. This supersedes prior writer notes. In addition to queue path isolation and runtime source pins, the report helper now recomputes the full collection membership/hash fingerprint for every graph job before report serialization and again after serialization. A newly added otherwise-unlisted collection file is rejected; known trajectory/oracle refs remain covered by consumed-byte checks. The report helper still runs only after the frozen global completion verifier and emits sampled detailed-row alignments without full-window or causal recovery claims.

Local Conda `dyg-cold`: combined verifier/reporter suite, 27 passed; `py_compile` succeeded. New fixture adds an unlisted collection member during oracle decode and confirms the final collection fence rejects. Log SHA-256 `1A98670FBFFC170B06089FD4E5DF2B2574137D97A3257662256A265C36146042`.

Current sources: completion verifier SHA-256 `59B3D529EFA76DC475448B3ECCF88529A5AB12CC7ADE53913F66D46B7A3850AF`; report helper SHA-256 `4398A9154411234931952A99F6AFE3E2E7A8D165ED7246497BE921A7A4A8250F`. Tests: completion `F5AEBFB55351341A08199274EB361029504AFBB680CA4BCCB5C9C972BA0B5311`; report `A630393209303FC263486BF99489E7A521E67FC0E64E0E9EB22F6D047687B81B`.

Only temporary hand-built toy archives were decoded. No actual data, oracle, prediction result, checkpoint, model, GPU, or training was accessed. Production queue/authority, actual report execution, resource feasibility, and scientific acceptance remain pending.
