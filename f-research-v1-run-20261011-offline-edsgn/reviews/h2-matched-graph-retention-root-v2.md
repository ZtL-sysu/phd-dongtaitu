# Root acceptance of H2 CPU prototype v2

Scope: engineering preparation only, during unfinished P3. No actual data,
checkpoint, optimizer, GPU, frozen test replay or hypothesis outcome used.

Root inspected the v1 teacher/arm implementation and the independent-agent
review by literature_protocol. The ordinary V4 full-policy forward path
generates teacher graphs from causal input observations. Each fresh-runtime arm
executes real candidate/scorer work and then installs those exact teacher
supports, weights and valid bits. Only the pending retention signal differs;
the original clock, local/neighbor update, spike consumption and readout remain.
Informative parity and action/source/state accounting are checked in the toys.
Discarded scorer graph loss and extra teacher/scorer overhead are diagnostic.

V1 incorrectly recorded informative parity=True when that arm was absent. Root
preserved v1 and made v2: parity is null/status=not-executed for absent arms,
and the report explicitly states whether all four required arms ran. The v2
source SHA256 is 2bb4d80f6f75a299bc56dce66283730646426004f86600692fdd3010fca35d91.
The six existing toy checks plus omitted-arm reporting and actual graph,
candidate/scorer/state counter equality checks passed: 8 passed in3.47s,
logs/h2-matched-graph-retention-toy-v2.log, local dyg-cold Conda CPU.

Verdict: accept v2 as the retained toy implementation. Real use still requires
immutable source/data/checkpoint/control descriptors, exact complete origins,
canonical P3A/P4 and resource/budget gates, output provenance and the prescribed
paired analysis. Input causality cannot be established from the prototype's
caller descriptor alone. No timing/energy or H2 claim is accepted.
