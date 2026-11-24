# Root review: dispatcher v6 needs final source/device association repair

The six planned pretest references and concrete side/seed, checkpoint, policy,
dataset, output and phase alias are now associated before dispatch. Twenty-six
CPU mocks pass and the H2 CLI matches its real full-policy driver. Root still
found that source matching accepts any declared plan producer source hash rather
than requiring the actual selected driver path/hash. The cost job also must
match the separate prospective CPU/CUDA endpoint device. Preserve v6 and use a
narrow successor with wrong-driver and wrong-device rejection before the child.

No actual dispatch, data, model, GPU, test or phase advancement ran.
