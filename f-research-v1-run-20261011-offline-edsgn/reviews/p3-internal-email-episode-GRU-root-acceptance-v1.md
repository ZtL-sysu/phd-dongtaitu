# Root acceptance: email-episode GRU seed0 validation

Root accepts the complete source-bound development training and validation
result after the unchanged numeric-v3 audit and independent saved-validation
recomputation. This run covers 334 training windows each epoch,
46 ordered validation origins, and 41
epochs; epoch31 was selected by validation MAE. Mean raw MAE over
H1/H3/H7 and the two features is 0.690539224844571. Every required horizon/feature
MAE and pooled RMSE matches the saved metric CSV within1e-6. Root reverified
the exact requested/resolved config, actual launch binding, direct phase config
input, phase input/output bytes, summary and original checkpoint hash.

The independent reviewer opened only source metadata and saved validation
arrays; numeric-v3 additionally verified validation targets/masks against the
source. Its chronological combined-array materialization may include other
partitions, so this acceptance does not assert untouched test bytes. No test
origins were scored, no oracle was read, and checkpoint bytes were hashed
without deserialization. Main email's all-zero validation cannot establish
generalization or identify a relative noninferiority margin. Episode fitting
is separate and does not replace original email.

This is seed0 validation evidence. P3 remains running; five-seed formal
comparisons, frozen tests, P3A and H1/H2/H3 remain pending.
