# Full core pilot scheduling correction

The sixteen default seed0 configurations prepared in
`configs/core-development-v1/` have never launched. Their old prospective
`p3-core-development-queue-v1.json` incorrectly permits full V2/V3/V4/V9
training during P3. It is retained as an unexecuted preparation artifact,
and is not an accepted or executable scientific allocation.

The successor `p4-core-pilot-queue-v2.json` uses the same configuration bytes
and output paths. It requires actual strong-baseline evidence, accepted P3A,
and P4 running. These are bounded default train/validation pilots, with no
test access; configuration selection and the five formal seeds remain later
requirements. No result or gate is inferred from queue preparation.

This corrects prospective execution sequencing without changing the model,
data, optimizer, trial limit or accepted execution contract. The campaign
controller must enforce the proposed-model gate from the actual config,
even if a queue labels a full proposed run as development.
