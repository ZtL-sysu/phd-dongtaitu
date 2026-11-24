# Independent metadata/hash review: six seed-0 strong results

Verdict: **passed** for the root-authored six-row seed-0 acceptance and the runner's numeric-acceptance verifier.

The aggregate binds exactly six unique pairs (synthetic/email/MOOC × DGCRN/SeqSNN), all seed 0. For every row, root's typed attestation and referenced review Markdown SHA are current; the corresponding independent numeric audit JSON passes its validation-only schema and agrees with the accepted pooled MAE. The phase receipt is completed/returncode 0 and every listed input/output file hash matches. Config and launch-binding hashes also match. The aggregate's 102 SHA pins are all current.

| Dataset | Method | Train origins/epoch | Val origins | Required-horizon MAE | Numeric audit |
|---|---|---:|---:|---:|---|
| synthetic | DGCRN | 16112 | 4028 | 0.02576999803 | pass |
| synthetic | SeqSNN | 16112 | 4028 | 0.02601654629 | pass |
| email | DGCRN | 528 | 74 | 0.001860108326 | pass |
| email | SeqSNN | 528 | 74 | 0.4197174658 | pass |
| mooc | DGCRN | 452 | 49 | 3.983563129 | pass |
| mooc | SeqSNN | 452 | 49 | 7.821371359 | pass |

The runner verifier returned successfully when given only the four fixed queue identity descriptors. This was strictly a schema/root-acceptance verification; this review did not inspect or accept the 56 predecessor job receipts. Root acceptance itself says `predecessor_56_receipts_complete_at_acceptance=false`, `formal_five_seed_result=false`, `pipeline_advanced=false`, and `scientific_hypothesis_accepted=false`; no broader conclusion is implied.

No prediction/source arrays, test values, model, checkpoint payload, or oracle were read.
