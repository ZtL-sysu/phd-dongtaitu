# Simple-baseline train-fit verifier v4

Writer self-review, not an independent review. The verifier now checks the accepted shared owned-process scanner result at every `Guard.checkpoint`; a nonempty active-job list raises before budget checks or data access.

Two new public-CLI CPU fixtures cover (1) an already-active owned process blocking initial lease acquisition and (2) a process appearing at the next in-phase guard check. Both assert no `datasets_at` call, failed phase receipt, lock cleanup, and no audit output. The inherited v3 suite also passes.

Validation: local `dyg-cold` Conda pytest, 20 passed in 6.62s; measured full command wall 8.25s. No real dataset, checkpoint, model, training, or held-out values were used. This is engineering evidence only; it does not authorize or complete a real fit audit.

Source SHA256: `ef3c91e9ffa75134298d6888d2c6b52ca13e746efd707366ebc01ecb155b7e1c`  
Test SHA256: `844e51d932479f0250c7080efaa6ef0c7af0d3e8bfda9efc57ece90b4d31a28c`  
Test log SHA256: `31f89af3cd3963c04206f53d1f3a75a80c0a98015f5b1b3cc13693e72b22ee8f`
