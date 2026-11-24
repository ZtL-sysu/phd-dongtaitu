# Independent review: blind reference cost policy v1

**Verdict: engineering-only pass on CPU/source evidence.** I ran the dedicated suite in local `dyg-cold` Conda: **8 passed in 2.88s**. No real data, model, GPU, training, or optimizer work was used.

The helper's periodic priority matches `Policy._select`: same `step`, `ceil(fraction*N)` rotation, eligible nonmandatory offer set, lower priority first. Its random priorities use the same torch generator seed (`seed + step*1009 + salt`), action salts 11/37, N-sized random vector, and descending ordering as the original selector. `_recording_select` calls the bound original `_select` and only records action step after the call, so the offered sets and overrun counts remain the original ones. `_blind_fit_cost_budget` never reads `sg`/`ss`; it ranks offered candidates by normalized within-action rank / nominal quota, then graph-before-state and node id. It retains mandatory masks, subtracts base cost from budget, uses the original `incremental_cost` interpolation after each candidate, and keeps mandatory actions even if their cost exceeds the envelope. The tests cover score swaps for both control policies, deterministic periodic rotation and random repeatability, tie order, base-cost behavior, mandatory overrun, and unchanged adaptive/full policy behavior.

The helper adds only dynamic instance attributes/method bindings. The tested `dataclasses.asdict(policy)` and `engine.policy_digest(policy)` remain unchanged before/after binding. Freeze extras bind the exact helper source SHA and rule version and reject stale/missing bindings.

Source SHA256: `2500F9687AA277AD5E67F3D4D41DDE095D2E51761109DA2F9DFCEB6A68E3218D`; test SHA256: `5DE9480763AF7DFABBB8449C0B5D827C67DAC9331D086BA979F5293021F573EE`; `scheduling.py` inspected SHA256: `544941A6BBE5717E3A7F1ACA20D578725C3722A180AE78838578F5F465F07D42`.

Limits: this validates equivalence of the offered-action ordering and accounting logic on small CPU fixtures, not runtime or scientific control performance. The cost table remains B1-specific, as required. This verdict alone does not show a run used the adapter; production freeze/config must bind and actually install it.
