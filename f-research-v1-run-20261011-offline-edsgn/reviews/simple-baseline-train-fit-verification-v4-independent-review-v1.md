# Independent review: simple baseline train-fit verifier v4

The verifier’s public CLI acquires its two leases before validation, runs the official before-P3 gate, takes source/data snapshots around the coefficient reconstruction and validation replay, applies the cumulative-budget checks, and fails closed when the shared owned-process scanner reports any active job. The 20 CPU tests passed, including a mocked active-job refusal before `datasets_at` and a process appearing midphase; both cases preserve a failed receipt and release only the verifier’s own locks. The test fixture’s target spy also confirms that coefficient fitting and replay use train/validation origins only.

Source SHA-256: `EF3C91E9FFA75134298D6888D2C6B52CA13E746EFD707366EBC01ECB155B7E1C`. Test SHA-256: `844E51D932479F0250C7080EFAA6EF0C7AF0D3E8BFDA9EFC57ECE90B4D31A28C`. Conda CPU result: 20 passed in 7.09 seconds; whole-command wall time 8.3679893 seconds.

Verdict is limited to engineering behavior under temporary fixtures. The process scanner is mocked in the active-job tests, and no real coefficients, dataset archive, model, GPU, or training phase was loaded. This review does not establish acceptance of a historical fit or any scientific result.
