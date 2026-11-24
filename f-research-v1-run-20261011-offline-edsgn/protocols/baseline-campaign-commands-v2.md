# P3 baseline launch addendum v2

The root froze `p3-training-resource-contract-v2.json` before new baseline optimizer launches. This adds the actual resource decisions to historical command notes; it does not rewrite old launch evidence. The canonical gate is P2 passed / P3 running.

Use the local Conda interpreter `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe`, working directory this run root. The campaign invokes the same interpreter, sets its package path and two CPU threads, binds source/config/data hashes before every fresh run and verifies them afterwards. No test evaluation is performed.

First launch:

```powershell
& 'C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe' edsgn/scripts/baseline_campaign.py --datasets synthetic --methods V0 V1
```

This is a complete seed0 training/validation development run (max100 epochs, patience10). All other baseline datasets/methods remain pending. Sequential queues and the cumulative72-hour stop-loss prohibit overlapping owned training jobs. Existing completed GRU/STID records may be reused only as explicitly labeled development evidence when configurations and predictor hashes match. Their missing contemporaneous launch bindings require fresh formal five-seed training; no retrospective binding is fabricated.

Core widths and genuine SNN semantics remain fixed. This resource contract chooses batch/device/cache settings, not predictor performance. The email episode uses its own fit and is a same-source coverage diagnostic. The main email validation partition is genuinely zero-target and cannot establish positive epsilon calibration.
