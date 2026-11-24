# P3 completed-campaign audit v6 preparation

Writer self-review, not an independent review. The new v6 route projects the canonical 80 P3 baseline jobs into 12 prior root-accepted rows and 68 host-recovery v2 physical rows. It verifies the frozen 68-row identity mapping, compares all training config fields except `output_dir`, pins the host queue/root acceptance and prior acceptance dependencies, and keeps the host-stopped attempt as a separate provenance record rather than another logical job. The existing validation-only numeric-v3 route remains unchanged; combined-NPZ materialization disclosure is retained.

CPU toy validation: 14 tests and 4 subtests passed in 3.45s (whole command wall 4.41s). Added tests cover exact 80=12+68 projection, rejection of a stale physical alias, and rejection of model/training config drift.

The host-recovery root acceptance file is still absent, so production preflight remains fail-closed. I did not call production preflight/campaign execution or read actual outputs; no numerical audit, training, model/checkpoint load, GPU, or test/oracle value access occurred. This is engineering preparation only.

Source SHA256: `ebf505415bb02e28541a6d0af4539e6d05d8f4de650e64da80dab98936ce8663`  
Test SHA256: `ce68ddc423e56fff15138c3d88f3cf92b9063283af7f4ee89ddcb75a137ce631`  
Test log SHA256: `c966565bdc9b4a4e95d333914b7a4b295c2df34853a0e1e04aa12e8f282972b8`
