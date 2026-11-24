# Root acceptance: episode DGCRN seed0 validation

The actual full train/validation run completed30 epochs, best epoch20, with334
training windows and46 ordered validation origins. Root rehashed every25-input/
7-output phase item and the unchanged checkpoint, without model loading.
Validation NumPy reconstruction verified source target/mask equality at declared
validation indices. The independent reviewer recomputed saved horizon1/3/7
metrics and receipt/binding hashes, but read only source metadata and did not
repeat source target-array reconstruction. This limitation is retained.

Required-horizon equal mean raw MAE is0.626444543037279; required-cell pooled
RMSE is2.279802970837173. The selection-log MAE differs only by saved floating
point precision. Training wall2494.7841156seconds and its managed phase are
deduplicated using their maximum in the same cumulative72-hour ledger.

This is one secondary episode seed0 development result. P3 remains running;
five-seed primary comparisons, final tests and H1/H2/H3 remain pending. Source
target-array loading can materialize other splits, while only validation indices
were scored. No test-origin metric, oracle join or hypothesis is accepted here.
