# Global frozen completion v4

Version 4 preserves the v3 contract and closes two receipt-validation gaps. It now requires full phase output hash maps for `simple_v3` and `case_v3`, and rejects duplicate resolved paths in `prediction_v5` phase output lists rather than allowing dictionary normalization to overwrite a prior entry.

The Conda CPU suite passed **65 tests** in 9.85 seconds; measured whole-command wall time was 10.6840675 seconds. The mixed fixture covers all seven supported drivers together with opaque artifacts. The two new omission tests reseal original receipt, alias, dispatcher, and completion hashes, so rejection exercises the missing-map rule rather than stale hashes. Prediction duplicate tests cover same-SHA and different-SHA duplicates. No NPZ payloads were decoded.

Source SHA-256: `E05EA055BE0852AECC3B9A0113B177C54A3567AD8ADF5C7BBB34AA08A887A5C3`. This is engineering-only metadata verification. It does not run test evaluation, dispatch jobs, or establish scientific acceptance.
