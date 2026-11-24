# Root engineering acceptance: oracle-free graph replay v8

Root reviewed the narrow graph7-to-graph8 change and actual full-origin replay.
The existing model, normalization, graph/state trace, B1 FP32 and source/resource/
budget fences remain. Oracle decoding, switch-relative arrays and alignment
serialization are removed. Complete forecasts/targets/masks/origins/model seed/
trajectory seed are emitted before any later reporting. Independent25 CPU toy
fixtures pass; root's combined candidate/graph fixture run passed36 cases.

The local phase receipt records origin_count but lacks per-file output hashes.
Actual root dispatch and global completion collection must bind all output bytes
and predecessor evidence. That integration is pending, so this engineering
acceptance does not authorize any actual test launch or oracle analysis. A later
CPU reporting helper may use oracle data only after root accepts the complete
global frozen-test campaign. No actual graph8 test or H1/H2 result occurred.
