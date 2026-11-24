# Endpoint sufficient statistics v4 writer evidence

V4 preserves v3 and rejects boolean, string, null, and complex timing values before float conversion; finite positive real timing values retain the prior aggregation semantics.

Local `dyg-cold` CPU verification is included in the combined dispatcher/endpoint command: 27 tests passed, comprising 12 dispatcher mocks and 15 endpoint converter tests. This is engineering-only; no endpoint data or model results were read.

SHA-256: source `AFBBD2F459CC680F2689334FE69DECC63F87D37308CBBE6956C62C3F994A2519`; tests `E9419DD7D910F6B4BB836F152C57C5B30384B1D8C6AF9926F024BA7207C46D0E`.
