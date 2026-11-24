# Root live-smoke rejection of auditor v4

The first actual one-shot invocation failed in `run_campaign` with
`KeyError: 'acceptance_path'`. Preflight returns the acceptance as a nested
`acceptance.path/sha256` object; the final collection fence expected top-level
fields. The failure is preserved in
`logs/completed-p3-campaign-root-audit-v1.log`. No aggregate output was produced,
no training was launched and no phase was advanced.

The earlier v4 root engineering acceptance is insufficient for the public
aggregate route and is superseded by this failed actual smoke. V4 must not be
used for later aggregates. Its source and evidence remain historical. A new v5
must fix the nested reference and pass a public complete/pending/exclusive-output
fixture before another root-controlled actual invocation. Existing development
numeric results retain their separate independent and root acceptances; this
wrapper failure does not invalidate or overwrite those artifacts.
