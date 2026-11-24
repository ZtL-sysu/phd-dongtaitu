# Cost estimate diagnostic counter repair

Root review after the focused calibration repair found a missing runtime
`estimated_cost` aggregate: calibration was reading a nonexistent counter as
zero. The runtime now accumulates each actually planned bin's estimated cost.
Full/off/G-only/S-only plans include the fixed measured base and report any
time-budget exceedance. Mandatory-cost remains the incremental mandatory G/S
action estimate; the fixed base is separately visible in the cost table.

Calibration exposes both estimated allocation cost per window and per observed
bin, with explicit names. These estimates are not measured runtime or evidence
of acceleration. No model equation, observation, action choice or gradient is
changed. The focused regression checks accumulated values, fixed base, full
budget overruns and identical full-reference predictions. Earlier worker reports
and their source hashes are preserved as historical, preceding this diagnostic
correction. Updated code identity is bound in the P2 delta snapshot.
