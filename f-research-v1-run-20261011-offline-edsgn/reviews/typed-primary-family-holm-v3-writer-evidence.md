# Typed primary family Holm v3 writer evidence

V3 preserves v2 and adds fail-closed validation that every registered endpoint has `primary is True` and `holm_family` exactly matches the ID prefix (H1/H2/H3). The original 49-row registry, exact 9/16/24 family sizes, type/claim checks, and full-family unidentifiable behavior remain unchanged.

Local `dyg-cold` CPU: `pytest edsgn/tests/test_typed_primary_family_holm_v3.py -q` — 12 passed. New mutations cover false, null, and missing `primary`, and cross-family/null/missing `holm_family`. No endpoint values, scientific data, or claim inference were run.

SHA-256: source `68EFECAE5E1DEF030844FDD881F20E0509141FA9DE2BB32D0A738A4C8BB8E3F2`; tests `C6099202F2FD6BD2D65F8C3EA51CBE2B45E36979E875749982765B3EFBB0C552`.
