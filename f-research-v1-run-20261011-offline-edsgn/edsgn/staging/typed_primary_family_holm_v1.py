"""Pure typed Holm kernel for already provenance-validated endpoint rows.

This module deliberately performs no file access, provenance checks, or phase
gating. Callers must bind the structural registry and numeric inputs before
calling :func:`assemble_primary_families` and revalidate them afterwards.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping


EXPECTED_FAMILY_SIZES = {"H1": 9, "H2": 16, "H3": 24}
SUPERIORITY_TYPES = {
    "graph_tv_noise_superiority",
    "graph_tv_response_superiority",
    "forecast_mae_superiority",
    "paired_window_cost_superiority",
}
NONINFERIORITY_TYPE = "forecast_mae_noninferiority"
SUPPORTED_TYPES = SUPERIORITY_TYPES | {NONINFERIORITY_TYPE}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _holm(p_values: list[float]) -> list[float]:
    """Return Holm adjusted p-values in original member order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def assemble_primary_families(
    registry_rows: Iterable[Mapping[str, Any]],
    numeric_rows: Iterable[Mapping[str, Any]],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Assemble H1/H2/H3 from an injected structural registry and result rows.

    Numeric row schema: ``endpoint_id``, ``outcome_type``, ``identifiable``.
    Identifiable superiority rows additionally provide ``raw_p``, ``estimate``
    and ``percentile95_upper``. Noninferiority rows provide ``raw_p``,
    ``estimate``, ``one_sided95_upper`` and a positive finite ``epsilon``.
    An explicitly unidentifiable row must have ``raw_p: None``; this blocks
    Holm adjustment for its entire family while preserving the full registry.
    ``supported: false`` is a distinct negative status for an identifiable
    endpoint and never means unidentifiable.
    """
    alpha = _finite_number(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")

    registry = list(registry_rows)
    results = list(numeric_rows)
    if any(not isinstance(r, Mapping) for r in registry + results):
        raise ValueError("registry and numeric rows must be mappings")

    def index_unique(rows: list[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
        indexed: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            endpoint_id = row.get("endpoint_id")
            if not isinstance(endpoint_id, str) or not endpoint_id:
                raise ValueError(f"{label} endpoint_id must be a nonempty string")
            if endpoint_id in indexed:
                raise ValueError(f"duplicate {label} endpoint_id: {endpoint_id}")
            indexed[endpoint_id] = row
        return indexed

    reg_by_id = index_unique(registry, "registry")
    result_by_id = index_unique(results, "numeric")
    if set(reg_by_id) != set(result_by_id):
        missing = sorted(set(reg_by_id) - set(result_by_id))
        extra = sorted(set(result_by_id) - set(reg_by_id))
        raise ValueError(f"endpoint ID set mismatch; missing={missing}; extra={extra}")

    expected_counts = Counter()
    for endpoint_id, row in reg_by_id.items():
        family = endpoint_id.split("-", 1)[0]
        if family not in EXPECTED_FAMILY_SIZES:
            raise ValueError(f"unexpected family in endpoint_id: {endpoint_id}")
        expected_counts[family] += 1
        outcome_type = row.get("outcome_type")
        if outcome_type not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported registry outcome_type for {endpoint_id}: {outcome_type}")
        numeric_type = result_by_id[endpoint_id].get("outcome_type")
        if numeric_type != outcome_type:
            raise ValueError(f"outcome_type mismatch for {endpoint_id}")
    if dict(expected_counts) != EXPECTED_FAMILY_SIZES:
        raise ValueError(f"registry family sizes mismatch: {dict(expected_counts)}")

    output: dict[str, Any] = {"alpha": alpha, "families": {}}
    for family, size in EXPECTED_FAMILY_SIZES.items():
        ids = [endpoint_id for endpoint_id in reg_by_id if endpoint_id.startswith(family + "-")]
        if len(ids) != size:
            raise ValueError(f"{family} must contain exactly {size} endpoints")
        members: list[dict[str, Any]] = []
        any_unidentifiable = False
        any_unsupported = False
        p_values: list[float] = []
        identifiable_positions: list[int] = []

        for endpoint_id in ids:
            desc = reg_by_id[endpoint_id]
            row = result_by_id[endpoint_id]
            kind = desc["outcome_type"]
            identifiable = row.get("identifiable")
            if not isinstance(identifiable, bool):
                raise ValueError(f"identifiable must be boolean for {endpoint_id}")
            member: dict[str, Any] = {
                "endpoint_id": endpoint_id,
                "outcome_type": kind,
                "identifiable": identifiable,
                "raw_p": None,
                "holm_adjusted_p": None,
                "directional_gate_passed": None,
                "supported": None,
            }
            if not identifiable:
                any_unidentifiable = True
                if row.get("raw_p") is not None:
                    raise ValueError(f"unidentifiable endpoint must have raw_p=null: {endpoint_id}")
                members.append(member)
                continue

            raw_p = _finite_number(row.get("raw_p"), f"raw_p for {endpoint_id}")
            if not 0 <= raw_p <= 1:
                raise ValueError(f"raw_p outside [0,1] for {endpoint_id}")
            estimate = _finite_number(row.get("estimate"), f"estimate for {endpoint_id}")
            if kind == NONINFERIORITY_TYPE:
                epsilon_value = row.get("epsilon")
                if epsilon_value is None:
                    raise ValueError(f"identifiable noninferiority endpoint has null epsilon: {endpoint_id}")
                epsilon = _finite_number(epsilon_value, f"epsilon for {endpoint_id}")
                if epsilon <= 0:
                    raise ValueError(f"epsilon must be positive for {endpoint_id}")
                upper = _finite_number(row.get("one_sided95_upper"), f"one_sided95_upper for {endpoint_id}")
                direction_pass = upper < epsilon
                member.update({"estimate": estimate, "one_sided95_upper": upper, "epsilon": epsilon})
            else:
                upper = _finite_number(row.get("percentile95_upper"), f"percentile95_upper for {endpoint_id}")
                direction_pass = estimate < 0 and upper < 0
                member.update({"estimate": estimate, "percentile95_upper": upper})
            supported = row.get("supported", True)
            if not isinstance(supported, bool):
                raise ValueError(f"supported must be boolean for {endpoint_id}")
            any_unsupported |= not supported
            member.update({"raw_p": raw_p, "directional_gate_passed": direction_pass, "supported": supported})
            p_values.append(raw_p)
            identifiable_positions.append(len(members))
            members.append(member)

        family_status = (
            "inconclusive_unidentifiable_member" if any_unidentifiable
            else "unsupported_identifiable_member" if any_unsupported
            else "identified"
        )
        # Never run a reduced-size Holm family. Inconclusive families retain
        # diagnostic raw p-values but every adjusted p and decision remains null.
        if not any_unidentifiable:
            adjusted = _holm(p_values)
            for position, adj_p in zip(identifiable_positions, adjusted):
                members[position]["holm_adjusted_p"] = adj_p
                members[position]["endpoint_gate_passed"] = bool(
                    members[position]["supported"]
                    and members[position]["directional_gate_passed"]
                    and adj_p <= alpha
                )
            family_supported = family_status == "identified" and all(
                m["endpoint_gate_passed"] for m in members
            )
        else:
            for m in members:
                m["endpoint_gate_passed"] = None
            family_supported = None
        output["families"][family] = {
            "expected_size": size,
            "status": family_status,
            "members": members,
            "all_directional_gates_passed": family_supported,
            "holm_denominator": None if any_unidentifiable else size,
        }
    return output
