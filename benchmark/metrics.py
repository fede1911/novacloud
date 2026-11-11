"""Deterministic metrics of the benchmark: routing, escalation, expected tools, aggregates."""

import dataclasses
import statistics
from typing import Literal

OPEN_TICKET_TOOL_PREFIX = "open_ticket_"
RoutingOutcome = Literal["exact", "over_routing", "under_routing", "wrong"]


@dataclasses.dataclass(frozen=True)
class RoutingScore:
    """Routing of the first turn compared with the gold scopes.

    Attributes:
        exact_match: Whether the chosen scopes are exactly the gold ones (order ignored).
        jaccard: Intersection over union of chosen and gold scopes (1.0 when both are empty).
        outcome: exact, over_routing (gold plus extra scopes), under_routing (a subset of the gold
            ones, including no choice) or wrong.
    """

    exact_match: bool
    jaccard: float
    outcome: RoutingOutcome


def score_routing(
    *, gold_scopes: list[str], predicted_scopes: list[str]
) -> RoutingScore:
    """Compares the scopes chosen by the triage with the gold ones."""
    gold, predicted = set(gold_scopes), set(predicted_scopes)
    union = gold | predicted
    jaccard = len(gold & predicted) / len(union) if union else 1.0
    if gold == predicted:
        outcome: RoutingOutcome = "exact"
    elif gold < predicted:
        outcome = "over_routing"
    elif predicted < gold:
        outcome = "under_routing"
    else:
        outcome = "wrong"
    return RoutingScore(
        exact_match=gold == predicted, jaccard=round(jaccard, 3), outcome=outcome
    )


def opened_ticket(tool_calls: list[tuple[str, str]]) -> bool:
    """Tells whether a ticket was opened, from the (tool name, status) of the conversation."""
    return any(
        name.startswith(OPEN_TICKET_TOOL_PREFIX) and status == "success"
        for name, status in tool_calls
    )


def expected_tools_hit_rate(
    *, expected_tools: list[str], used_tool_names: list[str]
) -> float | None:
    """Share of the expected tools (names or prefixes) used at least once, None if none is expected."""
    if not expected_tools:
        return None
    hits = [
        expected
        for expected in expected_tools
        if any(name.startswith(expected) for name in used_tool_names)
    ]
    return round(len(hits) / len(expected_tools), 3)


def mean_or_none(values: list[float | None]) -> float | None:
    """Mean of the values that are not None, rounded; None if there are none."""
    present = [value for value in values if value is not None]
    return round(statistics.fmean(present), 3) if present else None
