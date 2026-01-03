"""Metrics helpers for refusal experiments."""

from typing import Any


def average_refusal_rate(results: list[dict[str, Any]]) -> float:
    """Return average refusal rate across results."""
    if not results:
        return 0.0
    return sum(r["refusal_rate"] for r in results) / len(results)


def grouped_refusal_rates(
    results: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    """Return average refusal rates grouped by key."""
    buckets: dict[str, list[float]] = {}
    for result in results:
        bucket_key = str(result.get(key, "unknown"))
        buckets.setdefault(bucket_key, []).append(result["refusal_rate"])

    return {
        bucket: (sum(rates) / len(rates)) if rates else 0.0
        for bucket, rates in buckets.items()
    }


def tally_refusal_categories(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Count refusal categories by question category."""
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        category = str(record.get("category", "unknown"))
        label = record.get("refusal_category", "NO_REFUSAL")
        if label not in {"HARD_REFUSAL", "SOFT_REFUSAL", "NO_REFUSAL"}:
            label = "NO_REFUSAL"
        bucket = counts.setdefault(
            category,
            {"HARD_REFUSAL": 0, "SOFT_REFUSAL": 0, "NO_REFUSAL": 0},
        )
        bucket[label] += 1
    return counts
