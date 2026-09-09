"""Deterministic categorical k-modes trainer for sleep patterns.

The model is deliberately GPA-free.  It learns only from sleep duration,
bedtime and wake-time categories, so the resulting artifact can describe
common sleep patterns without pretending to predict academic performance.
"""
import csv
import json
from collections import Counter
from pathlib import Path

FEATURE_NAMES = ("sleep_duration", "bedtime", "wake_time")
FEATURE_WEIGHTS = (2.0, 1.0, 1.0)


def bedtime_category(hour: int, minute: int = 0) -> str:
    if 22 <= hour <= 23:
        return "Between 10pm and 12am"
    if 0 <= hour < 2:
        return "Between 12am and 2am"
    if 2 <= hour < 6:
        return "After 2am"
    if 20 <= hour < 22:
        return "Between 8pm and 10pm"
    return "Before 8pm"


def wake_time_category(hour: int, minute: int = 0) -> str:
    """Convert a clock time to the categories used by the survey."""
    if hour < 6:
        return "Before 6am"
    if hour < 8:
        return "Between 6am and 8am"
    if hour < 10:
        return "Between 8am and 10am"
    if hour < 12:
        return "Between 10am and 12pm"
    return "After 12pm"


def _distance(pattern: tuple[str, ...], mode: tuple[str, ...]) -> float:
    """Weighted Hamming distance used by k-modes."""
    return sum(
        weight for value, prototype, weight in zip(pattern, mode, FEATURE_WEIGHTS, strict=True)
        if value != prototype
    )


def _initial_modes(patterns: list[tuple[str, ...]], clusters: int) -> list[tuple[str, ...]]:
    """Choose deterministic, well-separated starting modes."""
    counts = Counter(patterns)
    candidates = sorted(counts, key=lambda item: (-counts[item], item))
    modes = [candidates[0]]
    while len(modes) < clusters:
        remaining = [item for item in candidates if item not in modes]
        modes.append(
            max(
                remaining,
                key=lambda item: (
                    min(_distance(item, mode) for mode in modes),
                    counts[item],
                    tuple(item),
                ),
            )
        )
    return modes


def _categorical_mode(rows: list[tuple[str, ...]]) -> tuple[str, ...]:
    """Return the most frequent value in each feature column."""
    values: list[str] = []
    for index in range(len(FEATURE_NAMES)):
        counts = Counter(row[index] for row in rows)
        highest = max(counts.values())
        values.append(min(value for value, count in counts.items() if count == highest))
    return tuple(values)


def train_artifact(path: str | Path, clusters: int = 4, max_iterations: int = 25) -> dict:
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    patterns = [
        tuple((row.get(feature) or "").strip() for feature in FEATURE_NAMES)
        for row in rows
    ]
    valid_patterns = [pattern for pattern in patterns if all(pattern)]
    skipped_rows = len(patterns) - len(valid_patterns)
    if not valid_patterns:
        raise ValueError("No complete sleep records were found")

    cluster_count = min(max(1, clusters), len(set(valid_patterns)))
    modes = _initial_modes(valid_patterns, cluster_count)
    assignments: list[int] = []
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        assignments = [
            min(range(cluster_count), key=lambda index: (_distance(pattern, modes[index]), index))
            for pattern in valid_patterns
        ]
        updated = []
        for index, old_mode in enumerate(modes):
            members = [
                pattern for pattern, assignment in zip(valid_patterns, assignments, strict=True)
                if assignment == index
            ]
            updated.append(_categorical_mode(members) if members else old_mode)
        if updated == modes:
            break
        modes = updated

    # Recalculate membership against the final modes before reporting sizes.
    assignments = [
        min(range(cluster_count), key=lambda index: (_distance(pattern, modes[index]), index))
        for pattern in valid_patterns
    ]

    learned = []
    total = len(valid_patterns)
    for index, mode in enumerate(modes):
        size = assignments.count(index)
        learned.append(
            {
                "id": index,
                "size": size,
                "share": round(size / total, 6),
                "prototype": dict(zip(FEATURE_NAMES, mode, strict=True)),
            }
        )
    learned.sort(key=lambda item: (-item["size"], item["id"]))
    for index, item in enumerate(learned):
        item["id"] = index

    return {
        "schema_version": 1,
        "model_type": "categorical_k_modes",
        "features": list(FEATURE_NAMES),
        "feature_weights": dict(zip(FEATURE_NAMES, FEATURE_WEIGHTS, strict=True)),
        "excluded_columns": ["record_id", "current_gpa", "gpa_quality_flag"],
        "training_rows": total,
        "skipped_rows": skipped_rows,
        "clusters": learned,
        "training": {
            "algorithm": "deterministic weighted k-modes",
            "iterations": iterations,
        },
        "scoring": {
            "scope": "adult sleep-duration adequacy, not clinical sleep quality",
            "adult_reference_hours": 7,
            "reference_urls": [
                "https://www.cdc.gov/sleep/about/index.html",
                "https://aasm.org/resources/pdf/pressroom/adult-sleep-duration-consensus.pdf",
            ],
        },
    }


def save_artifact(artifact: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
