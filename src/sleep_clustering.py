"""Small, dependency-free categorical model for the sleep survey."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

FEATURE_NAMES = ("sleep_duration", "bedtime", "wake_time")

DURATION_CATEGORIES = (
    "4 hours or less",
    "5 hours",
    "6 hours",
    "7 hours",
    "8 hours",
    "9 hours",
    "10 or more hours",
)
BEDTIME_CATEGORIES = (
    "Before 8pm",
    "Between 8pm and 10pm",
    "Between 10pm and 12am",
    "Between 12am and 2am",
    "After 2am",
)
WAKE_TIME_CATEGORIES = (
    "Before 6am",
    "Between 6am and 8am",
    "Between 8am and 10am",
    "Between 10am and 12pm",
    "After 12pm",
)

CATEGORY_ORDER = {
    "sleep_duration": DURATION_CATEGORIES,
    "bedtime": BEDTIME_CATEGORIES,
    "wake_time": WAKE_TIME_CATEGORIES,
}

REFERENCE_URLS = (
    "https://www.cdc.gov/sleep/about/index.html",
    "https://aasm.org/resources/pdf/pressroom/adult-sleep-duration-consensus.pdf",
)

SurveyRow = tuple[str, str, str]


def _as_tuple(row: dict[str, str]) -> SurveyRow:
    values = tuple((row.get(name) or "").strip() for name in FEATURE_NAMES)
    return values  # type: ignore[return-value]


def _as_dict(row: SurveyRow) -> dict[str, str]:
    return dict(zip(FEATURE_NAMES, row, strict=True))


def _validate_row(row: SurveyRow) -> None:
    for index, feature in enumerate(FEATURE_NAMES):
        if row[index] not in CATEGORY_ORDER[feature]:
            raise ValueError(f"Unsupported {feature} category: {row[index]!r}")


def load_survey_rows(path: str | Path) -> tuple[list[SurveyRow], int]:
    """Load only sleep columns; GPA and every other column are deliberately ignored."""
    rows: list[SurveyRow] = []
    skipped = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FEATURE_NAMES).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing required sleep columns: {sorted(missing)}")
        for source_row in reader:
            row = _as_tuple(source_row)
            if not all(row):
                skipped += 1
                continue
            _validate_row(row)
            rows.append(row)
    if not rows:
        raise ValueError("No complete sleep survey rows were found")
    return rows, skipped


def _distance(left: SurveyRow, right: SurveyRow) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def _initial_modes(row_counts: Counter[SurveyRow], clusters: int) -> list[SurveyRow]:
    ordered_rows = sorted(row_counts, key=lambda row: (-row_counts[row], row))
    modes = [ordered_rows[0]]
    while len(modes) < clusters:
        candidates = [row for row in ordered_rows if row not in modes]
        selected = max(
            candidates,
            key=lambda row: (
                min(_distance(row, mode) for mode in modes),
                row_counts[row],
                row,
            ),
        )
        modes.append(selected)
    return modes


def _nearest_mode(row: SurveyRow, modes: list[SurveyRow]) -> int:
    return min(range(len(modes)), key=lambda index: (_distance(row, modes[index]), index))


def fit_k_modes(
    rows: list[SurveyRow], clusters: int = 4, max_iterations: int = 100
) -> tuple[list[SurveyRow], list[int], int]:
    """Fit deterministic weighted k-modes to categorical survey rows."""
    row_counts = Counter(rows)
    if clusters < 1 or clusters > len(row_counts):
        raise ValueError(f"clusters must be between 1 and {len(row_counts)}")

    modes = _initial_modes(row_counts, clusters)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        assignments: list[list[tuple[SurveyRow, int]]] = [[] for _ in modes]
        for row, count in row_counts.items():
            assignments[_nearest_mode(row, modes)].append((row, count))

        new_modes: list[SurveyRow] = []
        for cluster_index, members in enumerate(assignments):
            if not members:
                new_modes.append(modes[cluster_index])
                continue
            values: list[str] = []
            for feature_index, feature in enumerate(FEATURE_NAMES):
                counts: Counter[str] = Counter()
                for row, count in members:
                    counts[row[feature_index]] += count
                order = CATEGORY_ORDER[feature]
                values.append(
                    max(order, key=lambda value: (counts[value], -order.index(value)))
                )
            new_modes.append(tuple(values))  # type: ignore[arg-type]

        if new_modes == modes:
            break
        modes = new_modes

    cluster_sizes = [0] * len(modes)
    for row, count in row_counts.items():
        cluster_sizes[_nearest_mode(row, modes)] += count
    return modes, cluster_sizes, iterations


def train_artifact(path: str | Path, clusters: int = 4) -> dict[str, Any]:
    rows, skipped = load_survey_rows(path)
    modes, cluster_sizes, iterations = fit_k_modes(rows, clusters=clusters)
    total = len(rows)
    cluster_records = [
        {
            "id": cluster_id,
            "size": size,
            "share": round(size / total, 6),
            "prototype": _as_dict(mode),
        }
        for cluster_id, (mode, size) in enumerate(zip(modes, cluster_sizes, strict=True))
    ]
    return {
        "schema_version": 1,
        "model_type": "categorical_k_modes",
        "features": list(FEATURE_NAMES),
        "excluded_columns": ["record_id", "current_gpa", "gpa_quality_flag"],
        "training_rows": total,
        "skipped_rows": skipped,
        "clusters": cluster_records,
        "training": {
            "algorithm": "deterministic weighted k-modes",
            "iterations": iterations,
        },
        "scoring": {
            "scope": "adult sleep-duration adequacy, not clinical sleep quality",
            "adult_reference_hours": 7,
            "reference_urls": list(REFERENCE_URLS),
        },
    }


def save_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def duration_category(hours: float) -> str:
    if hours < 4.5:
        return "4 hours or less"
    if hours < 5.5:
        return "5 hours"
    if hours < 6.5:
        return "6 hours"
    if hours < 7.5:
        return "7 hours"
    if hours < 8.5:
        return "8 hours"
    if hours < 9.5:
        return "9 hours"
    return "10 or more hours"


def bedtime_category(hour: int, minute: int = 0) -> str:
    value = hour + minute / 60
    if value < 2:
        return "Between 12am and 2am"
    if 2 <= value < 12:
        return "After 2am"
    if value < 20:
        return "Before 8pm"
    if value < 22:
        return "Between 8pm and 10pm"
    if value < 24:
        return "Between 10pm and 12am"
    return "Between 12am and 2am"


def wake_time_category(hour: int, minute: int = 0) -> str:
    value = hour + minute / 60
    if value < 6:
        return "Before 6am"
    if value < 8:
        return "Between 6am and 8am"
    if value < 10:
        return "Between 8am and 10am"
    if value < 12:
        return "Between 10am and 12pm"
    return "After 12pm"
