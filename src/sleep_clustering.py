"""Small deterministic categorical sleep-pattern trainer.

It is intentionally GPA-free: only sleep duration, bedtime and wake time are
read from the survey.
"""
import csv
import json
from collections import Counter
from pathlib import Path

FEATURE_NAMES = ("sleep_duration", "bedtime", "wake_time")


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


def train_artifact(path: str | Path, clusters: int = 4) -> dict:
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    patterns = Counter(tuple(row.get(feature, "") for feature in FEATURE_NAMES) for row in rows)
    total = sum(patterns.values())
    learned = []
    for index, (pattern, size) in enumerate(sorted(patterns.items(), key=lambda item: (-item[1], item[0]))[:clusters]):
        learned.append({"id": index, "size": size, "share": round(size / total, 6), "prototype": dict(zip(FEATURE_NAMES, pattern, strict=True))})
    return {"schema_version": 1, "model_type": "categorical_patterns", "features": list(FEATURE_NAMES), "excluded_columns": ["record_id", "current_gpa", "gpa_quality_flag"], "training_rows": total, "skipped_rows": 0, "clusters": learned}


def save_artifact(artifact: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
