"""Small CSV-to-signal loader; explicitly ignores GPA columns."""
import csv
import re
from pathlib import Path

from schema.student_wellness import WellnessSignals


def _number(value: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", value or "")
    return float(match.group()) if match else 0.0


def load_physical_activity_row(path: str | Path, row_number: int = 0) -> WellnessSignals:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[row_number]
    return WellnessSignals(
        active_days_per_week=_number(row["number_of_days_physically_active"]),
        strength_sessions_per_week=_number(row["strength_training"]),
    )
