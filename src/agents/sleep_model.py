"""Local sleep specialist backed by the trained survey-pattern artifact."""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path
from typing import Any

from sleep_clustering import (
    FEATURE_NAMES,
    bedtime_category,
    duration_category,
    wake_time_category,
)
from schema.student_wellness import MetricScore, StudentSchedule, WellnessSignals

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "sleep_model.json"


class SleepModel:
    """Summarize sleep adequacy and attach the nearest learned survey pattern."""

    name = "sleep"

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.artifact = json.loads(self.model_path.read_text(encoding="utf-8"))
        if self.artifact.get("features") != list(FEATURE_NAMES):
            raise ValueError("Sleep model artifact has an incompatible feature schema")

    async def score(self, schedule: StudentSchedule, signals: WellnessSignals) -> MetricScore:
        onset = signals.sleep_onset_time or schedule.sleep_time
        wake = signals.wake_time or schedule.wake_time
        duration = signals.sleep_duration_hours
        if duration is None and onset is not None and wake is not None:
            duration = _overnight_hours(onset, wake)

        survey_values: dict[str, str] = {}
        evidence: list[str] = []
        if duration is not None:
            survey_values["sleep_duration"] = duration_category(duration)
            comparison = "meets" if duration >= 7 else "is below"
            evidence.append(
                f"Sleep duration: {duration:.1f} hours ({comparison} the adult 7+ hour reference)"
            )
        if onset is not None:
            survey_values["bedtime"] = bedtime_category(onset.hour, onset.minute)
            evidence.append(f"Sleep onset: {onset.strftime('%H:%M')}")
        if wake is not None:
            survey_values["wake_time"] = wake_time_category(wake.hour, wake.minute)
            evidence.append(f"Wake time: {wake.strftime('%H:%M')}")

        if survey_values:
            match = self._nearest_cluster(survey_values)
            prototype = match["prototype"]
            pattern = ", ".join(prototype[name] for name in FEATURE_NAMES)
            evidence.append(
                "Nearest survey pattern: "
                f"{pattern} ({match['share'] * 100:.1f}% of training cases)"
            )
        else:
            evidence.append("No sleep measurements provided")

        score = _duration_score(duration)
        level = "good" if score >= 80 else "attention" if score >= 50 else "poor"
        return MetricScore(
            metric=self.name,
            score=round(score, 1),
            level=level,
            evidence=evidence,
        )

    def _nearest_cluster(self, values: dict[str, str]) -> dict[str, Any]:
        def rank(cluster: dict[str, Any]) -> tuple[int, float, int]:
            prototype = cluster["prototype"]
            mismatches = sum(prototype[name] != value for name, value in values.items())
            return mismatches, -float(cluster["share"]), int(cluster["id"])

        return min(self.artifact["clusters"], key=rank)


def _overnight_hours(onset: time, wake: time) -> float:
    onset_minutes = onset.hour * 60 + onset.minute
    wake_minutes = wake.hour * 60 + wake.minute
    elapsed = (wake_minutes - onset_minutes) % (24 * 60)
    return elapsed / 60


def _duration_score(duration: float | None) -> float:
    """Score adult sleep-duration sufficiency without claiming clinical quality."""
    if duration is None:
        return 50.0
    if duration >= 7:
        return 100.0
    if duration >= 6:
        return 65.0
    if duration >= 5:
        return 40.0
    return 20.0
