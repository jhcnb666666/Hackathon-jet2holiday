"""Runtime adapter for the checked-in sleep clustering artifact."""
import json
from pathlib import Path

from schema.student_wellness import MetricScore, StudentSchedule, WellnessSignals


class SleepModel:
    name = "sleep"

    def __init__(self, artifact_path: str | Path = "models/sleep_model.json"):
        self.artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

    async def score(self, schedule: StudentSchedule, signals: WellnessSignals) -> MetricScore:
        hours = signals.sleep_duration_hours
        if hours is None and schedule.sleep_time and schedule.wake_time:
            hours = (schedule.wake_time.hour - schedule.sleep_time.hour) % 24
        hours = hours if hours is not None else 0.0
        if hours >= 7:
            score = 100.0
        elif hours >= 6:
            score = 65.0
        elif hours >= 5:
            score = 40.0
        else:
            score = max(0.0, hours / 5 * 40)
        level = "good" if score >= 80 else "attention" if score >= 60 else "poor"
        evidence = [f"Sleep duration: {hours:.1f} hours"]
        if signals.sleep_onset_time:
            evidence.append(f"Sleep onset: {signals.sleep_onset_time.strftime('%H:%M')}")
        if signals.wake_time:
            evidence.append(f"Wake time: {signals.wake_time.strftime('%H:%M')}")
        prototype = self._nearest_pattern(hours, signals.sleep_onset_time, signals.wake_time)
        if prototype:
            evidence.append(
                "Nearest survey pattern: "
                f"{prototype['sleep_duration']}, {prototype['bedtime']}, {prototype['wake_time']} "
                f"({prototype['share']:.1%} of training cases)"
            )
        return MetricScore(metric=self.name, score=round(score, 1), level=level, evidence=evidence)

    def _nearest_pattern(self, hours, onset, wake):
        duration = f"{int(hours)} hours"
        candidates = self.artifact.get("clusters", [])
        if not candidates:
            return None
        def distance(cluster):
            prototype = cluster["prototype"]
            return int(prototype.get("sleep_duration") != duration)
        nearest = min(candidates, key=distance)
        return {**nearest["prototype"], "share": nearest["share"]}
