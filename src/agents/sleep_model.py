"""Runtime adapter for the checked-in sleep clustering artifact."""
import json
from pathlib import Path

from schema.student_wellness import MetricScore, StudentSchedule, WellnessSignals
from sleep_clustering import bedtime_category, wake_time_category


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
        onset = signals.sleep_onset_time or schedule.sleep_time
        wake = signals.wake_time or schedule.wake_time
        prototype = self._nearest_pattern(hours, onset, wake)
        if prototype:
            evidence.append(
                "Nearest survey pattern: "
                f"{prototype['sleep_duration']}, {prototype['bedtime']}, {prototype['wake_time']} "
                f"({prototype['share']:.1%} of training cases)"
            )
        return MetricScore(metric=self.name, score=round(score, 1), level=level, evidence=evidence)

    def _nearest_pattern(self, hours, onset, wake):
        observed = {
            "sleep_duration": f"{int(round(hours))} hours",
            "bedtime": bedtime_category(onset.hour, onset.minute) if onset else None,
            "wake_time": wake_time_category(wake.hour, wake.minute) if wake else None,
        }
        candidates = self.artifact.get("clusters", [])
        if not candidates:
            return None

        weights = self.artifact.get(
            "feature_weights",
            {"sleep_duration": 2.0, "bedtime": 1.0, "wake_time": 1.0},
        )

        def distance(cluster):
            prototype = cluster["prototype"]
            return sum(
                weights.get(feature, 1.0)
                for feature, value in observed.items()
                if value is not None and prototype.get(feature) != value
            )

        nearest = min(candidates, key=lambda cluster: (distance(cluster), cluster["id"]))
        return {**nearest["prototype"], "share": nearest["share"]}
