"""Default non-network components for wiring and testing the pipeline."""
from datetime import time

from schema.student_wellness import MetricScore, Recommendation, StudentSchedule


class FixedSelector:
    async def select(self, schedule, triggered_at):
        return ["sleep", "physical_activity"]


class ScheduleAwareSelector:
    """Select specialists using check-in time and the student's timetable.

    The rules are intentionally small and visible.  They can later be replaced
    by a learned selector without changing the pipeline interface.
    """

    def __init__(self, sedentary_minutes_threshold: int = 240) -> None:
        self.sedentary_minutes_threshold = sedentary_minutes_threshold

    async def select(self, schedule: StudentSchedule, triggered_at: time) -> list[str]:
        selected: list[str] = []
        hour = triggered_at.hour

        # Sleep is most useful at the morning and evening check-ins.
        if hour < 10 or hour >= 20:
            selected.append("sleep")

        # Activity is checked during the active part of the day.
        if 10 <= hour < 22:
            selected.append("physical_activity")

        sedentary_minutes = sum(
            max(0, int((item.end - item.start).total_seconds() // 60))
            for item in schedule.items
            if item.category in {"class", "study"}
        )
        if sedentary_minutes >= self.sedentary_minutes_threshold:
            selected.append("physical_activity")

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(selected or ["sleep"]))


class LocalAdviceModel:
    async def generate(self, schedule: StudentSchedule, scores: list[MetricScore]) -> Recommendation:
        worst = min(scores, key=lambda item: item.score)
        advice = {
            "sleep": "Try to get at least seven hours of sleep tonight.",
            "physical_activity": "Add a short walk or light workout to your schedule.",
        }.get(worst.metric, "Take a short break and protect your recovery time.")
        return Recommendation(text=advice, priority="high" if worst.level == "poor" else "medium", based_on=[worst.metric])


class FallbackAdviceModel:
    """Use an external advice model when available and fall back locally."""

    def __init__(self, primary, fallback: LocalAdviceModel | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or LocalAdviceModel()

    async def generate(self, schedule: StudentSchedule, scores: list[MetricScore]) -> Recommendation:
        try:
            return await self.primary.generate(schedule, scores)
        except Exception:
            return await self.fallback.generate(schedule, scores)
