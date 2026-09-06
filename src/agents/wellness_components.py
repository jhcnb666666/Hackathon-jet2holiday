"""Default non-network components for wiring and testing the pipeline."""
from schema.student_wellness import MetricScore, Recommendation, StudentSchedule


class FixedSelector:
    async def select(self, schedule, triggered_at):
        return ["sleep", "physical_activity"]


class LocalAdviceModel:
    async def generate(self, schedule: StudentSchedule, scores: list[MetricScore]) -> Recommendation:
        worst = min(scores, key=lambda item: item.score)
        advice = {
            "sleep": "Try to get at least seven hours of sleep tonight.",
            "physical_activity": "Add a short walk or light workout to your schedule.",
        }.get(worst.metric, "Take a short break and protect your recovery time.")
        return Recommendation(text=advice, priority="high" if worst.level == "poor" else "medium", based_on=[worst.metric])
