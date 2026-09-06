"""Local Physical Activity specialist.

This component uses only physical-activity features. Academic fields, including
GPA, are intentionally absent from its interface and calculations.
"""

from schema.student_wellness import MetricScore, StudentSchedule, WellnessSignals


class PhysicalActivityModel:
    name = "physical_activity"

    async def score(self, schedule: StudentSchedule, signals: WellnessSignals) -> MetricScore:
        """Return a 0-100 activity score without making an external LLM call."""
        active_days = signals.active_days_per_week
        strength_days = signals.strength_sessions_per_week
        minutes = signals.exercise_minutes

        components: list[tuple[float, float]] = []
        evidence: list[str] = []
        if active_days is not None:
            components.append((min(active_days / 5, 1) * 60, 60))
            evidence.append(f"{active_days:g} physically active days per week")
        if strength_days is not None:
            components.append((min(strength_days / 2, 1) * 20, 20))
            evidence.append(f"{strength_days:g} strength-training days per week")
        if minutes is not None:
            components.append((min(minutes / 150, 1) * 20, 20))
            evidence.append(f"{minutes:g} exercise minutes")

        score = sum(value for value, _ in components) / sum(weight for _, weight in components) * 100 if components else 50.0
        level = "good" if score >= 80 else "attention" if score >= 50 else "poor"
        if not evidence:
            evidence.append("No physical-activity measurements provided")
        return MetricScore(metric=self.name, score=round(score, 1), level=level, evidence=evidence)
