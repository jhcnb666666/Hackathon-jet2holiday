"""Framework for specialist wellness models and final AWS-LLM advice."""

import asyncio
from datetime import datetime, time
from typing import Protocol

from schema.student_wellness import (
    MetricScore,
    Recommendation,
    StudentSchedule,
    WellnessReport,
    WellnessSignals,
)


class SpecialistModel(Protocol):
    """Contract implemented later by each sleep/activity/diet/work model."""

    name: str

    async def score(
        self, schedule: StudentSchedule, signals: WellnessSignals
    ) -> MetricScore: ...


class ModelSelector(Protocol):
    async def select(self, schedule: StudentSchedule, triggered_at: time) -> list[str]: ...


class AdviceModel(Protocol):
    async def generate(
        self, schedule: StudentSchedule, scores: list[MetricScore]
    ) -> Recommendation: ...


class StudentWellnessPipeline:
    """Orchestrates selection, specialist scoring, and advice generation.

    No specialist implementation is bundled here. Inject the real models through
    ``models`` and the AWS Bedrock-backed final advice model through ``advice``.
    """

    def __init__(
        self,
        models: dict[str, SpecialistModel],
        selector: ModelSelector,
        advice: AdviceModel,
    ) -> None:
        self.models = models
        self.selector = selector
        self.advice = advice

    async def run(
        self,
        schedule: StudentSchedule,
        signals: WellnessSignals,
        triggered_at: time | None = None,
    ) -> WellnessReport:
        trigger = triggered_at or datetime.now().time().replace(microsecond=0)
        requested = await self.selector.select(schedule, trigger)
        selected = [
            name for name in dict.fromkeys(requested)
            if name in self.models
        ]
        if not selected:
            raise ValueError("No specialist model was selected")

        scores = list(
            await asyncio.gather(
                *(self.models[name].score(schedule, signals) for name in selected)
            )
        )
        recommendation = await self.advice.generate(schedule, scores)
        return WellnessReport(
            student_id=schedule.student_id,
            generated_at=datetime.now(),
            triggered_at=trigger,
            selected_metrics=[score.metric for score in scores],
            scores=scores,
            recommendation=recommendation,
        )
