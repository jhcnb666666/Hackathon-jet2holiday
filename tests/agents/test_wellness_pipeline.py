from datetime import date, datetime, time

import pytest

from agents.physical_activity_model import PhysicalActivityModel
from agents.sleep_model import SleepModel
from agents.student_wellness import StudentWellnessPipeline
from agents.wellness_components import (
    FallbackAdviceModel,
    LocalAdviceModel,
    ScheduleAwareSelector,
)
from schema.student_wellness import ScheduleItem, StudentSchedule, WellnessSignals


@pytest.mark.asyncio
async def test_selector_uses_time_of_day():
    selector = ScheduleAwareSelector()
    schedule = StudentSchedule(student_id="demo", day=date(2026, 9, 9))

    assert await selector.select(schedule, time(8)) == ["sleep"]
    assert await selector.select(schedule, time(12)) == ["physical_activity"]
    assert await selector.select(schedule, time(20)) == ["sleep", "physical_activity"]
    assert await selector.select(schedule, time(22)) == ["sleep"]


@pytest.mark.asyncio
async def test_long_class_day_adds_activity_specialist():
    selector = ScheduleAwareSelector()
    schedule = StudentSchedule(
        student_id="demo",
        day=date(2026, 9, 9),
        items=[
            ScheduleItem(
                start=datetime(2026, 9, 9, 8),
                end=datetime(2026, 9, 9, 13),
                title="Classes",
                category="class",
            )
        ],
    )

    assert await selector.select(schedule, time(8)) == ["sleep", "physical_activity"]


class BrokenAdviceModel:
    async def generate(self, schedule, scores):
        raise RuntimeError("external service unavailable")


@pytest.mark.asyncio
async def test_advice_falls_back_when_external_model_fails():
    advice = FallbackAdviceModel(BrokenAdviceModel())
    schedule = StudentSchedule(student_id="demo", day=date(2026, 9, 9))
    score = await PhysicalActivityModel().score(
        schedule,
        WellnessSignals(active_days_per_week=2, strength_sessions_per_week=1),
    )

    result = await advice.generate(schedule, [score])

    assert result.based_on == ["physical_activity"]
    assert "walk" in result.text.lower()


@pytest.mark.asyncio
async def test_complete_local_pipeline_returns_explainable_report():
    schedule = StudentSchedule(student_id="demo", day=date(2026, 9, 9))
    signals = WellnessSignals(
        sleep_duration_hours=6,
        active_days_per_week=2,
        strength_sessions_per_week=1,
    )
    pipeline = StudentWellnessPipeline(
        models={"sleep": SleepModel(), "physical_activity": PhysicalActivityModel()},
        selector=ScheduleAwareSelector(),
        advice=LocalAdviceModel(),
    )

    report = await pipeline.run(schedule, signals, time(20))

    assert report.selected_metrics == ["sleep", "physical_activity"]
    assert [score.score for score in report.scores] == [65.0, 42.5]
    assert report.recommendation.based_on == ["physical_activity"]
