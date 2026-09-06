"""Run the complete pipeline locally; this script never calls AWS."""
import asyncio
from datetime import date, time

from agents.physical_activity_model import PhysicalActivityModel
from agents.sleep_model import SleepModel
from agents.student_wellness import StudentWellnessPipeline
from agents.wellness_components import FixedSelector, LocalAdviceModel
from schema.student_wellness import StudentSchedule, WellnessSignals


async def main():
    schedule = StudentSchedule(student_id="demo", day=date.today())
    signals = WellnessSignals(sleep_duration_hours=6, active_days_per_week=2, strength_sessions_per_week=1)
    pipeline = StudentWellnessPipeline(
        models={"sleep": SleepModel(), "physical_activity": PhysicalActivityModel()},
        selector=FixedSelector(),
        advice=LocalAdviceModel(),
    )
    print((await pipeline.run(schedule, signals, time(20))).model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
