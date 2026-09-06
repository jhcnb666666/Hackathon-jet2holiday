"""Scheduling adapter. The callback can be wired to APScheduler/Celery later."""

from datetime import time
from typing import Awaitable, Callable

from schema.student_wellness import StudentSchedule, WellnessReport, WellnessSignals


class DailyTrigger:
    def __init__(self, trigger_time: time, callback: Callable[[StudentSchedule, WellnessSignals], Awaitable[WellnessReport]]):
        self.trigger_time = trigger_time
        self.callback = callback

    async def fire(self, schedule: StudentSchedule, signals: WellnessSignals, now: time) -> WellnessReport | None:
        """Fire once the configured time is reached; integrate with a real clock externally."""
        if now >= self.trigger_time:
            return await self.callback(schedule, signals)
        return None
