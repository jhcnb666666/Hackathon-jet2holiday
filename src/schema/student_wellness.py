"""Schemas for the student schedule -> wellness recommendation pipeline."""

from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, Field


class ScheduleItem(BaseModel):
    """One planned or completed activity in a student's day."""

    start: datetime
    end: datetime
    title: str
    category: Literal["class", "study", "exercise", "meal", "sleep", "social", "other"] = "other"
    notes: str | None = None


class StudentSchedule(BaseModel):
    student_id: str
    day: date
    items: list[ScheduleItem] = Field(default_factory=list)
    wake_time: time | None = None
    sleep_time: time | None = None


class WellnessSignals(BaseModel):
    """生活数据；明确不包含 GPA。"""

    sleep_duration_hours: float | None = Field(default=None, ge=0, le=24)
    sleep_onset_time: time | None = None
    wake_time: time | None = None
    exercise_minutes: float | None = Field(default=None, ge=0)
    active_days_per_week: float | None = Field(default=None, ge=0, le=7)
    strength_sessions_per_week: float | None = Field(default=None, ge=0)
    eating_regularity: float | None = Field(default=None, ge=0, le=10)
    healthy_food_frequency: float | None = Field(default=None, ge=0, le=10)
    weekly_work_hours: float | None = Field(default=None, ge=0)
    work_frequency: float | None = Field(default=None, ge=0)
    biological_sex: str | None = None
    year_level: int | None = Field(default=None, ge=1)


class MetricScore(BaseModel):
    metric: str
    score: float = Field(ge=0, le=100)
    level: Literal["good", "attention", "poor"]
    evidence: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    text: str
    priority: Literal["low", "medium", "high"] = "medium"
    based_on: list[str] = Field(default_factory=list)


class WellnessReport(BaseModel):
    student_id: str
    generated_at: datetime
    triggered_at: time
    selected_metrics: list[str]
    scores: list[MetricScore]
    recommendation: Recommendation
