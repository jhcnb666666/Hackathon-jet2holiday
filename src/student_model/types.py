from __future__ import annotations

from pydantic import BaseModel, Field


class Interaction(BaseModel):
    """One ordered student-question interaction."""

    student_id: int | str
    interaction_index: int = Field(ge=0)
    question_id: int = Field(ge=0)
    concept_id: int = Field(ge=0)
    difficulty: float = Field(ge=0.0, le=1.0)
    correct: int = Field(ge=0, le=1)
    response_time: float = Field(gt=0.0)
    hint_used: int = Field(ge=0, le=1)
    attempt_count: int = Field(ge=1)


class StudentState(BaseModel):
    """Compact learned state passed to the tutoring agent."""

    current_concept: str
    mastery: float = Field(ge=0.0, le=1.0)
    predicted_next_correct: float = Field(ge=0.0, le=1.0)
    struggling_probability: float = Field(ge=0.0, le=1.0)
    recent_trend: str
    needs_intervention: bool
    confidence: float = Field(ge=0.0, le=1.0)
