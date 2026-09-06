from __future__ import annotations

import numpy as np
import pandas as pd

from student_model.types import StudentState


def rule_based_student_state(history: pd.DataFrame) -> StudentState:
    """Transparent non-learned baseline using recent summary statistics."""

    if history.empty:
        raise ValueError("History cannot be empty")
    ordered = history.sort_values("interaction_index")
    recent = ordered.tail(6)
    previous = ordered.iloc[-12:-6]
    recent_correct = float(recent["correct"].mean())
    previous_correct = float(previous["correct"].mean()) if len(previous) >= 3 else recent_correct
    trend_delta = recent_correct - previous_correct
    trend = "improving" if trend_delta > 0.10 else "declining" if trend_delta < -0.10 else "stable"
    hint_rate = float(recent["hint_used"].mean())
    attempts = float(recent["attempt_count"].mean())
    time_score = float(np.clip((recent["response_time"].mean() - 35.0) / 100.0, 0.0, 1.0))
    struggle = float(
        np.clip(
            0.50 * (1 - recent_correct)
            + 0.20 * hint_rate
            + 0.20 * time_score
            + 0.10 * max(0.0, attempts - 1),
            0.0,
            1.0,
        )
    )
    mastery = float(np.clip(0.75 * recent_correct + 0.25 * (1 - hint_rate), 0.0, 1.0))
    next_correct = float(np.clip(0.65 * recent_correct + 0.35 * mastery, 0.02, 0.98))
    current = ordered.iloc[-1]
    concept = str(current.get("concept_name", f"Concept {int(current['concept_id'])}"))
    return StudentState(
        current_concept=concept,
        mastery=mastery,
        predicted_next_correct=next_correct,
        struggling_probability=struggle,
        recent_trend=trend,
        needs_intervention=bool(struggle >= 0.5 or next_correct < 0.55),
        confidence=float(np.clip(abs(next_correct - 0.5) * 2.0, 0.0, 1.0)),
    )
