from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

CONCEPT_NAMES = {
    0: "Fractions",
    1: "Conditional Probability",
    2: "Bayes Theorem",
    3: "Expected Value",
    4: "Hypothesis Testing",
    5: "Confidence Intervals",
}


@dataclass(frozen=True)
class SyntheticConfig:
    students: int = 800
    interactions_per_student: int = 36
    concepts: int = 6
    questions_per_concept: int = 12
    seed: int = 42


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _oracle_action(
    mastery: float,
    next_correct_probability: float,
    struggling: bool,
    prerequisite_mastery: float,
    declining: bool,
) -> str:
    if mastery >= 0.80 and next_correct_probability >= 0.75:
        return "GENERATE_CHALLENGE_QUESTION"
    if prerequisite_mastery < 0.45 and mastery < 0.55:
        return "REVIEW_PREREQUISITE"
    if next_correct_probability < 0.35:
        return "GIVE_HINT"
    if declining and struggling:
        return "EXPLAIN_CONCEPT"
    if mastery < 0.60:
        return "GENERATE_EASIER_QUESTION"
    return "GIVE_EXAMPLE"


def generate_interactions(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate a transparent knowledge-tracing benchmark.

    The simulator has a latent mastery value per student and concept. Responses,
    hint usage and time are sampled from that state, and practice updates mastery.
    Ground-truth state columns are retained solely for supervised training and
    evaluation; they must never be sent to a baseline that is not entitled to them.
    """

    config = config or SyntheticConfig()
    if config.concepts < 2:
        raise ValueError("At least two concepts are required for prerequisite evaluation")
    if config.interactions_per_student < 8:
        raise ValueError("At least eight interactions per student are required")

    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []

    for student_id in range(config.students):
        ability = float(rng.normal(0.0, 0.55))
        mastery = np.clip(rng.beta(2.2, 3.8, size=config.concepts) + ability * 0.06, 0.05, 0.92)
        recent_correct: list[int] = []

        for interaction_index in range(config.interactions_per_student):
            weak_weights = np.clip(1.15 - mastery, 0.12, None)
            concept_probs = weak_weights + np.full(config.concepts, 0.22)
            concept_probs /= concept_probs.sum()
            concept_id = int(rng.choice(config.concepts, p=concept_probs))
            question_offset = int(rng.integers(config.questions_per_concept))
            question_id = concept_id * config.questions_per_concept + question_offset
            difficulty = float(rng.choice([0.25, 0.50, 0.75], p=[0.30, 0.45, 0.25]))
            prerequisite_mastery = float(mastery[max(0, concept_id - 1)])
            prereq_penalty = 0.45 * max(0.0, 0.50 - prerequisite_mastery) if concept_id else 0.0
            correct_probability = _sigmoid(
                4.1 * (float(mastery[concept_id]) - difficulty) + 0.55 * ability - prereq_penalty
            )
            correct = int(rng.random() < correct_probability)
            hint_probability = _sigmoid(3.3 * (0.52 - correct_probability))
            hint_used = int(rng.random() < hint_probability)
            response_time = float(
                np.clip(
                    rng.normal(34 + 72 * (1 - correct_probability) + 14 * hint_used, 11),
                    8,
                    180,
                )
            )
            attempt_count = (
                1 + int(not correct) + int(rng.random() < max(0.0, 0.45 - correct_probability))
            )

            before = float(mastery[concept_id])
            learning_gain = (0.075 if correct else 0.035) * (1.0 - before)
            learning_gain *= 0.82 if hint_used else 1.0
            mastery[concept_id] = np.clip(before + learning_gain, 0.02, 0.99)
            mastery *= 0.9995

            recent_correct.append(correct)
            recent_window = recent_correct[-6:]
            recent_rate = float(np.mean(recent_window))
            previous_rate = (
                float(np.mean(recent_correct[-12:-6])) if len(recent_correct) >= 12 else recent_rate
            )
            declining = recent_rate + 0.08 < previous_rate
            overall_mastery = float(np.mean(mastery))
            struggling = bool(
                correct_probability < 0.48 or (len(recent_window) >= 4 and recent_rate < 0.45)
            )
            action = _oracle_action(
                mastery=float(mastery[concept_id]),
                next_correct_probability=correct_probability,
                struggling=struggling,
                prerequisite_mastery=prerequisite_mastery,
                declining=declining,
            )

            concept_name = CONCEPT_NAMES.get(concept_id, f"Concept {concept_id}")
            rows.append(
                {
                    "student_id": student_id,
                    "interaction_index": interaction_index,
                    "question_id": question_id,
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "difficulty": difficulty,
                    "correct": correct,
                    "response_time": round(response_time, 3),
                    "hint_used": hint_used,
                    "attempt_count": attempt_count,
                    "true_mastery": round(float(mastery[concept_id]), 6),
                    "true_overall_mastery": round(overall_mastery, 6),
                    "true_response_probability": round(correct_probability, 6),
                    "true_struggling": int(struggling),
                    "prerequisite_mastery": round(prerequisite_mastery, 6),
                    "recent_declining": int(declining),
                    "oracle_action": action,
                    "question_text": f"Help me understand the next {concept_name} problem.",
                    "is_synthetic": True,
                }
            )

    return pd.DataFrame(rows)
