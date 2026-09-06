from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

REQUIRED_INTERACTION_COLUMNS = {
    "student_id",
    "interaction_index",
    "question_id",
    "concept_id",
    "difficulty",
    "correct",
    "response_time",
    "hint_used",
    "attempt_count",
}


@dataclass(frozen=True)
class SequenceSample:
    concept_ids: np.ndarray
    question_ids: np.ndarray
    numeric_features: np.ndarray
    next_correct: float
    mastery: float
    struggling: float
    metadata: dict[str, Any]


def validate_interactions(frame: pd.DataFrame, require_supervision: bool = True) -> None:
    required = set(REQUIRED_INTERACTION_COLUMNS)
    if require_supervision:
        required |= {"true_mastery", "true_struggling"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Interaction data is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Interaction data is empty")
    duplicated = frame.duplicated(["student_id", "interaction_index"])
    if duplicated.any():
        raise ValueError("Each student interaction_index must be unique")
    if not frame["correct"].isin([0, 1]).all():
        raise ValueError("correct must contain only 0/1")
    if not frame["hint_used"].isin([0, 1]).all():
        raise ValueError("hint_used must contain only 0/1")


def split_student_ids(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int | str]]:
    """Split by student, preventing a learner from leaking across partitions."""

    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must be positive and leave a non-empty test partition")
    student_ids = frame["student_id"].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(student_ids)
    train_end = int(len(student_ids) * train_fraction)
    validation_end = train_end + int(len(student_ids) * validation_fraction)
    return {
        "train": student_ids[:train_end].tolist(),
        "validation": student_ids[train_end:validation_end].tolist(),
        "test": student_ids[validation_end:].tolist(),
    }


def _numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    response_time = np.log1p(frame["response_time"].astype(float).to_numpy()) / np.log1p(180.0)
    attempt_count = np.clip(frame["attempt_count"].astype(float).to_numpy() / 4.0, 0.0, 1.0)
    return np.column_stack(
        [
            frame["difficulty"].astype(float).to_numpy(),
            frame["correct"].astype(float).to_numpy(),
            response_time,
            frame["hint_used"].astype(float).to_numpy(),
            attempt_count,
        ]
    ).astype(np.float32)


def build_sequence_samples(
    frame: pd.DataFrame,
    student_ids: list[int | str] | None = None,
    max_sequence_length: int = 24,
    min_history: int = 4,
) -> list[SequenceSample]:
    validate_interactions(frame, require_supervision=True)
    selected = frame if student_ids is None else frame[frame["student_id"].isin(student_ids)]
    samples: list[SequenceSample] = []

    for student_id, group in selected.groupby("student_id", sort=False):
        ordered = group.sort_values("interaction_index").reset_index(drop=True)
        for target_position in range(min_history, len(ordered)):
            history = ordered.iloc[max(0, target_position - max_sequence_length) : target_position]
            current = history.iloc[-1]
            target = ordered.iloc[target_position]
            samples.append(
                SequenceSample(
                    concept_ids=history["concept_id"].to_numpy(dtype=np.int64),
                    question_ids=history["question_id"].to_numpy(dtype=np.int64),
                    numeric_features=_numeric_matrix(history),
                    next_correct=float(target["correct"]),
                    mastery=float(current["true_mastery"]),
                    struggling=float(current["true_struggling"]),
                    metadata={
                        "student_id": student_id,
                        "interaction_index": int(current["interaction_index"]),
                        "concept_id": int(current["concept_id"]),
                        "concept_name": str(current.get("concept_name", current["concept_id"])),
                        "question_text": str(
                            current.get("question_text", "Please help me with this topic.")
                        ),
                        "oracle_action": str(current.get("oracle_action", "GIVE_EXAMPLE")),
                        "prerequisite_mastery": float(current.get("prerequisite_mastery", 0.5)),
                        "recent_declining": bool(current.get("recent_declining", False)),
                        "history": history[list(REQUIRED_INTERACTION_COLUMNS)].to_dict("records"),
                    },
                )
            )
    return samples


class StudentSequenceDataset(Dataset):
    def __init__(self, samples: list[SequenceSample]) -> None:
        if not samples:
            raise ValueError("At least one sequence sample is required")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "concept_ids": torch.tensor(sample.concept_ids.copy(), dtype=torch.long),
            "question_ids": torch.tensor(sample.question_ids.copy(), dtype=torch.long),
            "numeric_features": torch.tensor(sample.numeric_features.copy(), dtype=torch.float32),
            "next_correct": torch.tensor(sample.next_correct, dtype=torch.float32),
            "mastery": torch.tensor(sample.mastery, dtype=torch.float32),
            "struggling": torch.tensor(sample.struggling, dtype=torch.float32),
            "metadata": sample.metadata,
        }


def collate_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([len(item["concept_ids"]) for item in batch], dtype=torch.long)
    return {
        "concept_ids": pad_sequence([item["concept_ids"] for item in batch], batch_first=True),
        "question_ids": pad_sequence([item["question_ids"] for item in batch], batch_first=True),
        "numeric_features": pad_sequence(
            [item["numeric_features"] for item in batch], batch_first=True
        ),
        "lengths": lengths,
        "next_correct": torch.stack([item["next_correct"] for item in batch]),
        "mastery": torch.stack([item["mastery"] for item in batch]),
        "struggling": torch.stack([item["struggling"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }


def aggregate_sample(sample: SequenceSample, num_concepts: int) -> np.ndarray:
    """Build a leakage-safe fixed-width representation for non-sequential baselines."""

    numeric = sample.numeric_features
    correct = numeric[:, 1]
    recent = numeric[-6:]
    previous = numeric[-12:-6] if len(numeric) >= 12 else recent
    concept_attempts = np.zeros(num_concepts, dtype=np.float32)
    concept_correct = np.zeros(num_concepts, dtype=np.float32)
    for concept_id, outcome in zip(sample.concept_ids, correct, strict=True):
        if 0 <= concept_id < num_concepts:
            concept_attempts[concept_id] += 1.0
            concept_correct[concept_id] += float(outcome)
    concept_rates = np.divide(
        concept_correct,
        np.maximum(concept_attempts, 1.0),
        dtype=np.float32,
    )
    concept_attempts /= max(float(len(sample.concept_ids)), 1.0)
    summary = np.array(
        [
            len(numeric) / 24.0,
            float(numeric[:, 0].mean()),
            float(correct.mean()),
            float(recent[:, 1].mean()),
            float(previous[:, 1].mean()),
            float(recent[:, 1].mean() - previous[:, 1].mean()),
            float(numeric[:, 2].mean()),
            float(recent[:, 2].mean()),
            float(numeric[:, 3].mean()),
            float(recent[:, 3].mean()),
            float(numeric[:, 4].mean()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([summary, concept_attempts, concept_rates])


def aggregate_samples(samples: list[SequenceSample], num_concepts: int) -> np.ndarray:
    return np.stack([aggregate_sample(sample, num_concepts) for sample in samples])
