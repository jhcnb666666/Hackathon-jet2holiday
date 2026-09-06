from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from student_model.data import _numeric_matrix, validate_interactions
from student_model.models import StudentLSTM
from student_model.types import StudentState


@dataclass
class StudentModelBundle:
    model: StudentLSTM
    concept_names: dict[int, str]
    max_sequence_length: int
    device: torch.device


def load_student_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> StudentModelBundle:
    resolved_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model = StudentLSTM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device).eval()
    names = {int(key): str(value) for key, value in checkpoint.get("concept_names", {}).items()}
    max_length = int(checkpoint.get("training_config", {}).get("max_sequence_length", 24))
    return StudentModelBundle(model, names, max_length, resolved_device)


def _trend(history: pd.DataFrame) -> str:
    recent = history["correct"].astype(float).tail(6)
    previous = history["correct"].astype(float).iloc[-12:-6]
    if len(previous) < 3:
        return "stable"
    delta = float(recent.mean() - previous.mean())
    if delta > 0.10:
        return "improving"
    if delta < -0.10:
        return "declining"
    return "stable"


def predict_student_state(
    bundle: StudentModelBundle,
    history: pd.DataFrame,
) -> StudentState:
    validate_interactions(history, require_supervision=False)
    ordered = history.sort_values("interaction_index").tail(bundle.max_sequence_length)
    if ordered.empty:
        raise ValueError("At least one historical interaction is required")
    concept_ids = torch.tensor(
        ordered["concept_id"].to_numpy(dtype=np.int64)[None, :],
        dtype=torch.long,
        device=bundle.device,
    )
    question_ids = torch.tensor(
        ordered["question_id"].to_numpy(dtype=np.int64)[None, :],
        dtype=torch.long,
        device=bundle.device,
    )
    if int(concept_ids.max()) >= bundle.model.config["num_concepts"]:
        raise ValueError("History contains a concept_id unseen during training")
    if int(question_ids.max()) >= bundle.model.config["num_questions"]:
        raise ValueError("History contains a question_id unseen during training")
    numeric = torch.tensor(
        _numeric_matrix(ordered)[None, :, :], dtype=torch.float32, device=bundle.device
    )
    lengths = torch.tensor([len(ordered)], dtype=torch.long, device=bundle.device)
    with torch.no_grad():
        outputs = bundle.model(concept_ids, question_ids, numeric, lengths)
    next_correct = float(torch.sigmoid(outputs["next_correct_logit"])[0].item())
    mastery = float(outputs["mastery"][0].item())
    struggling = float(torch.sigmoid(outputs["struggling_logit"])[0].item())
    current_concept_id = int(ordered.iloc[-1]["concept_id"])
    current_concept = bundle.concept_names.get(current_concept_id, f"Concept {current_concept_id}")
    confidence = float(np.clip(abs(next_correct - 0.5) * 2.0, 0.0, 1.0))
    return StudentState(
        current_concept=current_concept,
        mastery=mastery,
        predicted_next_correct=next_correct,
        struggling_probability=struggling,
        recent_trend=_trend(ordered),
        needs_intervention=bool(struggling >= 0.5 or next_correct < 0.55),
        confidence=confidence,
    )
