from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from student_model.data import (
    StudentSequenceDataset,
    aggregate_samples,
    build_sequence_samples,
    collate_sequences,
    split_student_ids,
)
from student_model.models import FixedWidthStudentModel, StudentLSTM, multitask_loss
from student_model.synthetic import SyntheticConfig, generate_interactions


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 18
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    max_sequence_length: int = 24
    min_history: int = 4
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _metric_bundle(
    correctness_true: np.ndarray,
    correctness_probability: np.ndarray,
    mastery_true: np.ndarray,
    mastery_predicted: np.ndarray,
    struggling_true: np.ndarray,
    struggling_probability: np.ndarray,
) -> dict[str, float]:
    correctness_probability = np.clip(correctness_probability, 1e-6, 1 - 1e-6)
    struggling_probability = np.clip(struggling_probability, 1e-6, 1 - 1e-6)
    correctness_predicted = (correctness_probability >= 0.5).astype(int)
    struggling_predicted = (struggling_probability >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(correctness_true, correctness_predicted)),
        "f1": float(f1_score(correctness_true, correctness_predicted, zero_division=0)),
        "log_loss": float(log_loss(correctness_true, correctness_probability, labels=[0, 1])),
        "brier": float(brier_score_loss(correctness_true, correctness_probability)),
        "mastery_mae": float(mean_absolute_error(mastery_true, mastery_predicted)),
        "struggling_f1": float(f1_score(struggling_true, struggling_predicted, zero_division=0)),
    }
    result["auroc"] = (
        float(roc_auc_score(correctness_true, correctness_probability))
        if len(np.unique(correctness_true)) == 2
        else float("nan")
    )
    return result


def _evaluate_sequence(
    model: StudentLSTM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "correct_true": [],
        "correct_probability": [],
        "mastery_true": [],
        "mastery_predicted": [],
        "struggling_true": [],
        "struggling_probability": [],
    }
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["concept_ids"].to(device),
                batch["question_ids"].to(device),
                batch["numeric_features"].to(device),
                batch["lengths"].to(device),
            )
            collected["correct_true"].append(batch["next_correct"].numpy())
            collected["correct_probability"].append(
                torch.sigmoid(outputs["next_correct_logit"]).cpu().numpy()
            )
            collected["mastery_true"].append(batch["mastery"].numpy())
            collected["mastery_predicted"].append(outputs["mastery"].cpu().numpy())
            collected["struggling_true"].append(batch["struggling"].numpy())
            collected["struggling_probability"].append(
                torch.sigmoid(outputs["struggling_logit"]).cpu().numpy()
            )
    arrays = {key: np.concatenate(values) for key, values in collected.items()}
    metrics = _metric_bundle(
        arrays["correct_true"],
        arrays["correct_probability"],
        arrays["mastery_true"],
        arrays["mastery_predicted"],
        arrays["struggling_true"],
        arrays["struggling_probability"],
    )
    return metrics, arrays


def _train_sequence_model(
    train_samples,
    validation_samples,
    test_samples,
    num_concepts: int,
    num_questions: int,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[StudentLSTM, dict[str, Any]]:
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        StudentSequenceDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_sequences,
    )
    validation_loader = DataLoader(
        StudentSequenceDataset(validation_samples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )
    test_loader = DataLoader(
        StudentSequenceDataset(test_samples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )
    model = StudentLSTM(num_concepts=num_concepts, num_questions=num_questions).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    remaining_patience = config.patience
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["concept_ids"].to(device),
                batch["question_ids"].to(device),
                batch["numeric_features"].to(device),
                batch["lengths"].to(device),
            )
            loss = multitask_loss(
                outputs,
                batch["next_correct"].to(device),
                batch["mastery"].to(device),
                batch["struggling"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(batch["next_correct"])
            seen += len(batch["next_correct"])

        validation_metrics, _ = _evaluate_sequence(model, validation_loader, device)
        validation_loss = validation_metrics["log_loss"] + 0.5 * validation_metrics["mastery_mae"]
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": running_loss / max(seen, 1),
                "validation_objective": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = config.patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    model.load_state_dict(best_state)
    validation_metrics, _ = _evaluate_sequence(model, validation_loader, device)
    test_metrics, arrays = _evaluate_sequence(model, test_loader, device)
    return model, {
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "test_predictions": arrays,
    }


def _evaluate_fixed(
    model: FixedWidthStudentModel,
    features: torch.Tensor,
    labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        outputs = model(features.to(device))
    return _metric_bundle(
        labels[0].numpy(),
        torch.sigmoid(outputs["next_correct_logit"]).cpu().numpy(),
        labels[1].numpy(),
        outputs["mastery"].cpu().numpy(),
        labels[2].numpy(),
        torch.sigmoid(outputs["struggling_logit"]).cpu().numpy(),
    )


def _train_fixed_model(
    train_samples,
    validation_samples,
    test_samples,
    num_concepts: int,
    hidden_dims: tuple[int, ...],
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    x_train = torch.tensor(aggregate_samples(train_samples, num_concepts), dtype=torch.float32)
    x_validation = torch.tensor(
        aggregate_samples(validation_samples, num_concepts), dtype=torch.float32
    )
    x_test = torch.tensor(aggregate_samples(test_samples, num_concepts), dtype=torch.float32)

    def labels(samples):
        return (
            torch.tensor([sample.next_correct for sample in samples], dtype=torch.float32),
            torch.tensor([sample.mastery for sample in samples], dtype=torch.float32),
            torch.tensor([sample.struggling for sample in samples], dtype=torch.float32),
        )

    y_train = labels(train_samples)
    y_validation = labels(validation_samples)
    y_test = labels(test_samples)
    dataset = TensorDataset(x_train, *y_train)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = FixedWidthStudentModel(x_train.shape[1], hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_objective = float("inf")
    remaining_patience = config.patience

    for _ in range(config.epochs):
        model.train()
        for features, correct, mastery, struggling in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features.to(device))
            loss = multitask_loss(
                outputs, correct.to(device), mastery.to(device), struggling.to(device)
            )
            loss.backward()
            optimizer.step()
        validation = _evaluate_fixed(model, x_validation, y_validation, device)
        objective = validation["log_loss"] + 0.5 * validation["mastery_mae"]
        if objective < best_objective - 1e-4:
            best_objective = objective
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = config.patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    model.load_state_dict(best_state)
    return {
        "validation": _evaluate_fixed(model, x_validation, y_validation, device),
        "test": _evaluate_fixed(model, x_test, y_test, device),
    }


def run_training_experiment(
    output_dir: str | Path,
    training_config: TrainingConfig | None = None,
    synthetic_config: SyntheticConfig | None = None,
    interactions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train baselines and LSTM, then save a reproducible experiment bundle."""

    training_config = training_config or TrainingConfig()
    synthetic_config = synthetic_config or SyntheticConfig()
    set_seed(training_config.seed)
    device = resolve_device(training_config.device)
    frame = interactions if interactions is not None else generate_interactions(synthetic_config)
    split_ids = split_student_ids(frame, seed=training_config.seed)
    samples = {
        name: build_sequence_samples(
            frame,
            ids,
            max_sequence_length=training_config.max_sequence_length,
            min_history=training_config.min_history,
        )
        for name, ids in split_ids.items()
    }
    num_concepts = int(frame["concept_id"].max()) + 1
    num_questions = int(frame["question_id"].max()) + 1

    logistic_metrics = _train_fixed_model(
        samples["train"],
        samples["validation"],
        samples["test"],
        num_concepts,
        hidden_dims=(),
        config=training_config,
        device=device,
    )
    mlp_metrics = _train_fixed_model(
        samples["train"],
        samples["validation"],
        samples["test"],
        num_concepts,
        hidden_dims=(64, 32),
        config=training_config,
        device=device,
    )
    lstm, lstm_result = _train_sequence_model(
        samples["train"],
        samples["validation"],
        samples["test"],
        num_concepts,
        num_questions,
        training_config,
        device,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "student_lstm.pt"
    torch.save(
        {
            "model_state_dict": lstm.state_dict(),
            "model_config": lstm.config,
            "training_config": asdict(training_config),
            "synthetic_config": asdict(synthetic_config),
            "concept_names": {
                int(key): str(value)
                for key, value in frame.groupby("concept_id")["concept_name"].first().items()
            }
            if "concept_name" in frame
            else {},
        },
        checkpoint_path,
    )
    prediction_frame = pd.DataFrame(
        {
            "correct_true": lstm_result["test_predictions"]["correct_true"],
            "correct_probability": lstm_result["test_predictions"]["correct_probability"],
            "mastery_true": lstm_result["test_predictions"]["mastery_true"],
            "mastery_predicted": lstm_result["test_predictions"]["mastery_predicted"],
            "struggling_true": lstm_result["test_predictions"]["struggling_true"],
            "struggling_probability": lstm_result["test_predictions"]["struggling_probability"],
        }
    )
    prediction_frame.to_csv(destination / "lstm_test_predictions.csv", index=False)

    result = {
        "data": {
            "is_synthetic": bool(frame.get("is_synthetic", pd.Series([False])).all()),
            "rows": len(frame),
            "students": int(frame["student_id"].nunique()),
            "split_students": {key: len(value) for key, value in split_ids.items()},
            "split_samples": {key: len(value) for key, value in samples.items()},
        },
        "config": asdict(training_config),
        "models": {
            "logistic_regression": logistic_metrics,
            "mlp": mlp_metrics,
            "lstm": {key: value for key, value in lstm_result.items() if key != "test_predictions"},
        },
        "checkpoint": str(checkpoint_path),
    }
    (destination / "model_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (destination / "student_split.json").write_text(
        json.dumps(split_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
