from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from sklearn.metrics import accuracy_score, f1_score

from student_model.data import build_sequence_samples
from student_model.inference import StudentModelBundle, predict_student_state
from tutoring.llm_agent import choose_intervention
from tutoring.types import EvaluationCondition, InterventionAction


def _balanced_scenarios(frame: pd.DataFrame, limit: int, seed: int) -> list[Any]:
    samples = build_sequence_samples(frame)
    rng = np.random.default_rng(seed)
    by_action: dict[str, list[Any]] = {}
    for sample in samples:
        by_action.setdefault(sample.metadata["oracle_action"], []).append(sample)
    per_action = max(1, limit // max(len(by_action), 1))
    selected = []
    for action in sorted(by_action):
        candidates = by_action[action]
        indices = rng.choice(len(candidates), size=min(per_action, len(candidates)), replace=False)
        selected.extend(candidates[int(index)] for index in indices)
    if len(selected) < limit:
        selected_ids = {id(item) for item in selected}
        remaining = [sample for sample in samples if id(sample) not in selected_ids]
        if remaining:
            indices = rng.choice(
                len(remaining), size=min(limit - len(selected), len(remaining)), replace=False
            )
            selected.extend(remaining[int(index)] for index in indices)
    rng.shuffle(selected)
    return selected[:limit]


def _cache_key(condition: EvaluationCondition, metadata: dict[str, Any]) -> str:
    value = f"{condition.value}:{metadata['student_id']}:{metadata['interaction_index']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            cache[item["cache_key"]] = item
    return cache


def _bootstrap_accuracy(
    truth: np.ndarray,
    predicted: np.ndarray,
    seed: int,
    draws: int = 1_000,
) -> list[float]:
    if len(truth) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = rng.integers(0, len(truth), size=len(truth))
        values.append(float(np.mean(truth[indices] == predicted[indices])))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


async def evaluate_llm_ablation(
    model: BaseChatModel,
    bundle: StudentModelBundle,
    interactions: pd.DataFrame,
    output_dir: str | Path,
    scenario_limit: int = 48,
    concurrency: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the four-condition intervention experiment against hidden oracle labels."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cache_path = destination / "llm_decisions.jsonl"
    cache = _load_cache(cache_path)
    scenarios = _balanced_scenarios(interactions, scenario_limit, seed)
    semaphore = asyncio.Semaphore(concurrency)
    records: list[dict[str, Any]] = []

    async def run_one(sample, condition: EvaluationCondition) -> dict[str, Any]:
        key = _cache_key(condition, sample.metadata)
        if key in cache:
            return cache[key]
        history = pd.DataFrame(sample.metadata["history"])
        learned_state = (
            predict_student_state(bundle, history)
            if condition == EvaluationCondition.LEARNED_STATE
            else None
        )
        async with semaphore:
            try:
                decision = await choose_intervention(
                    model,
                    condition,
                    sample.metadata["question_text"],
                    history,
                    learned_state,
                )
                return {
                    "cache_key": key,
                    "condition": condition.value,
                    "student_id": sample.metadata["student_id"],
                    "interaction_index": sample.metadata["interaction_index"],
                    "oracle_action": sample.metadata["oracle_action"],
                    "predicted_action": decision.action.value,
                    "reason": decision.reason,
                    "error": None,
                }
            # LLM providers expose different exception classes. A failed scenario is
            # recorded and excluded rather than aborting the entire paid experiment.
            except Exception as exc:  # noqa: BLE001
                return {
                    "cache_key": key,
                    "condition": condition.value,
                    "student_id": sample.metadata["student_id"],
                    "interaction_index": sample.metadata["interaction_index"],
                    "oracle_action": sample.metadata["oracle_action"],
                    "predicted_action": None,
                    "reason": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    tasks = [
        run_one(sample, condition) for sample in scenarios for condition in EvaluationCondition
    ]
    records = await asyncio.gather(*tasks)
    cache_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    labels = [action.value for action in InterventionAction]
    metrics: dict[str, Any] = {}
    for condition in EvaluationCondition:
        valid = [
            record
            for record in records
            if record["condition"] == condition.value and record["predicted_action"] is not None
        ]
        truth = np.array([record["oracle_action"] for record in valid])
        predicted = np.array([record["predicted_action"] for record in valid])
        metrics[condition.value] = {
            "requested": len(scenarios),
            "completed": len(valid),
            "failures": len(scenarios) - len(valid),
            "intervention_accuracy": float(accuracy_score(truth, predicted))
            if len(valid)
            else None,
            "macro_f1": float(
                f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
            )
            if len(valid)
            else None,
            "accuracy_95_percent_bootstrap_ci": _bootstrap_accuracy(truth, predicted, seed=seed)
            if len(valid)
            else None,
        }

    result = {
        "design": {
            "scenario_count": len(scenarios),
            "conditions": [condition.value for condition in EvaluationCondition],
            "oracle_hidden_from_llm": True,
            "synthetic_benchmark": bool(interactions.get("is_synthetic", pd.Series([False])).all()),
            "seed": seed,
        },
        "metrics": metrics,
    }
    (destination / "llm_ablation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
