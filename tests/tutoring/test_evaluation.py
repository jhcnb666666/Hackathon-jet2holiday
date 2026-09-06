import asyncio
from pathlib import Path

import torch

from student_model.inference import StudentModelBundle
from student_model.models import StudentLSTM
from student_model.synthetic import SyntheticConfig, generate_interactions
from tutoring.evaluation import evaluate_llm_ablation
from tutoring.types import InterventionAction, InterventionDecision


def test_ablation_runs_all_conditions_without_exposing_oracle(monkeypatch, tmp_path: Path):
    async def fake_choose(model, condition, question, history, learned_state=None):
        action = (
            InterventionAction.GIVE_HINT
            if float(history.tail(4)["correct"].mean()) < 0.5
            else InterventionAction.GIVE_EXAMPLE
        )
        return InterventionDecision(action=action, reason="Deterministic test policy")

    monkeypatch.setattr("tutoring.evaluation.choose_intervention", fake_choose)
    model = StudentLSTM(num_concepts=6, num_questions=72, hidden_dim=16)
    bundle = StudentModelBundle(model, {}, 24, torch.device("cpu"))
    frame = generate_interactions(SyntheticConfig(students=8, interactions_per_student=10, seed=21))
    result = asyncio.run(
        evaluate_llm_ablation(object(), bundle, frame, tmp_path, scenario_limit=8, concurrency=2)
    )
    assert result["design"]["oracle_hidden_from_llm"] is True
    assert set(result["metrics"]) == {
        "llm_only",
        "raw_history",
        "rule_based",
        "learned_state",
    }
    assert all(item["completed"] == 8 for item in result["metrics"].values())
    assert (tmp_path / "llm_ablation_metrics.json").exists()
