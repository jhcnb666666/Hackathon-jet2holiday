from pathlib import Path

import torch

from student_model.inference import load_student_model, predict_student_state
from student_model.models import StudentLSTM
from student_model.synthetic import SyntheticConfig, generate_interactions


def test_checkpoint_round_trip_produces_bounded_state(tmp_path: Path):
    model = StudentLSTM(num_concepts=6, num_questions=72, hidden_dim=16)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
            "training_config": {"max_sequence_length": 12},
            "concept_names": {0: "Fractions"},
        },
        checkpoint,
    )
    history = generate_interactions(
        SyntheticConfig(students=1, interactions_per_student=10, seed=3)
    )
    state = predict_student_state(load_student_model(checkpoint), history)
    assert 0 <= state.mastery <= 1
    assert 0 <= state.predicted_next_correct <= 1
    assert state.recent_trend in {"improving", "stable", "declining"}
