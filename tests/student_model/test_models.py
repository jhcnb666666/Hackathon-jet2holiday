import torch

from student_model.models import FixedWidthStudentModel, StudentLSTM, multitask_loss


def test_lstm_outputs_valid_state_and_supports_backpropagation():
    model = StudentLSTM(num_concepts=4, num_questions=20, hidden_dim=16)
    outputs = model(
        concept_ids=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        question_ids=torch.tensor([[0, 3, 9], [2, 7, 0]]),
        numeric_features=torch.rand(2, 3, 5),
        lengths=torch.tensor([3, 2]),
    )
    assert outputs["next_correct_logit"].shape == (2,)
    assert torch.all((outputs["mastery"] >= 0) & (outputs["mastery"] <= 1))
    loss = multitask_loss(
        outputs,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.6, 0.4]),
        torch.tensor([0.0, 1.0]),
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_fixed_width_baselines_have_same_prediction_contract():
    for hidden in [(), (8,)]:
        outputs = FixedWidthStudentModel(10, hidden)(torch.rand(3, 10))
        assert set(outputs) == {"next_correct_logit", "mastery", "struggling_logit"}
        assert outputs["mastery"].shape == (3,)
