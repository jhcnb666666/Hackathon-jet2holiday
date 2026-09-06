from student_model.synthetic import SyntheticConfig, generate_interactions
from student_model.types import StudentState
from tutoring.llm_agent import decision_context
from tutoring.policy import rule_based_student_state
from tutoring.types import EvaluationCondition


def test_rule_profile_is_bounded():
    history = generate_interactions(
        SyntheticConfig(students=1, interactions_per_student=12, seed=9)
    )
    state = rule_based_student_state(history)
    assert 0 <= state.mastery <= 1
    assert 0 <= state.struggling_probability <= 1


def test_ablation_context_respects_information_boundary():
    history = generate_interactions(
        SyntheticConfig(students=1, interactions_per_student=10, seed=11)
    )
    state = StudentState(
        current_concept="Bayes Theorem",
        mastery=0.3,
        predicted_next_correct=0.25,
        struggling_probability=0.8,
        recent_trend="declining",
        needs_intervention=True,
        confidence=0.5,
    )
    llm_only = decision_context(EvaluationCondition.LLM_ONLY, "Why multiply here?", history, state)
    raw = decision_context(EvaluationCondition.RAW_HISTORY, "Why multiply here?", history, state)
    learned = decision_context(
        EvaluationCondition.LEARNED_STATE, "Why multiply here?", history, state
    )
    assert "recent_interactions" not in llm_only
    assert "student_state" not in llm_only
    assert "recent_interactions" in raw
    assert "true_mastery" not in raw
    assert '"mastery": 0.3' in learned
    assert "oracle_action" not in learned
