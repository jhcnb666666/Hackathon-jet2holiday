from __future__ import annotations

import json
from typing import Any

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from student_model.types import StudentState
from tutoring.policy import rule_based_student_state
from tutoring.types import EvaluationCondition, InterventionDecision, TutorTurn

DECISION_SYSTEM_PROMPT = """You are a tutoring policy agent. Choose exactly one pedagogical
intervention before composing an answer. Optimize for the student's next useful learning step,
not for answer length. Do not make medical or psychological diagnoses. Use only the evidence
provided. Return the requested structured output."""

RESPONSE_SYSTEM_PROMPT = """You are a patient, concise tutor. Follow the selected intervention.
Do not reveal hidden student-state scores or claim a diagnosis. Ask at most one follow-up question
and keep the response focused on the student's current concept."""


def _safe_history(history: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    allowed = [
        "interaction_index",
        "question_id",
        "concept_id",
        "concept_name",
        "difficulty",
        "correct",
        "response_time",
        "hint_used",
        "attempt_count",
    ]
    columns = [column for column in allowed if column in history]
    return history.sort_values("interaction_index").tail(limit)[columns].to_dict("records")


def decision_context(
    condition: EvaluationCondition,
    question: str,
    history: pd.DataFrame,
    learned_state: StudentState | None = None,
) -> str:
    payload: dict[str, Any] = {"student_question": question}
    if condition == EvaluationCondition.RAW_HISTORY:
        payload["recent_interactions"] = _safe_history(history)
    elif condition == EvaluationCondition.RULE_BASED:
        payload["student_state"] = rule_based_student_state(history).model_dump()
        payload["state_source"] = "transparent rolling rules"
    elif condition == EvaluationCondition.LEARNED_STATE:
        if learned_state is None:
            raise ValueError("learned_state is required for the learned_state condition")
        payload["student_state"] = learned_state.model_dump()
        payload["state_source"] = "trained sequence model"
    elif condition != EvaluationCondition.LLM_ONLY:
        raise ValueError(f"Unsupported evaluation condition: {condition}")
    payload["available_actions"] = [
        "EXPLAIN_CONCEPT",
        "REVIEW_PREREQUISITE",
        "GIVE_HINT",
        "GIVE_EXAMPLE",
        "GENERATE_EASIER_QUESTION",
        "GENERATE_CHALLENGE_QUESTION",
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def choose_intervention(
    model: BaseChatModel,
    condition: EvaluationCondition,
    question: str,
    history: pd.DataFrame,
    learned_state: StudentState | None = None,
) -> InterventionDecision:
    structured_model = model.with_structured_output(InterventionDecision)
    result = await structured_model.ainvoke(
        [
            SystemMessage(content=DECISION_SYSTEM_PROMPT),
            HumanMessage(content=decision_context(condition, question, history, learned_state)),
        ]
    )
    return InterventionDecision.model_validate(result)


async def generate_tutor_response(
    model: BaseChatModel,
    question: str,
    decision: InterventionDecision,
    state: StudentState | None = None,
) -> str:
    payload: dict[str, Any] = {
        "student_question": question,
        "selected_intervention": decision.action.value,
        "decision_reason": decision.reason,
    }
    if state is not None:
        payload["student_state"] = state.model_dump()
    result = await model.ainvoke(
        [
            SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
        ]
    )
    content = result.content
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


async def run_tutor_turn(
    model: BaseChatModel,
    question: str,
    history: pd.DataFrame,
    learned_state: StudentState,
) -> TutorTurn:
    decision = await choose_intervention(
        model,
        EvaluationCondition.LEARNED_STATE,
        question,
        history,
        learned_state,
    )
    response = await generate_tutor_response(model, question, decision, learned_state)
    return TutorTurn(decision=decision, response=response)
