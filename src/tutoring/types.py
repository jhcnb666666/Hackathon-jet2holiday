from enum import StrEnum

from pydantic import BaseModel, Field


class InterventionAction(StrEnum):
    EXPLAIN_CONCEPT = "EXPLAIN_CONCEPT"
    REVIEW_PREREQUISITE = "REVIEW_PREREQUISITE"
    GIVE_HINT = "GIVE_HINT"
    GIVE_EXAMPLE = "GIVE_EXAMPLE"
    GENERATE_EASIER_QUESTION = "GENERATE_EASIER_QUESTION"
    GENERATE_CHALLENGE_QUESTION = "GENERATE_CHALLENGE_QUESTION"


class EvaluationCondition(StrEnum):
    LLM_ONLY = "llm_only"
    RAW_HISTORY = "raw_history"
    RULE_BASED = "rule_based"
    LEARNED_STATE = "learned_state"


class InterventionDecision(BaseModel):
    action: InterventionAction
    reason: str = Field(min_length=3, max_length=500)


class TutorTurn(BaseModel):
    decision: InterventionDecision
    response: str
