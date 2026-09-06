from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import get_model, settings
from student_model.inference import load_student_model, predict_student_state
from tutoring.llm_agent import run_tutor_turn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one personalised tutoring turn")
    parser.add_argument("question")
    parser.add_argument("--student-id", type=int, default=0)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "student_model" / "student_lstm.pt",
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=ROOT / "data" / "processed" / "synthetic_interactions.csv",
    )
    args = parser.parse_args()

    if settings.DEFAULT_MODEL is None:
        raise SystemExit("Set an LLM API key and DEFAULT_MODEL in .env before running the demo")
    all_interactions = pd.read_csv(args.interactions)
    history = (
        all_interactions[all_interactions["student_id"] == args.student_id]
        .sort_values("interaction_index")
        .head(args.history_length)
    )
    if history.empty:
        raise SystemExit(f"Student {args.student_id} was not found")
    bundle = load_student_model(args.checkpoint)
    state = predict_student_state(bundle, history)
    turn = asyncio.run(
        run_tutor_turn(get_model(settings.DEFAULT_MODEL), args.question, history, state)
    )
    print(
        json.dumps(
            {"student_state": state.model_dump(), **turn.model_dump()}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
