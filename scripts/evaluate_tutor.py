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
from student_model.inference import load_student_model
from tutoring.evaluation import evaluate_llm_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate four LLM tutoring conditions")
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "tutor_eval")
    parser.add_argument("--scenarios", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if settings.DEFAULT_MODEL is None:
        raise SystemExit("Set an LLM API key and DEFAULT_MODEL in .env before running evaluation")
    frame = pd.read_csv(args.interactions)
    bundle = load_student_model(args.checkpoint)
    model = get_model(settings.DEFAULT_MODEL)
    result = asyncio.run(
        evaluate_llm_ablation(
            model,
            bundle,
            frame,
            args.output_dir,
            scenario_limit=args.scenarios,
            concurrency=args.concurrency,
            seed=args.seed,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
