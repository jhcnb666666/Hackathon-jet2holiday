from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from student_model.synthetic import SyntheticConfig, generate_interactions
from student_model.training import TrainingConfig, run_training_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train student-state baselines and LSTM")
    parser.add_argument("--students", type=int, default=800)
    parser.add_argument("--interactions", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "student_model")
    parser.add_argument(
        "--data-output",
        type=Path,
        default=ROOT / "data" / "processed" / "synthetic_interactions.csv",
    )
    parser.add_argument("--quick", action="store_true", help="Run a small smoke experiment")
    args = parser.parse_args()

    students = 60 if args.quick else args.students
    interactions = 16 if args.quick else args.interactions
    epochs = 2 if args.quick else args.epochs
    synthetic_config = SyntheticConfig(
        students=students,
        interactions_per_student=interactions,
        seed=args.seed,
    )
    training_config = TrainingConfig(
        seed=args.seed,
        epochs=epochs,
        patience=2 if args.quick else 4,
        device=args.device,
    )
    frame = generate_interactions(synthetic_config)
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.data_output, index=False)
    result = run_training_experiment(
        args.output_dir,
        training_config=training_config,
        synthetic_config=synthetic_config,
        interactions=frame,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
