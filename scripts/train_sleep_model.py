#!/usr/bin/env python3
"""Train the categorical sleep survey model and write a portable JSON artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleep_clustering import save_artifact, train_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "sleep_cleaned.csv",
        help="CSV containing sleep_duration, bedtime, and wake_time",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "sleep_model.json",
        help="Destination JSON model artifact",
    )
    parser.add_argument("--clusters", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = train_artifact(args.input, clusters=args.clusters)
    save_artifact(artifact, args.output)
    print(
        f"Trained {artifact['model_type']} on {artifact['training_rows']} rows; "
        f"saved {len(artifact['clusters'])} clusters to {args.output}"
    )


if __name__ == "__main__":
    main()
