# ruff: noqa: E402
"""Train and save the GPA-free categorical sleep artifact."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleep_clustering import save_artifact, train_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", type=int, default=4)
    args = parser.parse_args()
    save_artifact(train_artifact(args.input, args.clusters), args.output)


if __name__ == "__main__":
    main()
