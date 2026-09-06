from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from student_model.health_data import evaluate_health_baselines, prepare_health_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare independent health survey matrices")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "real")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    manifest = prepare_health_datasets(args.input_dir, args.output_dir)
    result = {"manifest": manifest}
    if not args.skip_baselines:
        result["baseline_metrics"] = evaluate_health_baselines(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
