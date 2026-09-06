from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPECTED_FILES = (
    "demographics_cleaned.csv",
    "eating_habits_cleaned.csv",
    "physical_activity_cleaned.csv",
    "sleep_cleaned.csv",
    "work_cleaned.csv",
)

NON_FEATURE_COLUMNS = {"record_id", "current_gpa", "gpa_quality_flag"}


def _ordinal_value(value: object) -> float:
    text = str(value).strip().lower()
    explicit = {
        "0 times (not at all)": 0.0,
        "0 glasses (not at all)": 0.0,
        "0 days (not at all)": 0.0,
        "1 to 3 times": 2.0,
        "1 to 3 glasses": 2.0,
        "4 to 6 times": 5.0,
        "4 to 6 glasses": 5.0,
        "7 to 10 times": 8.5,
        "7 to 10 glasses": 8.5,
        "7 to 11 times": 9.0,
        "11 times or more": 11.0,
        "11 glasses or more": 11.0,
        "4 hours or less": 4.0,
        "10 or more hours": 10.0,
        "before 8pm": 19.0,
        "between 8pm and 10pm": 21.0,
        "between 10pm and 12am": 23.0,
        "between 12am and 2am": 1.0,
        "after 2am": 3.0,
        "before 6am": 5.0,
        "between 6am and 8am": 7.0,
        "between 8am and 10am": 9.0,
        "between 10am and 12pm": 11.0,
        "after 12pm": 13.0,
    }
    if text in explicit:
        return explicit[text]
    day_match = re.fullmatch(r"(\d+) day(?:s)?(?: \(everyday\))?", text)
    if day_match:
        return float(day_match.group(1))
    hour_match = re.fullmatch(r"(\d+) hours?", text)
    if hour_match:
        return float(hour_match.group(1))
    return float("nan")


def _encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.drop(columns=[column for column in NON_FEATURE_COLUMNS if column in frame])
    numeric = source.select_dtypes(include=["number", "bool"]).astype(float)
    categorical = source.drop(columns=numeric.columns, errors="ignore").astype("string")
    derived: dict[str, pd.Series] = {}
    for column in categorical.columns:
        ordinal = categorical[column].map(_ordinal_value)
        if ordinal.notna().all():
            derived[f"{column}__ordinal"] = ordinal.astype(float)
            if column in {"bedtime", "wake_time"}:
                radians = ordinal.astype(float) * (2 * np.pi / 24.0)
                derived[f"{column}__sin"] = np.sin(radians)
                derived[f"{column}__cos"] = np.cos(radians)
    one_hot = pd.get_dummies(categorical, prefix=categorical.columns, dtype=float)
    pieces = [numeric.reset_index(drop=True), pd.DataFrame(derived), one_hot.reset_index(drop=True)]
    encoded = pd.concat(pieces, axis=1)
    encoded.columns = [str(column).replace(" ", "_").lower() for column in encoded.columns]
    return encoded


def prepare_health_datasets(
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create independent model matrices without joining file-local record IDs."""

    source_dir = Path(input_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"datasets": {}, "warnings": []}

    for filename in EXPECTED_FILES:
        path = source_dir / filename
        if not path.exists():
            manifest["warnings"].append(f"Missing expected file: {filename}")
            continue
        frame = pd.read_csv(path)
        required = {"record_id", "current_gpa", "gpa_quality_flag"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
        if frame["record_id"].duplicated().any():
            raise ValueError(f"{filename} contains duplicate file-local record_id values")
        encoded = _encode_features(frame)
        processed = pd.concat(
            [
                frame[["record_id"]].reset_index(drop=True),
                pd.DataFrame(
                    {
                        "target_gpa": frame["current_gpa"].astype(float),
                        "include_primary": (
                            frame["gpa_quality_flag"].eq("valid")
                            & frame["current_gpa"].between(0.0, 4.0)
                        ),
                    }
                ),
                encoded,
            ],
            axis=1,
        )
        output_path = destination / filename.replace("_cleaned.csv", "_features.csv")
        processed.to_csv(output_path, index=False)
        manifest["datasets"][filename] = {
            "source_rows": len(frame),
            "primary_rows": int(processed["include_primary"].sum()),
            "excluded_gpa_rows": int((~processed["include_primary"]).sum()),
            "feature_count": int(encoded.shape[1]),
            "output": str(output_path),
        }

    report_path = source_dir.parent / "docs" / "data_quality_report.csv"
    if report_path.exists():
        quality = pd.read_csv(report_path)
        documented = set(quality["file"].astype(str))
        present = {path.name for path in source_dir.glob("*.csv")}
        for missing_name in sorted(documented - present):
            manifest["warnings"].append(
                f"Quality report references a file not present in data/real: {missing_name}"
            )
    (destination / "health_data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def evaluate_health_baselines(
    processed_dir: str | Path,
    seed: int = 42,
    folds: int = 5,
) -> dict[str, Any]:
    """Evaluate static GPA models separately for each survey extract."""

    results: dict[str, Any] = {
        "warning": (
            "These are cross-sectional association baselines, not causal models and not "
            "the sequential student-state model."
        ),
        "datasets": {},
    }
    for path in sorted(Path(processed_dir).glob("*_features.csv")):
        frame = pd.read_csv(path)
        primary = frame[frame["include_primary"].astype(bool)].reset_index(drop=True)
        x = primary.drop(columns=["record_id", "target_gpa", "include_primary"]).to_numpy(float)
        y = primary["target_gpa"].to_numpy(float)
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        models = {
            "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=5.0))]),
            "mlp": TransformedTargetRegressor(
                regressor=Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=(32, 16),
                                alpha=0.02,
                                learning_rate_init=5e-4,
                                max_iter=600,
                                early_stopping=True,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
                transformer=StandardScaler(),
            ),
        }
        dataset_result: dict[str, Any] = {}
        for model_name, model in models.items():
            fold_metrics = []
            for train_index, test_index in splitter.split(x):
                model.fit(x[train_index], y[train_index])
                predicted = model.predict(x[test_index])
                fold_metrics.append(
                    {
                        "mae": float(mean_absolute_error(y[test_index], predicted)),
                        "r2": float(r2_score(y[test_index], predicted)),
                    }
                )
            dataset_result[model_name] = {
                "mae_mean": float(np.mean([item["mae"] for item in fold_metrics])),
                "mae_std": float(np.std([item["mae"] for item in fold_metrics])),
                "r2_mean": float(np.mean([item["r2"] for item in fold_metrics])),
                "r2_std": float(np.std([item["r2"] for item in fold_metrics])),
            }
        results["datasets"][path.name] = dataset_result
    output_path = Path(processed_dir) / "health_baseline_metrics.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
