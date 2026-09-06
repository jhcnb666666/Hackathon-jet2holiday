from pathlib import Path

import pandas as pd

from student_model.health_data import prepare_health_datasets


def test_health_preprocessing_keeps_sources_independent(tmp_path: Path):
    input_dir = Path(__file__).resolve().parents[2] / "data" / "real"
    manifest = prepare_health_datasets(input_dir, tmp_path)
    assert len(manifest["datasets"]) == 5
    assert any("substance_use_cleaned.csv" in warning for warning in manifest["warnings"])
    sleep = pd.read_csv(tmp_path / "sleep_features.csv")
    assert "target_gpa" in sleep
    assert "include_primary" in sleep
    assert "sleep_duration__ordinal" in sleep
    assert int((~sleep["include_primary"].astype(bool)).sum()) == 2
    assert not any(column.startswith("work_") for column in sleep.columns)
