import json
from datetime import date, time

import pytest

from agents.sleep_model import SleepModel
from schema.student_wellness import StudentSchedule, WellnessSignals
from sleep_clustering import (
    FEATURE_NAMES,
    bedtime_category,
    save_artifact,
    train_artifact,
)


def _write_survey(path, gpa_values=("2.0", "4.0", "3.5")):
    path.write_text(
        "record_id,current_gpa,sleep_duration,bedtime,wake_time,gpa_quality_flag\n"
        f"1,{gpa_values[0]},7 hours,Between 10pm and 12am,Between 6am and 8am,valid\n"
        f"2,{gpa_values[1]},6 hours,Between 12am and 2am,Between 8am and 10am,valid\n"
        f"3,{gpa_values[2]},7 hours,Between 10pm and 12am,Between 6am and 8am,valid\n",
        encoding="utf-8",
    )


def test_training_ignores_gpa_values(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_survey(first)
    _write_survey(second, gpa_values=("4.0", "1.0", "0.0"))

    assert train_artifact(first, clusters=2) == train_artifact(second, clusters=2)


def test_artifact_contains_only_sleep_features(tmp_path):
    survey = tmp_path / "survey.csv"
    _write_survey(survey)
    artifact = train_artifact(survey, clusters=2)

    assert artifact["features"] == list(FEATURE_NAMES)
    assert "current_gpa" not in json.dumps(artifact["clusters"])
    assert artifact["model_type"] == "categorical_k_modes"
    assert artifact["training"]["algorithm"] == "deterministic weighted k-modes"
    assert sum(cluster["size"] for cluster in artifact["clusters"]) == artifact["training_rows"]


def test_bedtime_categories_cross_midnight():
    assert bedtime_category(23, 30) == "Between 10pm and 12am"
    assert bedtime_category(0, 30) == "Between 12am and 2am"
    assert bedtime_category(3, 0) == "After 2am"


@pytest.mark.asyncio
async def test_sleep_model_returns_pipeline_metric_score(tmp_path):
    survey = tmp_path / "survey.csv"
    model_path = tmp_path / "sleep_model.json"
    _write_survey(survey)
    save_artifact(train_artifact(survey, clusters=2), model_path)
    model = SleepModel(model_path)

    result = await model.score(
        StudentSchedule(student_id="student-1", day=date(2026, 9, 6)),
        WellnessSignals(
            sleep_duration_hours=7,
            sleep_onset_time=time(23, 0),
            wake_time=time(6, 0),
        ),
    )

    assert result.metric == "sleep"
    assert result.score == 100
    assert result.level == "good"
    assert any("Nearest survey pattern" in item for item in result.evidence)


@pytest.mark.asyncio
async def test_sleep_model_can_derive_duration_from_times(tmp_path):
    survey = tmp_path / "survey.csv"
    model_path = tmp_path / "sleep_model.json"
    _write_survey(survey)
    save_artifact(train_artifact(survey, clusters=2), model_path)
    model = SleepModel(model_path)

    result = await model.score(
        StudentSchedule(
            student_id="student-1",
            day=date(2026, 9, 6),
            sleep_time=time(1, 0),
            wake_time=time(6, 0),
        ),
        WellnessSignals(),
    )

    assert result.score == 40
    assert result.level == "poor"
    assert result.evidence[0].startswith("Sleep duration: 5.0 hours")
