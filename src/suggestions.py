"""Frontend adapter to the in-repo student-wellness pipeline.

The Today page collects structured numbers (sleep / water / movement + the class
timetable). This module turns one day's context into the pipeline's input models
(:class:`schema.student_wellness.StudentSchedule` /
:class:`~schema.student_wellness.WellnessSignals`), runs
:class:`agents.student_wellness.StudentWellnessPipeline` in-process, and maps the
returned :class:`~schema.student_wellness.WellnessReport` onto the card model the
Today page renders.

Nothing here imports Streamlit. The Today page wraps these calls with its own
per-slot cache, spinner and refresh button, so the pipeline run never blocks
first paint.

    ctx = build_context(daily_log_row, school="NTU", teaching_week=4)
    result = fetch_suggestions(ctx)
    for card in result.cards:
        render(card)

The final English recommendation comes from ``AWSBedrockAdviceModel`` when AWS is
configured; otherwise it falls back to the pipeline's local ``LocalAdviceModel``
(the sleep / activity scores are computed locally either way).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

SCHEMA_VERSION = 1

TONES = ("positive", "nudge", "watch")
DEFAULT_TONE = "nudge"

_LEVEL_TONE = {"good": "positive", "attention": "nudge", "poor": "watch"}
_PRIORITY_TONE = {"low": "positive", "medium": "nudge", "high": "watch"}
_METRIC_LABEL = {"sleep": "Sleep", "physical_activity": "Movement"}

# Metric identifiers a card may cite (daily_log.csv columns + the pipeline's).
KNOWN_METRICS = (
    "sleep_hours",
    "sleep_start",
    "exercise_minutes",
    "water_ml",
    "class_hours",
    "class_start",
    "class_end",
    "sleep",
    "physical_activity",
)


class SuggestionsUnavailable(RuntimeError):
    """The wellness pipeline could not run (import error, model failure, ...)."""


# --- data model -------------------------------------------------------------


@dataclass
class SuggestionCard:
    headline: str
    reasoning: str = ""
    action: str = ""
    metrics: list[str] = field(default_factory=list)
    tone: str = DEFAULT_TONE
    source_agent: str | None = None
    id: str | None = None


@dataclass
class SuggestionSet:
    cards: list[SuggestionCard] = field(default_factory=list)
    summary: str | None = None
    support: dict[str, str] | None = None
    generated_at: str | None = None
    schema_version: int = SCHEMA_VERSION
    thread_id: str | None = None
    run_id: str | None = None
    is_sample: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.is_sample and self.error is None and bool(self.cards)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON persistence (see daily_log's check-in store)."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SuggestionSet:
        return cls(
            cards=[
                SuggestionCard(**{k: c[k] for k in c if k in SuggestionCard.__annotations__})
                for c in raw.get("cards", [])
                if isinstance(c, Mapping)
            ],
            summary=raw.get("summary"),
            support=(dict(raw["support"]) if raw.get("support") else None),
            generated_at=raw.get("generated_at"),
            schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
            thread_id=raw.get("thread_id"),
            run_id=raw.get("run_id"),
            is_sample=bool(raw.get("is_sample", False)),
            error=raw.get("error"),
        )


@dataclass
class ProgressEvent:
    """A pipeline step starting / finishing, for the progress strip."""

    name: str
    state: str  # "new" | "running" | "complete"
    result: str | None = None  # "success" | "error"
    detail: dict[str, Any] = field(default_factory=dict)


# --- request side ---------------------------------------------------------


def build_context(
    row: Mapping[str, Any],
    *,
    school: str,
    teaching_week: int | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    """Normalise a daily_log row (plus calendar info) into a context dict.

    The Today page augments the result with ``class_blocks`` and ``triggered_at``
    before handing it to :func:`stream_suggestions`.
    """
    ctx: dict[str, Any] = {"school": school}
    if row.get("date") is not None:
        ctx["date"] = str(row["date"])
    if teaching_week is not None:
        ctx["teaching_week"] = int(teaching_week)
    if phase:
        ctx["phase"] = phase

    for key in ("sleep_hours", "class_hours"):
        value = _num(row.get(key))
        if value is not None:
            ctx[key] = value
    for key in ("exercise_minutes", "water_ml"):
        value = _num(row.get(key))
        if value is not None:
            ctx[key] = int(value)
    for key in ("sleep_start", "class_start", "class_end"):
        if row.get(key):
            ctx[key] = str(row[key])

    blocks = _pairs(row.get("exercise_blocks"))
    if blocks:
        ctx["exercise_blocks"] = blocks
    class_blocks = row.get("class_blocks")
    if class_blocks:
        ctx["class_blocks"] = [list(map(str, b)) for b in class_blocks]
    return ctx


# --- pipeline glue ------------------------------------------------------


def _pipeline_inputs(context: Mapping[str, Any]) -> tuple[Any, Any, time]:
    """Build (StudentSchedule, WellnessSignals, triggered_at) from a context dict."""
    from schema.student_wellness import ScheduleItem, StudentSchedule, WellnessSignals

    day = _parse_date(context.get("date")) or date.today()

    items = []
    for block in context.get("class_blocks", []):
        start_t = _parse_time(block[0]) if len(block) > 0 else None
        end_t = _parse_time(block[1]) if len(block) > 1 else None
        if start_t is None or end_t is None:
            continue
        title = str(block[2]) if len(block) > 2 and block[2] else "Class"
        items.append(
            ScheduleItem(
                start=datetime.combine(day, start_t),
                end=datetime.combine(day, end_t),
                title=title,
                category="class",
            )
        )

    onset = _parse_time(context.get("sleep_start"))
    duration = _num(context.get("sleep_hours"))
    wake = _add_hours(onset, duration) if (onset is not None and duration is not None) else None

    schedule = StudentSchedule(
        student_id=str(context.get("student_id", "demo")),
        day=day,
        items=items,
        sleep_time=onset,
        wake_time=wake,
    )
    signals = WellnessSignals(
        sleep_duration_hours=duration,
        sleep_onset_time=onset,
        wake_time=wake,
        exercise_minutes=_num(context.get("exercise_minutes")),
    )
    triggered_at = _parse_time(context.get("triggered_at")) or datetime.now().time().replace(
        microsecond=0
    )
    return schedule, signals, triggered_at


def _build_pipeline(advice: Any) -> Any:
    from agents.physical_activity_model import PhysicalActivityModel
    from agents.sleep_model import SleepModel
    from agents.student_wellness import StudentWellnessPipeline
    from agents.wellness_components import FixedSelector

    return StudentWellnessPipeline(
        models={"sleep": SleepModel(), "physical_activity": PhysicalActivityModel()},
        selector=FixedSelector(),
        advice=advice,
    )


def _aws_enabled() -> bool:
    try:
        from core.settings import settings

        return bool(getattr(settings, "USE_AWS_BEDROCK", False))
    except Exception:  # noqa: BLE001
        return False


def _run_pipeline(schedule: Any, signals: Any, triggered_at: time) -> tuple[Any, str]:
    """Run the pipeline. Use the AWS advice model when it is configured; otherwise
    (or on any AWS failure) use the pipeline's local advice model. The specialist
    sleep / activity scores are computed locally and identical either way.
    Returns (WellnessReport, mode) where mode is "aws" or "local"."""
    import asyncio

    if _aws_enabled():
        try:
            from agents.aws_advice_model import AWSBedrockAdviceModel
            from core import get_model
            from schema.models import AWSModelName

            advice = AWSBedrockAdviceModel(get_model(AWSModelName.BEDROCK_HAIKU))
            report = asyncio.run(_build_pipeline(advice).run(schedule, signals, triggered_at))
            return report, "aws"
        except Exception:  # noqa: BLE001 - AWS unreachable at runtime -> local fallback
            pass

    from agents.wellness_components import LocalAdviceModel

    report = asyncio.run(_build_pipeline(LocalAdviceModel()).run(schedule, signals, triggered_at))
    return report, "local"


def _report_to_set(report: Any, mode: str) -> SuggestionSet:
    rec = report.recommendation
    cards = [
        SuggestionCard(
            headline="Today's focus",
            reasoning=(
                "Weighing " + ", ".join(_metric_label(m) for m in rec.based_on)
                if rec.based_on
                else ""
            ),
            action=str(rec.text).strip(),
            metrics=list(rec.based_on),
            tone=_PRIORITY_TONE.get(rec.priority, DEFAULT_TONE),
            source_agent="advice · local" if mode == "local" else "advice",
            id="focus",
        )
    ]
    for score in report.scores:
        cards.append(
            SuggestionCard(
                headline=f"{_metric_label(score.metric)} — {score.score:.0f}/100",
                reasoning="  ·  ".join(score.evidence),
                action="",
                metrics=[score.metric],
                tone=_LEVEL_TONE.get(score.level, DEFAULT_TONE),
                source_agent=score.metric,
                id=score.metric,
            )
        )

    result = SuggestionSet(cards=cards, generated_at=_now_iso())
    if mode == "local":
        result.summary = "Recommendation written locally (AWS advice model not configured)."
    return result


def stream_suggestions(
    context: Mapping[str, Any],
    **_ignored: Any,
) -> Iterator[ProgressEvent | SuggestionSet]:
    """Run the wellness pipeline, yielding ProgressEvents then one SuggestionSet.

    Raises :class:`SuggestionsUnavailable` if the pipeline cannot run at all.
    Extra keyword arguments are accepted and ignored for call-site compatibility.
    """
    try:
        schedule, signals, triggered_at = _pipeline_inputs(context)
    except Exception as exc:  # noqa: BLE001
        raise SuggestionsUnavailable(f"could not build pipeline inputs: {exc}") from exc

    yield ProgressEvent(name="scoring sleep & movement", state="running")
    try:
        report, mode = _run_pipeline(schedule, signals, triggered_at)
    except Exception as exc:  # noqa: BLE001
        raise SuggestionsUnavailable(str(exc)) from exc

    for score in report.scores:
        yield ProgressEvent(name=score.metric, state="complete", result="success")
    yield ProgressEvent(
        name="advice · local" if mode == "local" else "advice", state="complete", result="success"
    )
    yield _report_to_set(report, mode)


def fetch_suggestions(
    context: Mapping[str, Any],
    *,
    on_error_sample: bool = False,
    **_ignored: Any,
) -> SuggestionSet:
    """Blocking wrapper: drain :func:`stream_suggestions`, return the final set.

    On failure returns an empty SuggestionSet with ``error`` set (the UI shows an
    honest "couldn't generate" state). Pass ``on_error_sample=True`` only for
    local development to get :func:`sample_suggestions` instead.
    """
    try:
        final: SuggestionSet | None = None
        for item in stream_suggestions(context):
            if isinstance(item, SuggestionSet):
                final = item
        if final is not None:
            return final
        reason = "pipeline produced no report"
    except SuggestionsUnavailable as exc:
        reason = str(exc)

    if on_error_sample:
        fallback = sample_suggestions(context)
        fallback.error = reason
        return fallback
    return SuggestionSet(generated_at=_now_iso(), error=reason)


# --- offline fallback -------------------------------------------------


def sample_suggestions(context: Mapping[str, Any]) -> SuggestionSet:
    """Plausible placeholder cards from the context numbers, for local dev only.

    Always marked ``is_sample=True`` -- never shown to a real user.
    """
    cards: list[SuggestionCard] = []

    sleep = _num(context.get("sleep_hours"))
    if sleep is not None and sleep < 7:
        cards.append(
            SuggestionCard(
                headline="Short night to notice",
                reasoning="Last night landed below the 7-hour reference.",
                action="Aim to start winding down a little earlier tonight.",
                metrics=["sleep"],
                tone="watch",
                source_agent="sleep",
                id="sample-sleep",
            )
        )

    move = _num(context.get("exercise_minutes"))
    class_hours = _num(context.get("class_hours"))
    if class_hours is not None and class_hours >= 4 and (move or 0) == 0:
        cards.append(
            SuggestionCard(
                headline="Long sit-down day",
                reasoning="Several hours of class and no movement logged.",
                action="Take a short walk between two of the blocks.",
                metrics=["physical_activity"],
                tone="nudge",
                source_agent="physical_activity",
                id="sample-move",
            )
        )

    if not cards:
        cards.append(
            SuggestionCard(
                headline="Nothing pressing",
                reasoning="Today's numbers all sit inside their usual ranges.",
                action="Keep the routine you have going.",
                metrics=[],
                tone="positive",
                source_agent="advice",
                id="sample-ok",
            )
        )

    return SuggestionSet(
        cards=cards,
        summary="Sample guidance — the wellness pipeline was not run.",
        generated_at=_now_iso(),
        is_sample=True,
    )


# --- helpers -----------------------------------------------------------


def _metric_label(metric: str) -> str:
    return _METRIC_LABEL.get(metric, metric.replace("_", " ").title())


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time(value: Any) -> time | None:
    if isinstance(value, time):
        return value
    if not value:
        return None
    try:
        hh, mm = str(value).split(":")[:2]
        return time(int(hh) % 24, int(mm) % 60)
    except (TypeError, ValueError):
        return None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _add_hours(onset: time, hours: float) -> time:
    return (
        (datetime.combine(date.today(), onset) + timedelta(hours=hours))
        .time()
        .replace(microsecond=0)
    )


def _pairs(value: Any) -> list[list[str]]:
    """Accept "07:00-07:30;18:00-18:45" or [("07:00","07:30"), ...] -> list of pairs.

    Anything else (None, NaN from an empty CSV cell, ...) yields an empty list.
    """
    out: list[list[str]] = []
    if isinstance(value, str):
        for part in value.split(";"):
            a, _, b = part.strip().partition("-")
            if a.strip() and b.strip():
                out.append([a.strip(), b.strip()])
        return out
    if not isinstance(value, (list, tuple)):
        return out
    for item in value:
        pair = list(item)
        if len(pair) >= 2 and str(pair[0]) and str(pair[1]):
            out.append([str(pair[0]), str(pair[1])])
    return out


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    demo = build_context(
        {
            "date": "2026-09-05",
            "sleep_hours": 6.2,
            "sleep_start": "01:10",
            "exercise_minutes": 20,
            "class_hours": 4.0,
        },
        school="NTU",
        teaching_week=4,
    )
    demo["class_blocks"] = [["09:30", "10:20", "EE2103 TUT"], ["14:30", "16:20", "ML0004 TUT"]]
    demo["triggered_at"] = "20:00"
    print("context:", demo)
    out = fetch_suggestions(demo, on_error_sample=True)
    print(f"\nis_sample={out.is_sample}  error={out.error}  summary={out.summary}")
    for card in out.cards:
        print(f"  [{card.tone}] {card.headline}  <-  {card.source_agent}")
        if card.reasoning:
            print(f"      {card.reasoning}")
        if card.action:
            print(f"      -> {card.action}")
