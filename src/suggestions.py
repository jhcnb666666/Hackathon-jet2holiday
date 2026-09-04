"""Frontend side of the suggestion-card interface with the agent service.

This module owns the *contract*: the shape of the context the frontend sends to
the supervisor agent, and the shape of the JSON the backend must return on
``ChatMessage.custom_data``. The backend follows these names.

Nothing here imports Streamlit. The Today page wraps these calls with its own
session-state cache, spinner and "refresh" button, so a 20-40 s supervisor run
never blocks first paint.

    ctx = build_context(daily_log_row, school="NTU", teaching_week=4)
    result = fetch_suggestions(ctx)          # real service; error state if unreachable
    for card in result.cards:
        render(card)

Response contract -- the backend puts this on the final message's ``custom_data``
(or on a dedicated ``type="custom"`` message)::

    {
      "suggestions": {
        "schema_version": 1,
        "generated_at": "2026-09-03T14:12:00Z",     # optional, stamped here if absent
        "summary": "One-line overall read of the day.",   # optional
        "support": {"label": "NTU UCS", "url": "https://..."},  # optional, shown first
        "cards": [
          {
            "id": "sleep-debt",                     # optional, stable key for the card
            "headline": "Protect tonight's sleep",
            "reasoning": "Two short nights and a 09:30 start tomorrow.",
            "action": "Start winding down by 23:30 tonight.",
            "metrics": ["sleep_hours", "class_start"],   # daily_log.csv column names
            "tone": "watch",                        # positive | nudge | watch
            "source_agent": "sleep-coach"           # optional, shown as a chip
          }
        ]
      }
    }

Request contract -- the frontend sends the raw numbers under ``agent_config``::

    agent_config = {"daily_context": { ...build_context() output... }}

plus a human-readable ``message`` from ``build_prompt()`` for agents that would
rather read prose.

Sub-agent progress is surfaced through ``stream_suggestions()``: any
``type="custom"`` message shaped like ``schema.task_data.TaskData`` is yielded as
a ``ProgressEvent`` before the final ``SuggestionSet``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# --- contract constants -------------------------------------------------------

SCHEMA_VERSION = 1
SUGGESTIONS_KEY = "suggestions"  # key on ChatMessage.custom_data
CONTEXT_KEY = "daily_context"  # key on agent_config
DEFAULT_AGENT = "langgraph-supervisor-agent"

TONES = ("positive", "nudge", "watch")
DEFAULT_TONE = "nudge"

# Metric identifiers a card may cite. These are the daily_log.csv columns plus a
# couple of derived ones the dashboard also shows.
KNOWN_METRICS = (
    "sleep_hours",
    "sleep_start",
    "exercise_minutes",
    "water_ml",
    "class_hours",
    "class_start",
    "class_end",
)


class SuggestionsUnavailable(RuntimeError):
    """The agent service could not be reached or returned an error."""


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

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SuggestionCard:
        tone = str(raw.get("tone", DEFAULT_TONE)).lower()
        metrics = raw.get("metrics") or []
        if isinstance(metrics, str):
            metrics = [metrics]
        return cls(
            headline=str(raw.get("headline", "")).strip(),
            reasoning=str(raw.get("reasoning", "")).strip(),
            action=str(raw.get("action", "")).strip(),
            metrics=[str(m) for m in metrics],
            tone=tone if tone in TONES else DEFAULT_TONE,
            source_agent=(str(raw["source_agent"]) if raw.get("source_agent") else None),
            id=(str(raw["id"]) if raw.get("id") else None),
        )


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

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SuggestionSet:
        cards_raw = payload.get("cards") or []
        cards = [
            SuggestionCard.from_dict(c)
            for c in cards_raw
            if isinstance(c, Mapping) and str(c.get("headline", "")).strip()
        ]
        support = payload.get("support")
        if not (isinstance(support, Mapping) and support.get("label") and support.get("url")):
            support = None
        return cls(
            cards=cards,
            summary=(str(payload["summary"]).strip() if payload.get("summary") else None),
            support=(dict(support) if support else None),
            generated_at=str(payload.get("generated_at") or _now_iso()),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        )

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
    """A sub-agent starting / working / finishing, for the progress strip."""

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
    """Normalise a daily_log row (plus calendar info) into the daily_context dict."""
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


def build_prompt(context: Mapping[str, Any]) -> str:
    """A short readable brief for agents that prefer prose over the raw dict."""
    bits: list[str] = []
    wk = context.get("teaching_week")
    when = context.get("date", "today")
    head = f"Wellbeing check for {when}"
    if wk is not None:
        head += f", teaching week {wk} at {context.get('school', '?')}"
    bits.append(head + ".")

    if "sleep_hours" in context:
        s = f"Slept {context['sleep_hours']:.1f} h"
        if context.get("sleep_start"):
            s += f" from {context['sleep_start']}"
        bits.append(s + ".")
    if "class_hours" in context:
        c = f"{context['class_hours']:.1f} h of class"
        if context.get("class_start") and context.get("class_end"):
            c += f" ({context['class_start']}-{context['class_end']})"
        bits.append(c + ".")
    if "exercise_minutes" in context:
        bits.append(f"{context['exercise_minutes']} min of movement.")
    if "water_ml" in context:
        bits.append(f"{context['water_ml']} ml of water logged.")

    bits.append(
        "Return suggestion cards on custom_data as per the frontend contract "
        "(headline, reasoning, action, metrics, tone). Do not diagnose and do "
        "not give numeric food or exercise targets."
    )
    return " ".join(bits)


# --- transport ----------------------------------------------------------


def service_url() -> str:
    """Resolve the agent service base URL the same way streamlit_app.py does."""
    url = os.getenv("AGENT_URL")
    if url:
        return url.rstrip("/")
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8080")
    return f"http://{host}:{port}"


def stream_suggestions(
    context: Mapping[str, Any],
    *,
    agent: str | None = None,
    base_url: str | None = None,
    timeout: float = 90.0,
) -> Iterator[ProgressEvent | SuggestionSet]:
    """Call the agent, yielding ProgressEvents as sub-agents report in, then one
    final SuggestionSet. Raises SuggestionsUnavailable on any transport error."""
    try:
        from client import AgentClient, AgentClientError
    except Exception as exc:  # noqa: BLE001 - toolkit not importable
        raise SuggestionsUnavailable(f"agent client unavailable: {exc}") from exc

    client = AgentClient(base_url=base_url or service_url(), get_info=False, timeout=timeout)
    client.agent = agent or DEFAULT_AGENT

    messages: list[Any] = []
    try:
        for chunk in client.stream(
            message=build_prompt(context),
            agent_config={CONTEXT_KEY: dict(context)},
            stream_tokens=False,
        ):
            if isinstance(chunk, str):
                continue
            messages.append(chunk)
            event = _progress_event(chunk)
            if event is not None:
                yield event
    except AgentClientError as exc:
        raise SuggestionsUnavailable(str(exc)) from exc

    result = parse_suggestions(messages)
    if result is not None:
        yield result


def fetch_suggestions(
    context: Mapping[str, Any],
    *,
    agent: str | None = None,
    base_url: str | None = None,
    timeout: float = 90.0,
    on_error_sample: bool = False,
) -> SuggestionSet:
    """Blocking call. Drains stream_suggestions and returns the final set.

    On a transport error (service down, etc.) returns an empty SuggestionSet with
    ``error`` set -- the UI is expected to show an honest "couldn't reach the
    advice service" state, not invented advice. Pass ``on_error_sample=True``
    only for local development / demos to get sample_suggestions() instead.
    """
    try:
        final: SuggestionSet | None = None
        for item in stream_suggestions(context, agent=agent, base_url=base_url, timeout=timeout):
            if isinstance(item, SuggestionSet):
                final = item
        if final is not None:
            return final
        reason = "agent returned no suggestions payload"
    except SuggestionsUnavailable as exc:
        reason = str(exc)

    if on_error_sample:
        fallback = sample_suggestions(context)
        fallback.error = reason
        return fallback
    return SuggestionSet(generated_at=_now_iso(), error=reason)


# --- response parsing --------------------------------------------------


def parse_suggestions(messages: list[Any]) -> SuggestionSet | None:
    """Scan agent messages for the suggestions payload on custom_data.

    Last payload wins. If none carry structured data but an AI message has text,
    return a SuggestionSet with only ``summary`` set so the UI can still show
    something and flag that the backend is not on the contract yet.
    """
    payload: Mapping[str, Any] | None = None
    fallback_text: str | None = None
    thread_id: str | None = None
    run_id: str | None = None

    for m in messages:
        cd = _attr(m, "custom_data")
        if isinstance(cd, Mapping) and isinstance(cd.get(SUGGESTIONS_KEY), Mapping):
            payload = cd[SUGGESTIONS_KEY]
        if _attr(m, "type") == "ai":
            text = _attr(m, "content")
            if text and str(text).strip():
                fallback_text = str(text).strip()
        run_id = _attr(m, "run_id") or run_id
        meta = _attr(m, "response_metadata")
        if isinstance(meta, Mapping) and meta.get("thread_id"):
            thread_id = str(meta["thread_id"])

    if payload is None and fallback_text is None:
        return None

    if payload is None:
        result = SuggestionSet(
            summary=fallback_text,
            generated_at=_now_iso(),
            error="agent replied without a structured suggestions payload",
        )
    else:
        result = SuggestionSet.from_payload(payload)

    result.thread_id = thread_id
    result.run_id = run_id
    return result


def _progress_event(message: Any) -> ProgressEvent | None:
    if _attr(message, "type") != "custom":
        return None
    cd = _attr(message, "custom_data")
    if not isinstance(cd, Mapping):
        return None
    # schema.task_data.TaskData shape, possibly nested under a "task_data" key.
    task = cd.get("task_data") if isinstance(cd.get("task_data"), Mapping) else cd
    if not isinstance(task, Mapping) or "state" not in task:
        return None
    return ProgressEvent(
        name=str(task.get("name") or "sub-agent"),
        state=str(task.get("state") or "running"),
        result=(str(task["result"]) if task.get("result") else None),
        detail=dict(task.get("data") or {}),
    )


# --- offline fallback -------------------------------------------------


def sample_suggestions(context: Mapping[str, Any]) -> SuggestionSet:
    """Plausible placeholder cards derived from the context numbers.

    Used when the service is unreachable so the UI is never empty. Always marked
    ``is_sample=True`` -- never present these as a real model response.
    """
    cards: list[SuggestionCard] = []

    sleep = _num(context.get("sleep_hours"))
    if sleep is not None and sleep < 7:
        cards.append(
            SuggestionCard(
                headline="Short night to notice",
                reasoning="Last night landed below your usual range.",
                action="Aim to start winding down a little earlier tonight.",
                metrics=["sleep_hours"],
                tone="watch",
                source_agent="sleep-coach",
                id="sample-sleep",
            )
        )

    water = _num(context.get("water_ml"))
    if water is not None and water < 1000:
        cards.append(
            SuggestionCard(
                headline="Water is low so far",
                reasoning="Not much logged yet against a long teaching day.",
                action="Fill your bottle before the next class.",
                metrics=["water_ml", "class_hours"],
                tone="nudge",
                source_agent="hydration",
                id="sample-water",
            )
        )

    class_hours = _num(context.get("class_hours"))
    move = _num(context.get("exercise_minutes"))
    if class_hours is not None and class_hours >= 4 and (move or 0) == 0:
        cards.append(
            SuggestionCard(
                headline="Long sit-down day",
                reasoning="Several hours of class and no movement logged.",
                action="Take a short walk between two of the blocks.",
                metrics=["class_hours", "exercise_minutes"],
                tone="nudge",
                source_agent="movement",
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
                source_agent="supervisor",
                id="sample-ok",
            )
        )

    return SuggestionSet(
        cards=cards,
        summary="Sample guidance — the agent service was not reachable.",
        generated_at=_now_iso(),
        is_sample=True,
    )


# --- helpers -----------------------------------------------------------


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pairs(value: Any) -> list[list[str]]:
    """Accept "07:00-07:30;18:00-18:45" or [("07:00","07:30"), ...] -> list of pairs."""
    if not value:
        return []
    if isinstance(value, str):
        out = []
        for part in value.split(";"):
            a, _, b = part.strip().partition("-")
            if a.strip() and b.strip():
                out.append([a.strip(), b.strip()])
        return out
    out = []
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
            "date": "2026-09-03",
            "sleep_hours": 6.2,
            "sleep_start": "01:10",
            "water_ml": 500,
            "exercise_minutes": 0,
            "class_hours": 4.0,
            "class_start": "09:30",
            "class_end": "16:20",
            "exercise_blocks": "",
        },
        school="NTU",
        teaching_week=4,
        phase="teaching",
    )
    print("context:", demo)
    print("prompt :", build_prompt(demo))
    result = fetch_suggestions(demo, timeout=5.0, on_error_sample=True)
    print(f"\nis_sample={result.is_sample}  error={result.error}")
    for card in result.cards:
        print(f"  [{card.tone}] {card.headline} -> {card.action}  ({', '.join(card.metrics)})")
