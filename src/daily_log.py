"""Daily log.

Preview on its own:  streamlit run src/daily_log.py

Class hours are not asked for. They come from the timetable, so the only
things left to enter are the three that change day to day.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timedelta
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from academic_calendar import SCHOOLS, academic_context, parse_weeks

LOG_PATH = Path("data/daily_log.csv")
TIMETABLE_PATH = Path("data/timetable.csv")

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

INK = "#16202A"
MUTED = "#6B7A87"
TRACK = "#F1F4F6"
WATER = "#3E9BC4"
SLEEP = "#3D4E9E"
MOVE = "#2F8F6B"
CLASS = "#E8A33D"

SIP = 250
BOTTLE = 500

SUGGEST_TONE = {"positive": "#2F8F6B", "nudge": "#3E7CB1", "watch": "#C0492F"}
METRIC_LABELS = {
    "sleep_hours": "sleep",
    "sleep_start": "bedtime",
    "exercise_minutes": "movement",
    "water_ml": "water",
    "class_hours": "class hours",
    "class_start": "class start",
    "class_end": "class end",
}

COLUMNS = [
    "date",
    "sleep_hours",
    "sleep_start",
    "exercise_minutes",
    "exercise_blocks",
    "water_ml",
    "class_hours",
    "class_start",
    "class_end",
]


def _hours(hhmm: str) -> float:
    h, m = str(hhmm).split(":")
    return int(h) + int(m) / 60


def _clock(hour: float) -> str:
    h, m = int(hour) % 24, int(round((hour % 1) * 60))
    return f"{h:02d}:{m:02d}"


def parse_movement(text: str) -> list[tuple[str, str]]:
    """"07:00-07:30;18:00-18:45" -> [("07:00", "07:30"), ("18:00", "18:45")]."""
    out = []
    for part in str(text).split(";"):
        a, _, b = part.strip().partition("-")
        if a.strip() and b.strip():
            out.append((a.strip(), b.strip()))
    return out


def _join_movement(blocks: list[tuple[str, str]]) -> str:
    return ";".join(f"{a}-{b}" for a, b in blocks)


def merge_movement(blocks: list[tuple[str, str]]) -> list[tuple[float, float]]:
    """Overlapping or touching stretches collapsed into their union, as sorted
    (start_hour, end_hour) pairs, so time covered by two stretches is counted once."""
    spans = sorted((_hours(a), _hours(b)) for a, b in blocks if _hours(b) > _hours(a))
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _movement_minutes(blocks: list[tuple[str, str]]) -> int:
    return int(round(sum(e - s for s, e in merge_movement(blocks)) * 60))


def load_timetable() -> pd.DataFrame:
    if not TIMETABLE_PATH.exists():
        return pd.DataFrame(columns=["day", "start", "end", "name", "weeks"])
    return pd.read_csv(TIMETABLE_PATH).fillna("")


def class_totals(blocks: pd.DataFrame, day: date, school: str) -> tuple[float, str, str]:
    ctx = academic_context(day, school)
    if not ctx.classes_expected or day.weekday() > 5 or blocks.empty:
        return 0.0, "", ""

    today = blocks[blocks["day"] == DAYS[day.weekday()]]
    spans = []
    for _, r in today.iterrows():
        weeks = parse_weeks(str(r.get("weeks", "")))
        if weeks is None or ctx.week in weeks:
            spans.append((_hours(r["start"]), _hours(r["end"])))
    if not spans:
        return 0.0, "", ""
    total = round(sum(e - s for s, e in spans), 1)
    return total, _clock(min(s for s, _ in spans)), _clock(max(e for _, e in spans))


def day_classes(blocks: pd.DataFrame, day: date, school: str) -> list[tuple[float, float, str]]:
    """Individual classes that run on `day`, each as (start_hour, end_hour, name)."""
    ctx = academic_context(day, school)
    if not ctx.classes_expected or day.weekday() > 5 or blocks.empty:
        return []
    rows = blocks[blocks["day"] == DAYS[day.weekday()]]
    spans = []
    for _, r in rows.iterrows():
        weeks = parse_weeks(str(r.get("weeks", "")))
        if weeks is None or ctx.week in weeks:
            spans.append(
                (_hours(r["start"]), _hours(r["end"]), str(r.get("name", "") or "Class"))
            )
    return sorted(spans)


TIMELINE_CSS = f"""
<style>
.ct-wrap {{ max-width: 640px; margin: .5rem 0 .2rem; padding: 0 14px; }}
.ct {{ position: relative; height: 56px; }}
.ct-base {{ position: absolute; top: 10px; left: 0; right: 0; height: 3px;
  border-radius: 2px; background: {TRACK}; }}
.ct-seg {{ position: absolute; top: 10px; height: 3px; border-radius: 2px; }}
.ct-dot {{ position: absolute; top: 5px; width: 11px; height: 11px; margin-left: -6px;
  border-radius: 50%; background: #fff; border: 2px solid; box-sizing: border-box; }}
.ct-lab {{ position: absolute; top: 22px; transform: translateX(-50%);
  text-align: center; white-space: nowrap; line-height: 1.25; }}
.ct-code {{ font-size: .72rem; font-weight: 600; color: {INK}; }}
.ct-time {{ font-size: .66rem; color: {MUTED}; }}
</style>
"""


def _timeline(spans: list[tuple[float, float, str, str]], color: str) -> str:
    """One row: an open circle at each start and end, joined by a bar, with a
    headline and a sub-label under it. `spans` is (start_hour, end_hour, top, bottom)."""
    if not spans:
        return ""
    lo = min(s for s, *_ in spans)
    hi = max(e for _, e, *_ in spans)
    width = hi - lo or 1.0

    rows = []
    for start, end, top, bottom in spans:
        left = (start - lo) / width * 100
        run = (end - start) / width * 100
        rows.append(
            f'<div class="ct-seg" style="left:{left:.2f}%;width:{run:.2f}%;background:{color}"></div>'
            f'<div class="ct-dot" style="left:{left:.2f}%;border-color:{color}"></div>'
            f'<div class="ct-dot" style="left:{left + run:.2f}%;border-color:{color}"></div>'
            f'<div class="ct-lab" style="left:{left + run / 2:.2f}%">'
            f'<span class="ct-code">{escape(top)}</span><br>'
            f'<span class="ct-time">{escape(bottom)}</span></div>'
        )
    return (
        TIMELINE_CSS
        + '<div class="ct-wrap"><div class="ct"><div class="ct-base"></div>'
        + "".join(rows)
        + "</div></div>"
    )


def class_timeline(spans: list[tuple[float, float, str]]) -> str:
    """Timeline of the day's classes: (start_hour, end_hour, course_name)."""
    return _timeline(
        [(s, e, name, f"{_clock(s)}–{_clock(e)}") for s, e, name in spans], CLASS
    )


def movement_timeline(spans: list[tuple[float, float]]) -> str:
    """Timeline of movement coverage (merged), each stretch labelled with its minutes."""
    rows = [(s, e, f"{round((e - s) * 60)} min", f"{_clock(s)}–{_clock(e)}") for s, e in spans]
    return _timeline(rows, MOVE)


def sleep_length(bed: time, wake: time) -> float:
    delta = (
        datetime.combine(date.today() + timedelta(days=1), wake)
        - datetime.combine(date.today(), bed)
    )
    hours = delta.total_seconds() / 3600
    return round(hours % 24 or 24, 1)


WATER_CSS = f"""
<style>
.bt-row {{ display: flex; gap: .5rem; align-items: flex-end; flex-wrap: wrap; margin: .2rem 0 .4rem; }}
.bt {{ position: relative; width: 34px; height: 90px; }}
.bt-neck {{ position: absolute; top: 0; left: 50%; width: 12px; height: 8px; margin-left: -6px;
  border: 1.5px solid {MUTED}; border-bottom: none; border-radius: 3px 3px 0 0; box-sizing: border-box; }}
.bt-body {{ position: absolute; top: 8px; left: 0; right: 0; bottom: 0; box-sizing: border-box;
  border: 1.5px solid {MUTED}; border-radius: 4px 4px 8px 8px; overflow: hidden; }}
.bt-fill {{ position: absolute; left: 0; right: 0; bottom: 0; background: {WATER}; }}
.bt-ghost .bt-body, .bt-ghost .bt-neck {{ border-style: dashed; opacity: .55; }}
</style>
"""


def water_bottles(total_ml: int) -> str:
    """One bottle per 500 ml, part-filled bottles shown at their real level.
    A dashed outline hints at the next bottle."""
    count = max(1, math.ceil(total_ml / BOTTLE))
    bottles = []
    for i in range(count + 1):
        fill = min(max(total_ml - i * BOTTLE, 0), BOTTLE) / BOTTLE
        ghost = " bt-ghost" if i == count else ""
        bottles.append(
            f'<div class="bt{ghost}"><div class="bt-neck"></div>'
            f'<div class="bt-body"><div class="bt-fill" style="height:{fill * 100:.1f}%"></div></div>'
            f"</div>"
        )
    return WATER_CSS + f'<div class="bt-row">{"".join(bottles)}</div>'


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(LOG_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def save_entry(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_log()
    df = df[df["date"] != row["date"]]
    df = pd.concat([df, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
    df.sort_values("date").to_csv(LOG_PATH, index=False)


# --- suggestion cards (backed by suggestions.py / the agent service) --------

SUGGEST_CSS = f"""
<style>
.sg-wrap {{ margin: .1rem 0 .5rem; }}
.sg-card {{ border: 1px solid #E7ECEF; border-left: 4px solid {MUTED}; border-radius: 8px;
  padding: .7rem .9rem; margin-bottom: .55rem; background: #FCFDFD; }}
.sg-top {{ display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; }}
.sg-head {{ font-size: 1rem; font-weight: 600; color: {INK}; }}
.sg-agent {{ font-size: .68rem; color: {MUTED}; border: 1px solid #E1E6E9; border-radius: 999px;
  padding: .05rem .5rem; white-space: nowrap; }}
.sg-why {{ font-size: .85rem; color: {MUTED}; margin: .25rem 0 .4rem; }}
.sg-do {{ font-size: .9rem; color: {INK}; }}
.sg-do b {{ color: {MUTED}; font-weight: 600; font-size: .72rem; text-transform: uppercase;
  letter-spacing: .04em; margin-right: .4rem; }}
.sg-metrics {{ margin-top: .5rem; }}
.sg-metrics span {{ font-size: .72rem; color: {MUTED}; background: {TRACK}; border-radius: 4px;
  padding: .08rem .4rem; margin-right: .3rem; }}
.sg-support {{ border: 1px solid #E0B23C; background: #FDF7E7; border-radius: 8px;
  padding: .7rem .9rem; margin-bottom: .55rem; font-size: .88rem; color: {INK}; }}
</style>
"""


def _ago(iso: str) -> str:
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return "recently"
    secs = (datetime.now(then.tzinfo) - then).total_seconds()
    if secs < 90:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return then.strftime("%d %b %H:%M")


def _suggestion_card_html(card) -> str:
    color = SUGGEST_TONE.get(card.tone, MUTED)
    agent = f'<span class="sg-agent">{escape(card.source_agent)}</span>' if card.source_agent else ""
    why = f'<div class="sg-why">{escape(card.reasoning)}</div>' if card.reasoning else ""
    action = (
        f'<div class="sg-do"><b>Today</b>{escape(card.action)}</div>' if card.action else ""
    )
    metrics = ""
    if card.metrics:
        chips = "".join(
            f"<span>{escape(METRIC_LABELS.get(m, m))}</span>" for m in card.metrics
        )
        metrics = f'<div class="sg-metrics">{chips}</div>'
    return (
        f'<div class="sg-card" style="border-left-color:{color}">'
        f'<div class="sg-top"><span class="sg-head">{escape(card.headline)}</span>{agent}</div>'
        f"{why}{action}{metrics}</div>"
    )


def _render_suggestion_set(result) -> None:
    parts = [SUGGEST_CSS, '<div class="sg-wrap">']
    if result.support:
        parts.append(
            '<div class="sg-support">If you want to talk to someone, '
            f'<a href="{escape(result.support["url"])}" target="_blank" rel="noopener">'
            f'{escape(result.support["label"])}</a> is there for students.</div>'
        )
    if result.is_sample:
        parts.append(
            '<div class="sg-why">Sample guidance — the advice service is not connected.</div>'
        )
    for card in result.cards:
        parts.append(_suggestion_card_html(card))
    parts.append("</div>")
    st.html("".join(parts))

    if not result.cards:
        if result.summary:
            st.info(result.summary)
        st.caption(
            "Couldn't reach the advice service — try Refresh in a moment."
            if result.error
            else "No suggestions came back."
        )
        return
    if result.summary:
        st.caption(result.summary)


# One check-in per fixed time of day. The frontend fills a slot when someone
# opens the app after that time; unattended pushes would need a real scheduler.
CHECK_IN_SLOTS = ("08:00", "12:00", "16:00", "20:00", "22:00")
SUGGEST_STORE = Path("data/suggestions.json")


def _ampm(hhmm: str) -> str:
    h = int(hhmm[:2])
    return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"


def _due_slot(now: time) -> str | None:
    """The latest slot whose time has arrived today, or None before the first."""
    mins = now.hour * 60 + now.minute
    passed = [s for s in CHECK_IN_SLOTS if int(s[:2]) * 60 + int(s[3:]) <= mins]
    return passed[-1] if passed else None


def _store_read() -> dict:
    try:
        return json.loads(SUGGEST_STORE.read_text("utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _store_write(store: dict) -> None:
    try:
        SUGGEST_STORE.parent.mkdir(parents=True, exist_ok=True)
        SUGGEST_STORE.write_text(json.dumps(store, indent=2), "utf-8")
    except OSError:
        pass


def _run_stream(context: dict, label: str):
    """Call the agent, showing sub-agent progress in an st.status. Returns a
    SuggestionSet (with .error set on any failure)."""
    from suggestions import ProgressEvent, SuggestionSet, stream_suggestions

    status = st.status(label, expanded=True)
    result = SuggestionSet(error="no response")
    mark = {"new": "•", "running": "▸", "complete": "✓"}
    try:
        for item in stream_suggestions(context):
            if isinstance(item, ProgressEvent):
                status.write(f"{mark.get(item.state, '•')} {item.name}")
            else:
                result = item
    except Exception as exc:  # noqa: BLE001 - surfaced in the card area
        result = SuggestionSet(error=str(exc))
    status.update(
        label="Done" if result.cards else "Couldn't generate",
        state="complete" if result.cards else "error",
        expanded=False,
    )
    return result


def render_suggestions(day, school, ctx, prior: dict, class_spans: list) -> None:
    """Check-in cards, one per fixed time of day. The due slot generates itself
    once (rendered last in render_daily_log, so it never blocks first paint);
    results are cached per day in session and mirrored to data/suggestions.json."""
    from suggestions import SuggestionSet, build_context

    st.markdown("#### Today's check-ins")
    if not prior:
        st.caption("Save today's log — check-ins start after that.")
        return

    is_today = day == date.today()
    day_key = str(day)
    store = _store_read()
    disk_slots: dict = dict(store.get(day_key, {}))

    sess_key = f"suggest_{day}"
    if not isinstance(st.session_state.get(sess_key), dict):
        st.session_state[sess_key] = {}
    live: dict = st.session_state[sess_key]
    for slot, raw in disk_slots.items():
        live.setdefault(slot, SuggestionSet.from_dict(raw))
    failed: set = st.session_state.setdefault(f"{sess_key}_failed", set())

    current = _due_slot(datetime.now().time()) if is_today else CHECK_IN_SLOTS[-1]

    def _context() -> dict:
        c = build_context(prior, school=school, teaching_week=ctx.week, phase=ctx.phase)
        if class_spans:
            c["class_blocks"] = [[_clock(s), _clock(e), n] for s, e, n in class_spans]
        return c

    def _commit(slot: str, result) -> None:
        live[slot] = result
        disk_slots[slot] = result.to_dict()
        store[day_key] = disk_slots
        _store_write(store)

    # auto-run the current slot once, when it is due and not done yet
    if is_today and current and current not in live and current not in failed:
        result = _run_stream(_context(), f"Preparing your {_ampm(current)} check-in…")
        if result.cards:
            _commit(current, result)
            st.rerun()
        failed.add(current)

    shown = [s for s in CHECK_IN_SLOTS if s in live]
    if is_today and current and current not in shown:
        shown.append(current)
    if not shown:
        if is_today:
            st.caption(f"First check-in at {_ampm(current or CHECK_IN_SLOTS[0])}.")
            return
        shown = list(CHECK_IN_SLOTS)  # other day: let the user pick any slot to generate

    default = current if current in shown else shown[-1]
    picked = st.radio(
        "Check-in",
        shown,
        index=shown.index(default),
        horizontal=True,
        format_func=lambda s: _ampm(s) + (" · now" if is_today and s == current else ""),
        label_visibility="collapsed",
        key=f"slot_pick_{day}",
    )

    chosen = live.get(picked)
    if chosen is None:
        if st.button(f"Generate {_ampm(picked)} check-in", key=f"gen_{day}_{picked}"):
            result = _run_stream(_context(), f"Preparing your {_ampm(picked)} check-in…")
            if result.cards:
                _commit(picked, result)
            failed.discard(picked)
            st.rerun()
        if picked in failed:
            st.caption("Couldn't reach the advice service — try again in a moment.")
        return

    _render_suggestion_set(chosen)
    row = st.columns([3, 1])
    stamp = _ago(chosen.generated_at) if chosen.generated_at else "earlier"
    row[0].caption(f"{_ampm(picked)} check-in · generated {stamp}")
    if row[1].button("Refresh", key=f"refresh_{day}_{picked}", use_container_width=True):
        result = _run_stream(_context(), "Refreshing…")
        if result.cards:
            _commit(picked, result)
        st.rerun()


def render_daily_log() -> None:
    top = st.columns([1.3, 1.2, 2])
    day = top[0].date_input("Date", value=date.today(), label_visibility="collapsed")
    school = top[1].selectbox("School", SCHOOLS, label_visibility="collapsed")
    ctx = academic_context(day, school)
    top[2].markdown(
        f'<div style="padding-top:.45rem;color:{MUTED};font-size:.85rem">'
        f'{day:%A %d %b} · {ctx.label}{" · " + ctx.holiday if ctx.holiday else ""}</div>',
        unsafe_allow_html=True,
    )

    existing = load_log()
    prior = existing[existing["date"] == day]
    prior = prior.iloc[0].to_dict() if not prior.empty else {}

    water_key = f"water_{day}"
    if water_key not in st.session_state:
        st.session_state[water_key] = int(prior.get("water_ml", 0) or 0)

    blocks = load_timetable()
    auto_hours, auto_start, auto_end = class_totals(blocks, day, school)

    # Rendered at the top of the page, but filled last (see end of function) so
    # a slow check-in generation never blocks the rest of the page painting.
    check_in_slot = st.container()
    st.divider()

    st.markdown("#### From your timetable")
    if auto_hours:
        st.markdown(
            f'<span style="font-size:1.5rem;font-weight:600;color:{INK}">{auto_hours:.1f}</span>'
            f'<span style="color:{MUTED}"> h of class · {auto_start}–{auto_end}</span>',
            unsafe_allow_html=True,
        )
        st.html(class_timeline(day_classes(blocks, day, school)))
    else:
        st.markdown(f'<span style="color:{MUTED}">No class scheduled.</span>', unsafe_allow_html=True)
    with st.expander("Different from the plan?"):
        auto_hours = st.number_input(
            "Hours actually in class", 0.0, 14.0, float(auto_hours), 0.5
        )

    st.divider()
    st.markdown("#### Last night")
    s1, s2, s3 = st.columns([1, 1, 1.4])
    bed = s1.time_input("Asleep at", value=time(23, 30), step=900)
    wake = s2.time_input("Awake at", value=time(7, 30), step=900)
    hours = sleep_length(bed, wake)
    s3.markdown(
        f'<div style="padding-top:1.9rem;font-size:1.3rem;font-weight:600;color:{SLEEP}">'
        f'{hours:.1f} h</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### Water")
    total = st.session_state[water_key]
    st.html(water_bottles(total))
    st.markdown(
        f'<div style="margin:.35rem 0 .6rem;color:{MUTED};font-size:.85rem">'
        f'<span style="color:{INK};font-size:1.3rem;font-weight:600">{total:,}</span> ml'
        f' · {total / BOTTLE:.1f} bottles</div>',
        unsafe_allow_html=True,
    )
    w1, w2, w3, _ = st.columns([1.1, 1, 1, 2])
    if w1.button(f"+{SIP} ml", type="primary", use_container_width=True):
        st.session_state[water_key] += SIP
        st.rerun()
    if w2.button("Undo", use_container_width=True, disabled=total == 0):
        st.session_state[water_key] = max(0, total - SIP)
        st.rerun()
    if w3.button("Reset", use_container_width=True, disabled=total == 0):
        st.session_state[water_key] = 0
        st.rerun()

    st.divider()
    st.markdown("#### Movement")
    move_key = f"move_{day}"
    if move_key not in st.session_state:
        st.session_state[move_key] = parse_movement(prior.get("exercise_blocks", ""))

    window = st.slider(
        "Movement window",
        min_value=time(5, 0),
        max_value=time(23, 45),
        value=(time(18, 0), time(18, 30)),
        step=timedelta(minutes=15),
        format="HH:mm",
        label_visibility="collapsed",
    )
    if st.button("Add this stretch", use_container_width=True):
        if window[1] <= window[0]:
            st.warning("Drag the handles apart to cover a real stretch of time.")
        else:
            pair = (window[0].strftime("%H:%M"), window[1].strftime("%H:%M"))
            if pair not in st.session_state[move_key]:
                st.session_state[move_key].append(pair)
            st.rerun()

    st.session_state[move_key].sort()
    segments = st.session_state[move_key]
    minutes = _movement_minutes(segments)

    if segments:
        st.html(movement_timeline(merge_movement(segments)))
        chips = st.columns(min(len(segments), 4))
        for i, (a, b) in enumerate(segments):
            mins = round((_hours(b) - _hours(a)) * 60)
            if chips[i % len(chips)].button(
                f"✕  {a}–{b} · {mins}m", key=f"mv_rm_{day}_{i}", use_container_width=True
            ):
                st.session_state[move_key].remove((a, b))
                st.rerun()
        st.caption(f"{minutes} min active today.")
    else:
        st.caption("Walking to class counts. Add a stretch if you moved, or leave it empty.")

    st.divider()
    if st.button("Save today", type="primary", use_container_width=True):
        save_entry(
            {
                "date": day,
                "sleep_hours": hours,
                "sleep_start": bed.strftime("%H:%M"),
                "exercise_minutes": minutes,
                "exercise_blocks": _join_movement(segments),
                "water_ml": st.session_state[water_key],
                "class_hours": auto_hours,
                "class_start": auto_start,
                "class_end": auto_end,
            }
        )
        st.success(f"Saved to {LOG_PATH}")

    if not existing.empty:
        with st.expander(f"Past entries ({len(existing)})"):
            st.dataframe(existing.sort_values("date", ascending=False), hide_index=True)

    with check_in_slot:
        render_suggestions(day, school, ctx, prior, day_classes(blocks, day, school))


if __name__ == "__main__":
    st.set_page_config(page_title="Daily log", layout="centered", menu_items={})
    st.title("Today")
    render_daily_log()
