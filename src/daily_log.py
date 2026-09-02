"""Daily log.

Preview on its own:  streamlit run src/daily_log.py

Class hours are not asked for. They come from the timetable, so the only
things left to enter are the three that change day to day.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from academic_calendar import SCHOOLS, academic_context, parse_weeks

LOG_PATH = Path("data/daily_log.csv")
TIMETABLE_PATH = Path("data/timetable.csv")

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

INK = "#16202A"
MUTED = "#6B7A87"
WATER = "#3E9BC4"
SLEEP = "#3D4E9E"
MOVE = "#2F8F6B"
CLASS = "#E8A33D"

SIP = 250
BOTTLE = 500

COLUMNS = [
    "date",
    "sleep_hours",
    "sleep_start",
    "exercise_minutes",
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


def sleep_length(bed: time, wake: time) -> float:
    delta = (
        datetime.combine(date.today() + timedelta(days=1), wake)
        - datetime.combine(date.today(), bed)
    )
    hours = delta.total_seconds() / 3600
    return round(hours % 24 or 24, 1)


def bottles_svg(total_ml: int) -> str:
    """One bottle per 500 ml, part-filled bottles shown at their real level."""
    count = max(1, math.ceil(total_ml / BOTTLE))
    parts = []
    for i in range(count + 1):  # trailing outline hints at the next bottle
        fill = min(max(total_ml - i * BOTTLE, 0), BOTTLE) / BOTTLE
        ghost = i == count
        body_top, body_bottom = 18, 106
        y = body_bottom - fill * (body_bottom - body_top)
        parts.append(
            f'<svg viewBox="0 0 44 112" width="40" height="102" role="img">'
            f'<defs><clipPath id="c{i}">'
            f'<path d="M18 8h8v9h1a6 6 0 0 1 5 6v79a4 4 0 0 1-4 4H16a4 4 0 0 1-4-4V23a6 6 0 0 1 5-6h1z"/>'
            f"</clipPath></defs>"
            f'<rect x="16" y="1" width="12" height="7" rx="2" fill="{MUTED}" opacity="{0.2 if ghost else 0.5}"/>'
            f'<rect x="0" y="{y:.1f}" width="44" height="{body_bottom - y:.1f}" '
            f'fill="{WATER}" clip-path="url(#c{i})"/>'
            f'<path d="M18 8h8v9h1a6 6 0 0 1 5 6v79a4 4 0 0 1-4 4H16a4 4 0 0 1-4-4V23a6 6 0 0 1 5-6h1z" '
            f'fill="none" stroke="{MUTED}" stroke-width="1.5" opacity="{0.25 if ghost else 0.55}"/>'
            f"</svg>"
        )
    return f'<div style="display:flex;gap:.45rem;align-items:flex-end;flex-wrap:wrap">{"".join(parts)}</div>'


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

    auto_hours, auto_start, auto_end = class_totals(load_timetable(), day, school)

    st.markdown("#### From your timetable")
    if auto_hours:
        st.markdown(
            f'<span style="font-size:1.5rem;font-weight:600;color:{INK}">{auto_hours:.1f}</span>'
            f'<span style="color:{MUTED}"> h of class · {auto_start}–{auto_end}</span>',
            unsafe_allow_html=True,
        )
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
    st.html(bottles_svg(total))
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
    minutes = st.slider(
        "Minutes active", 0, 180, int(prior.get("exercise_minutes", 0) or 0), 5,
        label_visibility="collapsed",
    )
    st.caption("Walking to class counts. Leave it at zero if it was a still day.")

    st.divider()
    if st.button("Save today", type="primary", use_container_width=True):
        save_entry(
            {
                "date": day,
                "sleep_hours": hours,
                "sleep_start": bed.strftime("%H:%M"),
                "exercise_minutes": minutes,
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


if __name__ == "__main__":
    st.set_page_config(page_title="Daily log", layout="centered", menu_items={})
    st.title("Today")
    render_daily_log()
