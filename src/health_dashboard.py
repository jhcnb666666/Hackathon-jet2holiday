"""Student wellbeing dashboard.

Preview on its own:  streamlit run src/health_dashboard.py
Or import render_dashboard() into streamlit_app.py once the chat side is ready.

Adding a metric = append one Metric to METRICS and add the matching column to
the dataframe. Nothing else needs to change.
"""

from __future__ import annotations
from pathlib import Path

from dataclasses import dataclass
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

INK = "#16202A"
MUTED = "#6B7A87"
TRACK = "#F1F4F6"

SLEEP = "#3D4E9E"
MOVE = "#2F8F6B"
WATER = "#3E9BC4"
CLASS = "#E8A33D"

DAY_START = 18  # rows run 18:00 to 18:00 so a night's sleep stays in one piece


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    color: str
    low: float
    high: float
    decimals: int = 0


METRICS: list[Metric] = [
    Metric("sleep_hours", "Sleep", "h", SLEEP, 7, 9, 1),
    Metric("exercise_minutes", "Movement", "min", MOVE, 30, 120),
    Metric("water_ml", "Water", "ml", WATER, 1800, 3000),
    Metric("class_hours", "Class time", "h", CLASS, 0, 6, 1),
]

# weekday -> class_start, class_end, class_hours, sleep_start, sleep_hours, exercise, water
WEEKDAY_PATTERN = {
    0: ("08:30", "16:30", 5.5, "00:40", 6.3, 0, 1350),
    1: ("10:30", "13:30", 3.0, "01:20", 7.8, 45, 2100),
    2: ("08:30", "17:30", 6.0, "00:50", 6.1, 0, 1250),
    3: ("11:30", "13:30", 2.0, "01:40", 7.9, 35, 2400),
    4: ("13:30", "15:30", 2.0, "01:10", 8.2, 55, 2500),
    5: ("", "", 0.0, "02:20", 9.2, 20, 1900),
    6: ("", "", 0.0, "02:00", 8.8, 0, 1650),
}


def load_data() -> pd.DataFrame:
    """Fake data, used only when no real log exists yet."""
    days = [today - timedelta(days=13 - i) for i in range(14)]

    rows = []
    for i, d in enumerate(days):
        c_start, c_end, c_hours, s_start, s_hours, ex, water = WEEKDAY_PATTERN[d.weekday()]
        drift = 0.3 if i < 7 else -0.2
        rows.append(
            {
                "date": d,
                "sleep_hours": round(s_hours + drift, 1),
                "sleep_start": s_start,
                "exercise_minutes": max(0, ex + (10 if i >= 7 else -5)),
                "water_ml": water + (150 if i >= 7 else -100),
                "class_hours": c_hours,
                "class_start": c_start,
                "class_end": c_end,
            }
        )
    return pd.DataFrame(rows)

LOG_PATH = Path("data/daily_log.csv")


def load_data() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return _placeholder_data()
    df = pd.read_csv(LOG_PATH).fillna("")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _hours(hhmm: str) -> float | None:
    if not hhmm:
        return None
    h, m = hhmm.split(":")
    return int(h) + int(m) / 60


def _offset(hour: float) -> float:
    """Position on the 18:00-anchored track, in hours from the left edge."""
    return (hour - DAY_START) % 24


def _fmt(value: float, m: Metric) -> str:
    return f"{value:,.{m.decimals}f}"


CSS = f"""
<style>
.wk {{ font-variant-numeric: tabular-nums; margin-top: .4rem; }}
.wk-row {{
  display: grid; grid-template-columns: 3rem 1fr 4rem;
  align-items: center; gap: .8rem; padding: .2rem 0;
}}
.wk-day {{ font-size: .8rem; color: {MUTED}; }}
.wk-track {{ position: relative; height: 1.1rem; background: {TRACK}; border-radius: 2px; overflow: hidden; }}
.wk-blk {{ position: absolute; top: 0; bottom: 0; }}
.wk-note {{ font-size: .75rem; color: {MUTED}; text-align: right; }}
.wk-ticks {{
  display: grid; grid-template-columns: 3rem 1fr 4rem;
  gap: .8rem; font-size: .7rem; color: {MUTED}; padding-top: .3rem;
}}
.wk-scale {{ display: flex; justify-content: space-between; }}
.wk-key {{ font-size: .75rem; color: {MUTED}; margin-top: .9rem; }}
.wk-swatch {{ display: inline-block; width: .65rem; height: .65rem; border-radius: 2px; margin-right: .3rem; }}
.mx-label {{ font-size: .78rem; color: {MUTED}; }}
.mx-value {{ font-size: 1.9rem; font-weight: 600; color: {INK}; line-height: 1.15; font-variant-numeric: tabular-nums; }}
.mx-unit {{ font-size: .9rem; font-weight: 400; color: {MUTED}; margin-left: .15rem; }}
.mx-note {{ font-size: .75rem; color: {MUTED}; margin-top: .1rem; }}
.mx-rule {{ height: 3px; border-radius: 2px; margin-bottom: .5rem; }}
</style>
"""


def render_today(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    week = df.tail(7)

    st.markdown("#### Today")
    for col, m in zip(st.columns(len(METRICS)), METRICS):
        value = float(latest[m.key])
        avg = float(week[m.key].mean())
        opacity = "1" if m.low <= value <= m.high else ".25"
        col.html(
            f'<div class="mx-rule" style="background:{m.color};opacity:{opacity}"></div>'
            f'<div class="mx-label">{m.label}</div>'
            f'<div class="mx-value">{_fmt(value, m)}<span class="mx-unit">{m.unit}</span></div>'
            f'<div class="mx-note">7-day average {_fmt(avg, m)} {m.unit}</div>'
        )
    st.caption("A faded bar means the day fell outside its usual range.")


def render_week_rhythm(df: pd.DataFrame) -> None:
    st.markdown("#### When the day is spent")
    st.caption("Each row runs from 6pm to 6pm, so a night's sleep stays in one piece.")

    rows = []
    for _, r in df.tail(7).iterrows():
        blocks = ""
        s_start = _hours(r["sleep_start"])
        if s_start is not None:
            left = _offset(s_start)
            blocks += (
                f'<span class="wk-blk" style="left:{left / 24 * 100:.3f}%;'
                f'width:{float(r["sleep_hours"]) / 24 * 100:.3f}%;background:{SLEEP}"></span>'
            )
        c_start, c_end = _hours(r["class_start"]), _hours(r["class_end"])
        if c_start is not None and c_end is not None:
            blocks += (
                f'<span class="wk-blk" style="left:{_offset(c_start) / 24 * 100:.3f}%;'
                f'width:{(c_end - c_start) / 24 * 100:.3f}%;background:{CLASS}"></span>'
            )
        rows.append(
            f'<div class="wk-row"><div class="wk-day">{r["date"]:%a}</div>'
            f'<div class="wk-track">{blocks}</div>'
            f'<div class="wk-note">{r["sleep_hours"]:.1f} h</div></div>'
        )

    ticks = "".join(f"<span>{(DAY_START + h) % 24:02d}</span>" for h in (0, 6, 12, 18, 24))
    st.html(
        CSS
        + f'<div class="wk">{"".join(rows)}'
        + f'<div class="wk-ticks"><span></span><div class="wk-scale">{ticks}</div><span></span></div>'
        + '<div class="wk-key">'
        + f'<span class="wk-swatch" style="background:{SLEEP}"></span>Sleep'
        + f'<span class="wk-swatch" style="background:{CLASS};margin-left:1rem"></span>Class'
        + "</div></div>"
    )


def render_trend(df: pd.DataFrame) -> None:
    st.markdown("#### Trend")
    label = st.selectbox("Metric", [m.label for m in METRICS], label_visibility="collapsed")
    m = next(x for x in METRICS if x.label == label)

    data = df[["date", m.key]].rename(columns={m.key: "value"})
    data["day"] = pd.to_datetime(data["date"]).dt.strftime("%a %d/%m")

    band = (
        alt.Chart(pd.DataFrame({"low": [m.low], "high": [m.high]}))
        .mark_rect(color=m.color, opacity=0.07)
        .encode(y="low:Q", y2="high:Q")
    )
    line = (
        alt.Chart(data)
        .mark_line(
            color=m.color,
            point={"filled": True, "size": 45, "color": m.color},
            strokeWidth=2,
        )
        .encode(
            x=alt.X("day:O", title=None, sort=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("value:Q", title=f"{m.label} ({m.unit})"),
            tooltip=[alt.Tooltip("day:O", title="Day"), alt.Tooltip("value:Q", title=m.label)],
        )
    )
    st.altair_chart(band + line, use_container_width=True)
    st.caption(f"Shaded band shows the usual {_fmt(m.low, m)}–{_fmt(m.high, m)} {m.unit} range.")


def render_dashboard(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = load_data()
    render_today(df)
    st.divider()
    render_week_rhythm(df)
    st.divider()
    render_trend(df)


if __name__ == "__main__":
    st.set_page_config(page_title="Wellbeing dashboard", layout="wide", menu_items={})
    st.title("Your two weeks")
    st.caption("Placeholder data — swap load_data() when the real dataset arrives.")
    render_dashboard()
