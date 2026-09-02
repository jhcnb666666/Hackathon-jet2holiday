"""Weekly timetable setup.

Preview on its own:  streamlit run src/timetable_input.py

A timetable is a one-time setup, not a daily chore. Once it is in, the app
knows what classes any given date holds, so the daily log never has to ask.

Classes are stored as blocks rather than ticked cells because a block can
carry the things a cell cannot: a course code, and the weeks it runs in
(NTU labs are often marked Wk9,12 or Wk2-13).
"""

from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from academic_calendar import SCHOOLS, academic_context, parse_weeks

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
FIRST_HOUR = 8.0
LAST_HOUR = 22.0
STEP = 0.5
N_SLOTS = int((LAST_HOUR - FIRST_HOUR) / STEP)

SAVE_PATH = Path("data/timetable.csv")

CLASS = "#E8A33D"
INK = "#16202A"
MUTED = "#6B7A87"
TRACK = "#F1F4F6"


def _clock(hour: float) -> str:
    h, m = int(hour), int(round((hour % 1) * 60))
    return f"{h:02d}:{m:02d}"


def _hours(hhmm: str) -> float:
    h, m = str(hhmm).split(":")
    return int(h) + int(m) / 60


TIME_OPTIONS = [_clock(FIRST_HOUR + i * STEP) for i in range(N_SLOTS + 1)]

COLUMNS = ["day", "start", "end", "name", "weeks"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _seed() -> pd.DataFrame:
    """A couple of rows so the grid is not blank on first open."""
    return pd.DataFrame(
        [
            {"day": "Mon", "start": "09:30", "end": "12:20", "name": "E2104L LAB", "weeks": "9,12"},
            {"day": "Mon", "start": "13:30", "end": "15:20", "name": "IE2106 LEC", "weeks": ""},
            {"day": "Wed", "start": "10:30", "end": "12:20", "name": "CL0105 LEC", "weeks": ""},
        ],
        columns=COLUMNS,
    )


def classes_on(blocks: pd.DataFrame, day: date, school: str) -> pd.DataFrame:
    """Blocks that actually run on a given date, after the calendar has its say."""
    ctx = academic_context(day, school)
    if not ctx.classes_expected or day.weekday() > 5:
        return _empty()

    name = DAYS[day.weekday()]
    today = blocks[blocks["day"] == name].copy()
    keep = []
    for _, r in today.iterrows():
        weeks = parse_weeks(str(r.get("weeks", "")))
        keep.append(weeks is None or ctx.week in weeks)
    return today[pd.Series(keep, index=today.index)] if keep else _empty()


def day_summary(blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in DAYS:
        sub = blocks[blocks["day"] == d]
        if sub.empty:
            rows.append({"day": d, "first": "", "last": "", "hours": 0.0})
            continue
        starts = [_hours(v) for v in sub["start"]]
        ends = [_hours(v) for v in sub["end"]]
        rows.append(
            {
                "day": d,
                "first": _clock(min(starts)),
                "last": _clock(max(ends)),
                "hours": round(sum(e - s for s, e in zip(starts, ends)), 1),
            }
        )
    return pd.DataFrame(rows)


IMPORT_PROMPT = """You are reading a university timetable screenshot.

Return ONLY a JSON array. No prose, no markdown fences.

Each element must be:
{"day": "Mon"|"Tue"|"Wed"|"Thu"|"Fri"|"Sat", "start": "HH:MM", "end": "HH:MM",
 "name": "<course code and type, e.g. EE2103 TUT>", "weeks": ""}

Rules:
- Times often appear as 0930to1050. Convert those to "09:30" and "10:50".
- A cell marked -Wk9,12 or -Wk2-13 means the class only runs in those teaching
  weeks. Put "9,12" or "2-13" in "weeks". Otherwise "weeks" must be "".
- The same course at the same day and time repeated with different venue codes
  is ONE class. Emit it once.
- Ignore venue and room codes, and ignore empty cells.
- Do not invent classes. If a cell is unreadable, leave it out."""


def _as_text(content) -> str:
    """LangChain may hand back a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return "".join(parts)
    return str(content)


def _extract_json(text) -> list[dict]:
    text = _as_text(text).strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text.strip())


def read_timetable_image(data: bytes, mime: str) -> list[dict]:
    """Ask Gemini to turn a timetable screenshot into class blocks."""
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    from core.settings import settings

    key = settings.GOOGLE_API_KEY
    if key is None:
        raise RuntimeError("GOOGLE_API_KEY is not set in .env")

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=key.get_secret_value(),
        temperature=0,
    )
    b64 = base64.b64encode(data).decode()
    message = HumanMessage(
        content=[
            {"type": "text", "text": IMPORT_PROMPT},
            {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
        ]
    )
    rows = _extract_json(model.invoke([message]).content)

    cleaned = []
    for r in rows:
        if r.get("day") in DAYS and r.get("start") and r.get("end"):
            cleaned.append(
                {
                    "day": r["day"],
                    "start": str(r["start"])[:5],
                    "end": str(r["end"])[:5],
                    "name": str(r.get("name", "") or ""),
                    "weeks": str(r.get("weeks", "") or ""),
                }
            )
    return cleaned


GRID_CSS = f"""
<style>
.tt {{ display: grid; grid-template-columns: 3.2rem repeat({len(DAYS)}, 1fr); gap: .3rem; margin-top: .5rem; }}
.tt-head {{ font-size: .78rem; color: {MUTED}; text-align: center; padding-bottom: .2rem; }}
.tt-axis {{ position: relative; height: 420px; }}
.tt-tick {{ position: absolute; right: .4rem; font-size: .68rem; color: {MUTED}; transform: translateY(-50%); }}
.tt-col {{ position: relative; height: 420px; background: {TRACK}; border-radius: 3px; }}
.tt-blk {{
  position: absolute; left: 2px; right: 2px; background: {CLASS};
  border-radius: 3px; padding: 2px 4px; overflow: hidden;
  font-size: .64rem; line-height: 1.15; color: #4A3105;
}}
.tt-time {{ opacity: .75; }}
</style>
"""


def render_grid(blocks: pd.DataFrame) -> None:
    span = LAST_HOUR - FIRST_HOUR
    cells = ['<div class="tt-head"></div>']
    cells += [f'<div class="tt-head">{d}</div>' for d in DAYS]

    ticks = "".join(
        f'<div class="tt-tick" style="top:{(h - FIRST_HOUR) / span * 100:.2f}%">{int(h):02d}:00</div>'
        for h in range(int(FIRST_HOUR), int(LAST_HOUR) + 1, 2)
    )
    cells.append(f'<div class="tt-axis">{ticks}</div>')

    for d in DAYS:
        inner = ""
        for _, r in blocks[blocks["day"] == d].iterrows():
            try:
                top = (_hours(r["start"]) - FIRST_HOUR) / span * 100
                height = (_hours(r["end"]) - _hours(r["start"])) / span * 100
            except (ValueError, AttributeError):
                continue
            weeks = str(r.get("weeks", "") or "")
            tag = f" · Wk{weeks}" if weeks else ""
            inner += (
                f'<div class="tt-blk" style="top:{top:.2f}%;height:{height:.2f}%">'
                f'{r.get("name", "") or "Class"}'
                f'<div class="tt-time">{r["start"]}–{r["end"]}{tag}</div></div>'
            )
        cells.append(f'<div class="tt-col">{inner}</div>')

    st.html(GRID_CSS + f'<div class="tt">{"".join(cells)}</div>')


def render_timetable() -> pd.DataFrame:
    if "blocks" not in st.session_state:
        st.session_state.blocks = _seed()
    if "blocks_version" not in st.session_state:
        st.session_state.blocks_version = 0

    school = st.radio("School", SCHOOLS, horizontal=True)
    today = date.today()
    ctx = academic_context(today, school)

    st.caption(f"{today:%A %d %b %Y} · {ctx.label}" + (f" · {ctx.holiday}" if ctx.holiday else ""))

    st.markdown("#### Import from a screenshot")
    up = st.file_uploader(
        "Timetable screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )
    c_read, c_mode = st.columns([1, 2])
    replace = c_mode.checkbox("Replace what is already there", value=True)
    if c_read.button("Read timetable", disabled=up is None, use_container_width=True):
        with st.spinner("Reading the image…"):
            try:
                found = read_timetable_image(up.getvalue(), up.type)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user below
                st.error(f"Could not read that image: {exc}")
                found = []
        if found:
            new = pd.DataFrame(found, columns=COLUMNS)
            st.session_state.blocks = (
                new if replace else pd.concat([st.session_state.blocks, new], ignore_index=True)
            )
            st.session_state.blocks_version += 1
            st.success(f"Found {len(found)} classes. Check them below before saving.")
            st.rerun()
        elif up is not None:
            st.warning("Nothing was picked up. Try a sharper crop, or add the classes by hand.")

    st.markdown("#### Add a class")
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1.6, 1, 0.8])
    day = c1.selectbox("Day", DAYS, label_visibility="collapsed")
    start = c2.selectbox("From", TIME_OPTIONS, index=3, label_visibility="collapsed")
    end = c3.selectbox("To", TIME_OPTIONS, index=9, label_visibility="collapsed")
    name = c4.text_input("Name", placeholder="Course code", label_visibility="collapsed")
    weeks = c5.text_input("Weeks", placeholder="Weeks e.g. 9,12", label_visibility="collapsed")
    if c6.button("Add", use_container_width=True, type="primary"):
        if TIME_OPTIONS.index(end) <= TIME_OPTIONS.index(start):
            st.warning("End time must come after the start time.")
        else:
            new = pd.DataFrame(
                [{"day": day, "start": start, "end": end, "name": name, "weeks": weeks}],
                columns=COLUMNS,
            )
            st.session_state.blocks = pd.concat([st.session_state.blocks, new], ignore_index=True)
            st.session_state.blocks_version += 1
            st.rerun()

    st.markdown("#### Your week")
    render_grid(st.session_state.blocks)

    with st.expander("Edit rows", expanded=False):
        edited = st.data_editor(
            st.session_state.blocks,
            column_config={
                "day": st.column_config.SelectboxColumn("Day", options=DAYS, width="small"),
                "start": st.column_config.SelectboxColumn("From", options=TIME_OPTIONS, width="small"),
                "end": st.column_config.SelectboxColumn("To", options=TIME_OPTIONS, width="small"),
                "name": st.column_config.TextColumn("Course"),
                "weeks": st.column_config.TextColumn("Weeks", help="Blank means every week"),
            },
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key=f"editor_{st.session_state.blocks_version}",
        )
        st.session_state.blocks = edited

    blocks = st.session_state.blocks

    st.markdown("#### On today")
    running = classes_on(blocks, today, school)
    if not ctx.classes_expected:
        st.info(f"No classes during {ctx.label.lower()}.")
    elif running.empty:
        st.info("Nothing scheduled today.")
    else:
        hours = sum(_hours(r["end"]) - _hours(r["start"]) for _, r in running.iterrows())
        for _, r in running.iterrows():
            st.write(f"**{r['start']}–{r['end']}**  {r['name'] or 'Class'}")
        st.caption(f"{hours:.1f} hours in class today.")

    skipped = len(blocks[blocks["day"] == DAYS[today.weekday()]]) - len(running) if today.weekday() < 6 else 0
    if ctx.classes_expected and skipped > 0:
        st.caption(f"{skipped} class(es) on this weekday do not run in week {ctx.week}.")

    st.markdown("#### Weekly total")
    st.dataframe(day_summary(blocks), hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    if left.button("Save timetable", use_container_width=True):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        out = blocks.copy()
        out["school"] = school
        out.to_csv(SAVE_PATH, index=False)
        st.success(f"Saved to {SAVE_PATH}")
    if right.button("Clear all", use_container_width=True):
        st.session_state.blocks = _empty()
        st.session_state.blocks_version += 1
        st.rerun()

    return blocks


if __name__ == "__main__":
    st.set_page_config(page_title="Timetable", layout="wide", menu_items={})
    st.title("Your timetable")
    st.caption("Set this up once. The daily log will fill class hours in for you.")
    render_timetable()
