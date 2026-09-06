"""Single entry point for the wellbeing app.

Run:  streamlit run src/app.py

Everything lives behind one navigation sidebar, so the timetable, the daily
log and the trends are all reachable without restarting anything.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Wellbeing", layout="wide", menu_items={})


def today_page() -> None:
    from daily_log import render_daily_log

    st.title("Today")
    render_daily_log()


def timetable_page() -> None:
    from timetable_input import render_timetable

    st.title("Your timetable")
    st.caption("Set this up once. The daily log will fill class hours in for you.")
    render_timetable()


def trends_page() -> None:
    from health_dashboard import render_dashboard

    render_dashboard()


pages = [
    st.Page(today_page, title="Today", icon=":material/today:", default=True),
    st.Page(timetable_page, title="Timetable", icon=":material/calendar_month:"),
    st.Page(trends_page, title="Trends", icon=":material/insights:"),
]

st.navigation(pages).run()
