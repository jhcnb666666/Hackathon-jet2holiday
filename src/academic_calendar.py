"""Academic calendar lookup for NTU and NUS, AY2026-27.

Dates are transcribed from each university's published calendar. They are
hard-coded on purpose: the calendar is a fixed table, so reading it from a
PDF or an image at runtime would turn something certain into something
uncertain.

    academic_context(date(2026, 9, 2), "NTU")
    -> AcademicContext(week=4, phase="teaching", label="Teaching Week 4", ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

SCHOOLS = ["NTU", "NUS"]

Segment = tuple[date, date, str, str]  # start, end, phase, label


def _weeks(first: date, numbers: range | list[int], offset: int = 0) -> list[Segment]:
    out = []
    for i, n in enumerate(numbers):
        start = first + timedelta(days=7 * (i + offset))
        out.append((start, start + timedelta(days=6), "teaching", f"Teaching Week {n}"))
    return out


def _span(start: date, end: date, phase: str, label: str) -> Segment:
    return (start, end, phase, label)


D = date

NTU_SEGMENTS: list[Segment] = [
    _span(D(2026, 8, 3), D(2026, 8, 8), "orientation", "Orientation"),
    *_weeks(D(2026, 8, 9), range(1, 8)),
    _span(D(2026, 9, 27), D(2026, 10, 3), "recess", "Recess Week"),
    *_weeks(D(2026, 10, 4), range(8, 14)),
    _span(D(2026, 11, 15), D(2026, 11, 21), "revision", "Revision Week"),
    _span(D(2026, 11, 22), D(2026, 12, 5), "exams", "Examinations"),
    _span(D(2026, 12, 6), D(2027, 1, 9), "vacation", "Vacation"),
    *_weeks(D(2027, 1, 10), range(1, 8)),
    _span(D(2027, 2, 28), D(2027, 3, 6), "recess", "Recess Week"),
    *_weeks(D(2027, 3, 7), range(8, 14)),
    _span(D(2027, 4, 18), D(2027, 4, 24), "revision", "Revision Week"),
    _span(D(2027, 4, 25), D(2027, 5, 9), "exams", "Examinations"),
]

NUS_SEGMENTS: list[Segment] = [
    _span(D(2026, 8, 3), D(2026, 8, 8), "orientation", "Orientation"),
    *_weeks(D(2026, 8, 10), range(1, 7)),
    _span(D(2026, 9, 19), D(2026, 9, 27), "recess", "Recess Week"),
    *_weeks(D(2026, 9, 28), [7]),
    *_weeks(D(2026, 10, 5), range(8, 14)),
    _span(D(2026, 11, 14), D(2026, 11, 20), "revision", "Reading Week"),
    _span(D(2026, 11, 21), D(2026, 12, 5), "exams", "Examinations"),
    _span(D(2026, 12, 6), D(2027, 1, 10), "vacation", "Vacation"),
    *_weeks(D(2027, 1, 11), range(1, 7)),
    _span(D(2027, 2, 20), D(2027, 2, 28), "recess", "Recess Week"),
    *_weeks(D(2027, 3, 1), [7]),
    *_weeks(D(2027, 3, 8), range(8, 14)),
    _span(D(2027, 4, 17), D(2027, 4, 23), "revision", "Reading Week"),
    _span(D(2027, 4, 24), D(2027, 5, 8), "exams", "Examinations"),
]

SEGMENTS = {"NTU": NTU_SEGMENTS, "NUS": NUS_SEGMENTS}

HOLIDAYS = {
    D(2026, 8, 10): "National Day (observed)",
    D(2026, 11, 9): "Deepavali (observed)",
    D(2026, 12, 25): "Christmas Day",
    D(2027, 1, 1): "New Year's Day",
    D(2027, 2, 8): "Chinese New Year (observed)",
    D(2027, 3, 10): "Hari Raya Puasa",
    D(2027, 3, 26): "Good Friday",
    D(2027, 5, 17): "Hari Raya Haji",
    D(2027, 5, 20): "Vesak Day",
}


@dataclass(frozen=True)
class AcademicContext:
    week: int | None
    phase: str
    label: str
    holiday: str | None

    @property
    def classes_expected(self) -> bool:
        return self.phase == "teaching" and self.holiday is None


def academic_context(day: date, school: str = "NTU") -> AcademicContext:
    holiday = HOLIDAYS.get(day)
    for start, end, phase, label in SEGMENTS.get(school, NTU_SEGMENTS):
        if start <= day <= end:
            week = int(label.split()[-1]) if phase == "teaching" else None
            return AcademicContext(week, phase, label, holiday)
    return AcademicContext(None, "break", "Outside term", holiday)


def parse_weeks(text: str) -> set[int] | None:
    """'9,12' or '2-13' or blank. None means every teaching week."""
    text = (text or "").strip()
    if not text:
        return None
    weeks: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if a.isdigit() and b.isdigit():
                weeks.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            weeks.add(int(part))
    return weeks or None
