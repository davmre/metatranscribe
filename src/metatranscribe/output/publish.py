from __future__ import annotations

import re
import shutil
import unicodedata
from datetime import date
from pathlib import Path

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_RECORDER_PATTERN = re.compile(r"^(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})\s+at\s+", re.IGNORECASE)


def infer_date_from_filename(filename: str, today: date) -> date | None:
    stem = Path(filename).stem
    m = _RECORDER_PATTERN.match(stem)
    if not m:
        return None
    month = _MONTH_NAMES.get(m.group("month").lower())
    if month is None:
        return None
    day = int(m.group("day"))
    for year in (today.year, today.year - 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate <= today:
            return candidate
    return None


def sanitize_suggested_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    lowered = normalized.strip().lower()
    collapsed = re.sub(r"[^a-z0-9]+", "_", lowered)
    clean = collapsed.strip("_")
    return clean or "untitled"


def build_publish_filename(current_date: date, suggested_name: str) -> str:
    return f"{current_date.isoformat()}_{sanitize_suggested_name(suggested_name)}.md"


def resolve_collision(destination_dir: Path, filename: str) -> Path:
    candidate = destination_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        with_suffix = destination_dir / f"{stem}_{index}{suffix}"
        if not with_suffix.exists():
            return with_suffix
        index += 1


def copy_published_markdown(source_markdown: Path, destination_dir: Path, filename: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = resolve_collision(destination_dir, filename)
    shutil.copy2(source_markdown, destination)
    return destination
