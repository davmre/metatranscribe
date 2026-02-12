from __future__ import annotations

import re
import shutil
import unicodedata
from datetime import date
from pathlib import Path


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
