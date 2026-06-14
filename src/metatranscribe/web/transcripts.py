from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import markdown as markdown_lib

from metatranscribe.config import Settings
from metatranscribe.models import AudioFileRecord
from metatranscribe.state.store import StateStore

logger = logging.getLogger(__name__)

# Human-readable label for each pipeline status, shown in the UI.
STATUS_LABELS = {
    "ingested": "Transcribing",
    "transcribed": "Reconciling",
    "reconciled": "Polishing",
    "exported": "Ready",
    "failed": "Failed",
}

TERMINAL_STATUSES = {"exported", "failed"}


@dataclass
class TranscriptItem:
    audio_id: str
    title: str
    status: str
    status_label: str
    created_at: str
    duration_sec: float | None
    error: str | None

    @property
    def is_ready(self) -> bool:
        return self.status == "exported"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def in_progress(self) -> bool:
        return self.status not in TERMINAL_STATUSES


@dataclass
class TranscriptDetail:
    item: TranscriptItem
    body_html: str | None
    body_markdown: str | None


def _final_json_path(settings: Settings, audio_id: str) -> Path:
    return settings.output_root / "final" / f"{audio_id}.json"


def _final_md_path(settings: Settings, audio_id: str) -> Path:
    return settings.output_root / "final" / f"{audio_id}.md"


def _load_canonical(settings: Settings, audio_id: str) -> dict | None:
    path = _final_json_path(settings, audio_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to read canonical json for audio_id=%s", audio_id, exc_info=True)
        return None


def _derive_title(settings: Settings, record: AudioFileRecord) -> str:
    canonical = _load_canonical(settings, record.audio_id)
    if canonical and canonical.get("title"):
        return str(canonical["title"])
    if record.source_path.strip():
        return Path(record.source_path).name
    return record.audio_id


def _derive_duration(settings: Settings, record: AudioFileRecord) -> float | None:
    canonical = _load_canonical(settings, record.audio_id)
    if canonical and canonical.get("duration_sec") is not None:
        return float(canonical["duration_sec"])
    return None


def _to_item(settings: Settings, record: AudioFileRecord) -> TranscriptItem:
    return TranscriptItem(
        audio_id=record.audio_id,
        title=_derive_title(settings, record),
        status=record.status,
        status_label=STATUS_LABELS.get(record.status, record.status),
        created_at=record.created_at,
        duration_sec=_derive_duration(settings, record),
        error=record.error,
    )


def list_items(settings: Settings, store: StateStore) -> list[TranscriptItem]:
    """Return all transcripts newest-first for the browse/home list."""
    records = store.get_records()  # ORDER BY created_at ASC
    records.reverse()
    return [_to_item(settings, record) for record in records]


def _strip_frontmatter(markdown_text: str) -> str:
    """Remove a leading YAML frontmatter block (--- ... ---) if present."""
    if not markdown_text.startswith("---"):
        return markdown_text
    lines = markdown_text.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1:]).lstrip("\n")
    return markdown_text


def _render_markdown(markdown_text: str) -> str:
    """Render transcript markdown to HTML.

    The source text is HTML-escaped *before* rendering so any literal HTML in a
    transcript (e.g. ``<script>``) is neutralized while standard markdown syntax
    (#, *, -, [](), backticks) is preserved -- Python-Markdown has no built-in
    sanitizer, and this avoids pulling in a separate sanitization dependency.
    """
    body = _strip_frontmatter(markdown_text)
    escaped = html.escape(body)
    return markdown_lib.markdown(escaped, extensions=["extra", "sane_lists", "nl2br"])


def load_detail(settings: Settings, store: StateStore, audio_id: str) -> TranscriptDetail | None:
    record = store.get_record(audio_id)
    if record is None:
        return None
    item = _to_item(settings, record)
    body_html: str | None = None
    body_markdown: str | None = None
    md_path = _final_md_path(settings, audio_id)
    if item.is_ready and md_path.exists():
        try:
            raw = md_path.read_text(encoding="utf-8")
            body_markdown = _strip_frontmatter(raw).strip()
            body_html = _render_markdown(raw)
        except OSError:
            logger.warning("Failed to read final markdown for audio_id=%s", audio_id, exc_info=True)
    return TranscriptDetail(item=item, body_html=body_html, body_markdown=body_markdown)
