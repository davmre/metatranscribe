from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from recorder_transcribe.models import CanonicalSegment, CanonicalTranscript


def _fmt_time(sec: float) -> str:
    sec = int(max(0, round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_markdown(
    canonical: CanonicalTranscript,
    source_file: str,
    providers: list[str],
    reconciler_model: str,
    pipeline_version: str,
    output_style: str = "paragraph",
    paragraph_gap_seconds: int = 20,
    paragraph_max_chars: int = 600,
) -> str:
    frontmatter = {
        "audio_id": canonical.audio_id,
        "source_file": source_file,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": canonical.duration_sec,
        "language": canonical.language,
        "providers": providers,
        "reconciler_model": reconciler_model,
        "pipeline_version": pipeline_version,
    }

    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}: {json.dumps(value)}")
        elif isinstance(value, (float, int)):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f'{key}: "{str(value).replace(chr(34), chr(39))}"')
    lines.extend(["---", "", f"# {canonical.title}", ""])

    silence_by_start = {round(marker.start_sec, 3): marker for marker in canonical.silence_markers}
    ordered_segments = sorted(canonical.segments, key=lambda s: s.start_sec)

    lines.extend(["", "## Transcript", ""])

    if output_style.strip().lower() == "timestamped":
        _render_timestamped(lines, ordered_segments, silence_by_start)
    else:
        _render_paragraphs(
            lines,
            ordered_segments,
            silence_by_start,
            paragraph_gap_seconds=paragraph_gap_seconds,
            paragraph_max_chars=paragraph_max_chars,
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Detailed timing and segment evidence are preserved in the JSON sidecar artifact.",
        ]
    )

    return "\n".join(lines).strip() + "\n"


def _render_timestamped(lines: list[str], ordered_segments: list[CanonicalSegment], silence_by_start: dict[float, object]) -> None:
    for idx, segment in enumerate(ordered_segments):
        if idx > 0:
            prev = ordered_segments[idx - 1]
            key = round(prev.end_sec, 3)
            marker = silence_by_start.get(key)
            if marker:
                lines.append(marker.label)
                lines.append("")

        lines.append(f"[{_fmt_time(segment.start_sec)} - {_fmt_time(segment.end_sec)}] {segment.text}")
        lines.append("")


def _render_paragraphs(
    lines: list[str],
    ordered_segments: list[CanonicalSegment],
    silence_by_start: dict[float, object],
    paragraph_gap_seconds: int,
    paragraph_max_chars: int,
) -> None:
    if not ordered_segments:
        return

    current: list[CanonicalSegment] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_len
        if not current:
            return
        start = current[0].start_sec
        end = current[-1].end_sec
        text = " ".join(seg.text.strip() for seg in current if seg.text.strip()).strip()
        if text:
            lines.append(f"[{_fmt_time(start)} - {_fmt_time(end)}] {text}")
            lines.append("")
        current.clear()
        current_len = 0

    for idx, segment in enumerate(ordered_segments):
        if idx > 0:
            prev = ordered_segments[idx - 1]
            gap = segment.start_sec - prev.end_sec
            marker = silence_by_start.get(round(prev.end_sec, 3))
            if marker:
                flush()
                lines.append(marker.label)
                lines.append("")
            elif gap >= paragraph_gap_seconds:
                flush()

        text_len = len(segment.text.strip())
        if current and current_len + text_len > paragraph_max_chars:
            flush()

        current.append(segment)
        current_len += text_len + 1

    flush()


def write_outputs(canonical: CanonicalTranscript, markdown_path: Path, json_path: Path, markdown_text: str) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    json_path.write_text(json.dumps(canonical.model_dump(), indent=2), encoding="utf-8")
