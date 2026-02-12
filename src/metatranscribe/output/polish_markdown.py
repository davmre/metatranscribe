from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from pathlib import Path

from metatranscribe.llm.http_client import (
    extract_anthropic_text,
    extract_openai_message_text,
    post_anthropic_message,
    post_openai_chat_completion,
)
from metatranscribe.models import CanonicalTranscript

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PolishResult:
    markdown: str
    suggested_name: str


def render_polished_markdown(
    canonical: CanonicalTranscript,
    provider: str,
    model: str,
    api_key: str,
    long_silence_seconds: int,
    artifacts_dir: Path | None = None,
    dry_run: bool = False,
) -> PolishResult:
    prompt = _build_polish_prompt(canonical, long_silence_seconds)
    _write_artifact(artifacts_dir, "request_prompt.json", prompt)

    if dry_run:
        logger.info("Polish dry-run: wrote request prompt only audio_id=%s", canonical.audio_id)
        return PolishResult(markdown="", suggested_name="")

    response_text = _call_model(provider, model, api_key, prompt)
    _write_artifact(artifacts_dir, "response_raw.md", response_text)
    parsed = _parse_polish_response(response_text)
    markdown = _normalize_markdown(parsed.get("markdown"), canonical.title)
    suggested_name, suggested_fallback = _normalize_suggested_name(
        parsed.get("filename"), canonical.title, canonical.audio_id
    )
    _write_artifact(
        artifacts_dir,
        "response_parsed.json",
        json.dumps(
            {
                "filename": parsed.get("filename"),
                "suggested_name": suggested_name,
                "suggested_name_used_fallback": suggested_fallback,
                "markdown": markdown,
            },
            indent=2,
        ),
    )
    return PolishResult(markdown=markdown, suggested_name=suggested_name)


def _build_polish_prompt(canonical: CanonicalTranscript, long_silence_seconds: int) -> str:
    timeline = [
        {
            "start_sec": seg.start_sec,
            "end_sec": seg.end_sec,
            "text": seg.text,
        }
        for seg in canonical.segments
    ]
    silence_notes = [
        {
            "start_sec": marker.start_sec,
            "end_sec": marker.end_sec,
            "duration_sec": marker.duration_sec,
            "label": marker.label,
        }
        for marker in canonical.silence_markers
        if marker.duration_sec >= long_silence_seconds
    ]

    instructions = {
        "task": "Produce polished markdown transcript for human reading.",
        "style": [
            "Please use semantic paragraphs and smooth transitions.",
            "Remove filler words and obvious disfluencies when meaning is unchanged.",
            "When the transcript is ambiguous or unclear, try your best to infer the speaker's intent. If this is not possible, you can add a parenthetical note with alternative interpretations, but try to do this rarely.",
            "Do not include timestamps, but please annotate any gaps from silences of more than a minute if this is evident from provided metadata.",
            "Return a JSON object with keys `markdown` and `filename` only.",
            "`markdown` should be the transcript body.",
            "`filename` should be a short suggested filename with no spaces and no extension (e.g., `my_file`, not `my file` or `my_file.md`).",
        ],
        "input": {
            "title": canonical.title,
            "language": canonical.language,
            "duration_sec": canonical.duration_sec,
            "long_silences": silence_notes,
            "timeline_segments": timeline,
        },
    }
    return json.dumps(instructions)


def _call_model(provider: str, model: str, api_key: str, prompt: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return _call_openai(model, api_key, prompt)
    if normalized == "anthropic":
        return _call_anthropic(model, api_key, prompt)
    raise ValueError(f"Unsupported polish provider '{provider}'")


def _call_openai(model: str, api_key: str, prompt: str) -> str:
    logger.info("OpenAI polish request started model=%s", model)
    started = time.perf_counter()
    payload = post_openai_chat_completion(
        api_key=api_key,
        model=model,
        system_prompt="You are an expert transcript editor. Return valid JSON only.",
        user_prompt=prompt,
        temperature=0.2,
    )
    elapsed = time.perf_counter() - started
    logger.info("OpenAI polish request finished model=%s elapsed_sec=%.2f", model, elapsed)
    return extract_openai_message_text(payload)


def _call_anthropic(model: str, api_key: str, prompt: str) -> str:
    logger.info("Anthropic polish request started model=%s", model)
    started = time.perf_counter()
    payload = post_anthropic_message(
        api_key=api_key,
        model=model,
        system_prompt="You are an expert transcript editor. Return valid JSON only.",
        user_prompt=prompt,
        temperature=0.2,
        max_tokens=64000,
    )
    elapsed = time.perf_counter() - started
    logger.info("Anthropic polish request finished model=%s elapsed_sec=%.2f", model, elapsed)
    return extract_anthropic_text(payload)


def _strip_markdown_fences(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return raw


def _parse_polish_response(response_text: str) -> dict[str, object]:
    raw = _strip_markdown_fences(response_text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"markdown": response_text, "filename": ""}
    if not isinstance(payload, dict):
        return {"markdown": response_text, "filename": ""}
    return payload


def _normalize_markdown(markdown_value: object, title: str) -> str:
    markdown = str(markdown_value).strip() if isinstance(markdown_value, str) else ""
    if not markdown:
        markdown = str(markdown_value).strip() if markdown_value is not None else ""
    if not markdown:
        markdown = ""
    if not markdown.startswith("#"):
        markdown = f"# {title}\n\n" + markdown
    return markdown.rstrip() + "\n"


def _normalize_suggested_name(
    suggested_name_value: object, title: str, audio_id: str
) -> tuple[str, bool]:
    if isinstance(suggested_name_value, str) and suggested_name_value.strip():
        return suggested_name_value.strip(), False
    if title.strip():
        return title.strip(), True
    return audio_id, True


def _write_artifact(artifacts_dir: Path | None, filename: str, content: str) -> None:
    if not artifacts_dir:
        return
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / filename).write_text(content, encoding="utf-8")
