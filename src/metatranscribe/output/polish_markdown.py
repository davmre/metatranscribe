from __future__ import annotations

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


def render_polished_markdown(
    canonical: CanonicalTranscript,
    provider: str,
    model: str,
    api_key: str,
    long_silence_seconds: int,
    artifacts_dir: Path | None = None,
    dry_run: bool = False,
) -> str:
    prompt = _build_polish_prompt(canonical, long_silence_seconds)
    _write_artifact(artifacts_dir, "request_prompt.json", prompt)

    if dry_run:
        logger.info("Polish dry-run: wrote request prompt only audio_id=%s", canonical.audio_id)
        return ""

    response_text = _call_model(provider, model, api_key, prompt)
    _write_artifact(artifacts_dir, "response_raw.md", response_text)

    markdown = _strip_markdown_fences(response_text).strip()
    if not markdown.startswith("#"):
        markdown = f"# {canonical.title}\n\n" + markdown
    return markdown.rstrip() + "\n"


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
            "Use semantic paragraphs and smooth transitions.",
            "Remove filler words and obvious disfluencies when meaning is unchanged.",
            "Do not include timestamps, but please annotate any gaps from silences of more than a minute if this is evident from provided metadata.",
            "Return markdown only.",
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
        system_prompt="You are an expert transcript editor.",
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
        system_prompt="You are an expert transcript editor. Return markdown only.",
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


def _write_artifact(artifacts_dir: Path | None, filename: str, content: str) -> None:
    if not artifacts_dir:
        return
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / filename).write_text(content, encoding="utf-8")
