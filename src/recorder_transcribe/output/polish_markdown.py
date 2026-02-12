from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests
from requests import HTTPError

from recorder_transcribe.models import CanonicalTranscript

logger = logging.getLogger(__name__)


def render_polished_markdown(
    canonical: CanonicalTranscript,
    provider: str,
    model: str,
    api_key: str,
    long_silence_seconds: int,
    artifacts_dir: Path | None = None,
) -> str:
    prompt = _build_polish_prompt(canonical, long_silence_seconds)
    _write_artifact(artifacts_dir, "request_prompt.json", prompt)

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
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are an expert transcript editor."},
            {"role": "user", "content": prompt},
        ],
    }
    logger.info("OpenAI polish request started model=%s", model)
    started = time.perf_counter()
    response = requests.post(url, headers=headers, json=body, timeout=600)
    elapsed = time.perf_counter() - started
    _raise_for_status_with_body(response)
    logger.info("OpenAI polish request finished model=%s status=%s elapsed_sec=%.2f", model, response.status_code, elapsed)
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def _call_anthropic(model: str, api_key: str, prompt: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "temperature": 0.2,
        "system": "You are an expert transcript editor. Return markdown only.",
        "messages": [{"role": "user", "content": prompt}],
    }
    logger.info("Anthropic polish request started model=%s", model)
    started = time.perf_counter()
    response = requests.post(url, headers=headers, json=body, timeout=600)
    elapsed = time.perf_counter() - started
    _raise_for_status_with_body(response)
    logger.info("Anthropic polish request finished model=%s status=%s elapsed_sec=%.2f", model, response.status_code, elapsed)
    payload = response.json()
    parts = payload.get("content", [])
    texts = [part.get("text", "") for part in parts if part.get("type") == "text"]
    return "\n".join(texts).strip()


def _strip_markdown_fences(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return raw


def _raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except HTTPError as exc:
        detail = response.text.strip()
        raise HTTPError(f"{exc}. Response body: {detail}", response=response) from exc


def _write_artifact(artifacts_dir: Path | None, filename: str, content: str) -> None:
    if not artifacts_dir:
        return
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / filename).write_text(content, encoding="utf-8")
