from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests
from requests import HTTPError

from metatranscribe.models import ProviderTranscript, Segment
from metatranscribe.preprocess.audio_prep import guess_audio_content_type
from metatranscribe.transcribe.base import TranscriptionProvider


class OpenAITranscriptionProvider(TranscriptionProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def transcribe(self, audio_id: str, audio_path: Path, language_hint: str) -> ProviderTranscript:
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        logger = logging.getLogger(__name__)

        with audio_path.open("rb") as f:
            files = {"file": (audio_path.name, f, guess_audio_content_type(audio_path))}
            data = _build_form_data(self.model, language_hint)
            logger.info(
                "OpenAI transcription request started audio_id=%s model=%s file=%s",
                audio_id,
                self.model,
                audio_path.name,
            )
            started = time.perf_counter()
            response = requests.post(url, headers=headers, data=data, files=files, timeout=600)
            elapsed = time.perf_counter() - started

        try:
            response.raise_for_status()
        except HTTPError as exc:
            detail = response.text.strip()
            raise HTTPError(f"{exc}. Response body: {detail}", response=response) from exc
        logger.info(
            "OpenAI transcription request finished audio_id=%s model=%s status=%s elapsed_sec=%.2f",
            audio_id,
            self.model,
            response.status_code,
            elapsed,
        )
        payload = response.json()
        raw_text = str(payload.get("text", "")).strip()

        segments = [
            Segment(
                start_sec=float(seg.get("start", 0.0)),
                end_sec=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
                confidence=float(seg.get("avg_logprob", 0.0)) if seg.get("avg_logprob") is not None else None,
            )
            for seg in payload.get("segments", [])
        ]

        if not segments:
            segments = _segments_from_words(payload)

        duration = float(payload.get("duration", 0.0) or 0.0)
        if duration <= 0.0 and segments:
            duration = max(seg.end_sec for seg in segments)
        if not segments and raw_text:
            segments = [Segment(start_sec=0.0, end_sec=duration, text=raw_text, confidence=None, speaker=None)]

        return ProviderTranscript(
            provider_name=self.name,
            audio_id=audio_id,
            language=str(payload.get("language", language_hint)),
            duration_sec=duration,
            segments=segments,
            raw_text=raw_text,
            confidence_summary=_confidence_summary(payload),
            raw_payload_path="",
        )


def _confidence_summary(payload: dict[str, Any]) -> dict[str, float]:
    summary: dict[str, float] = {}
    segments = payload.get("segments", [])
    if not segments:
        return summary

    probs = [seg.get("avg_logprob") for seg in segments if seg.get("avg_logprob") is not None]
    if probs:
        summary["avg_logprob_mean"] = float(sum(probs) / len(probs))
    compression = [seg.get("compression_ratio") for seg in segments if seg.get("compression_ratio") is not None]
    if compression:
        summary["compression_ratio_mean"] = float(sum(compression) / len(compression))
    return summary


def _build_form_data(model: str, language_hint: str) -> dict[str, str]:
    model_name = model.strip()
    data = {
        "model": model_name,
        "language": language_hint,
    }
    # OpenAI docs: timestamp_granularities is only supported by whisper-1.
    if model_name == "whisper-1":
        data["response_format"] = "verbose_json"
        data["timestamp_granularities[]"] = "segment"
    else:
        data["response_format"] = "json"
    return data


def _segments_from_words(payload: dict[str, Any]) -> list[Segment]:
    words = payload.get("words", [])
    if not words:
        return []
    segments: list[Segment] = []
    for word in words:
        text = str(word.get("word", "")).strip()
        if not text:
            continue
        segments.append(
            Segment(
                start_sec=float(word.get("start", 0.0)),
                end_sec=float(word.get("end", 0.0)),
                text=text,
                confidence=float(word.get("probability")) if word.get("probability") is not None else None,
            )
        )
    return segments
