from __future__ import annotations

import logging
import time
from pathlib import Path

import requests
from requests import HTTPError

from recorder_transcribe.models import ProviderTranscript, Segment
from recorder_transcribe.preprocess.audio_prep import guess_audio_content_type
from recorder_transcribe.transcribe.base import TranscriptionProvider


class DeepgramTranscriptionProvider(TranscriptionProvider):
    name = "deepgram"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def transcribe(self, audio_id: str, audio_path: Path, language_hint: str) -> ProviderTranscript:
        url = "https://api.deepgram.com/v1/listen"
        logger = logging.getLogger(__name__)
        params = {
            "model": self.model,
            "language": language_hint,
            "smart_format": "true",
            "utterances": "true",
            "punctuate": "true",
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": guess_audio_content_type(audio_path),
        }
        data = audio_path.read_bytes()
        logger.info(
            "Deepgram transcription request started audio_id=%s model=%s file=%s",
            audio_id,
            self.model,
            audio_path.name,
        )
        started = time.perf_counter()
        response = requests.post(url, params=params, headers=headers, data=data, timeout=600)
        elapsed = time.perf_counter() - started
        try:
            response.raise_for_status()
        except HTTPError as exc:
            detail = response.text.strip()
            raise HTTPError(f"{exc}. Response body: {detail}", response=response) from exc
        logger.info(
            "Deepgram transcription request finished audio_id=%s model=%s status=%s elapsed_sec=%.2f",
            audio_id,
            self.model,
            response.status_code,
            elapsed,
        )
        payload = response.json()

        channel = payload.get("results", {}).get("channels", [{}])[0]
        alternatives = channel.get("alternatives", [{}])[0]
        transcript_text = str(alternatives.get("transcript", "")).strip()

        segments: list[Segment] = []
        for utt in payload.get("results", {}).get("utterances", []):
            segments.append(
                Segment(
                    start_sec=float(utt.get("start", 0.0)),
                    end_sec=float(utt.get("end", 0.0)),
                    text=str(utt.get("transcript", "")).strip(),
                    confidence=float(utt.get("confidence", 0.0)) if utt.get("confidence") is not None else None,
                    speaker=str(utt.get("speaker")) if utt.get("speaker") is not None else None,
                )
            )

        duration = 0.0
        if segments:
            duration = max(seg.end_sec for seg in segments)

        return ProviderTranscript(
            provider_name=self.name,
            audio_id=audio_id,
            language=language_hint,
            duration_sec=duration,
            segments=segments,
            raw_text=transcript_text,
            confidence_summary={
                "confidence": float(alternatives.get("confidence", 0.0) or 0.0),
            },
            raw_payload_path="",
        )
