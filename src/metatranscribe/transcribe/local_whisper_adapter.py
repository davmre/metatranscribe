from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from metatranscribe.models import ProviderTranscript, Segment
from metatranscribe.transcribe.base import TranscriptionProvider


class LocalWhisperProvider(TranscriptionProvider):
    name = "local_whisper"

    def __init__(self, model: str = "large-v3") -> None:
        self.model = model

    def transcribe(self, audio_id: str, audio_path: Path, language_hint: str) -> ProviderTranscript:
        whisper_cmd = shutil.which("whisper")
        if not whisper_cmd:
            raise RuntimeError("Local whisper CLI not found in PATH")

        out_dir = audio_path.parent / "whisper_tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                whisper_cmd,
                str(audio_path),
                "--model",
                self.model,
                "--language",
                language_hint,
                "--output_format",
                "json",
                "--output_dir",
                str(out_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        json_path = out_dir / f"{audio_path.stem}.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        segments = [
            Segment(
                start_sec=float(seg.get("start", 0.0)),
                end_sec=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
                confidence=float(seg.get("avg_logprob", 0.0)) if seg.get("avg_logprob") is not None else None,
            )
            for seg in payload.get("segments", [])
        ]

        return ProviderTranscript(
            provider_name=self.name,
            audio_id=audio_id,
            language=str(payload.get("language", language_hint)),
            duration_sec=float(payload.get("duration", 0.0) or 0.0),
            segments=segments,
            raw_text=str(payload.get("text", "")).strip(),
            confidence_summary={},
            raw_payload_path="",
        )
