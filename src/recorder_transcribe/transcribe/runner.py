from __future__ import annotations

import json
from pathlib import Path

from recorder_transcribe.config import Settings
from recorder_transcribe.models import ProviderTranscript
from recorder_transcribe.transcribe.base import TranscriptionProvider
from recorder_transcribe.transcribe.deepgram_adapter import DeepgramTranscriptionProvider
from recorder_transcribe.transcribe.local_whisper_adapter import LocalWhisperProvider
from recorder_transcribe.transcribe.openai_adapter import OpenAITranscriptionProvider


def build_providers(settings: Settings) -> list[TranscriptionProvider]:
    providers: list[TranscriptionProvider] = []
    for name in settings.transcribe_providers:
        if name == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when openai provider is enabled")
            providers.append(OpenAITranscriptionProvider(settings.openai_api_key, settings.transcribe_openai_model))
        elif name == "deepgram":
            if not settings.deepgram_api_key:
                raise ValueError("DEEPGRAM_API_KEY is required when deepgram provider is enabled")
            providers.append(DeepgramTranscriptionProvider(settings.deepgram_api_key, settings.transcribe_deepgram_model))
        elif name == "local_whisper":
            providers.append(LocalWhisperProvider())
        else:
            raise ValueError(f"Unsupported provider '{name}'")

    return providers


def save_provider_transcript(transcript: ProviderTranscript, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{transcript.provider_name}.json"
    path.write_text(json.dumps(transcript.model_dump(), indent=2), encoding="utf-8")
    return path


def load_provider_transcripts(artifacts_dir: Path) -> list[ProviderTranscript]:
    transcripts: list[ProviderTranscript] = []
    for path in sorted(artifacts_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        transcripts.append(ProviderTranscript.model_validate(payload))
    return transcripts
