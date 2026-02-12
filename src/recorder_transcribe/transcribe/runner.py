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


def _chunk_provider_path(artifacts_root: Path, chunk_index: int, provider_name: str) -> Path:
    return artifacts_root / "transcribe" / "chunks" / f"{chunk_index:03d}" / "providers" / f"{provider_name}.json"


def save_provider_chunk_transcript(transcript: ProviderTranscript, artifacts_root: Path, chunk_index: int) -> Path:
    path = _chunk_provider_path(artifacts_root, chunk_index, transcript.provider_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    persisted = transcript.model_copy(update={"raw_payload_path": str(path)})
    path.write_text(json.dumps(persisted.model_dump(), indent=2), encoding="utf-8")
    return path


def load_provider_chunk_transcripts(artifacts_root: Path, chunk_index: int) -> list[ProviderTranscript]:
    transcripts: list[ProviderTranscript] = []
    providers_dir = artifacts_root / "transcribe" / "chunks" / f"{chunk_index:03d}" / "providers"
    for path in sorted(providers_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        transcripts.append(ProviderTranscript.model_validate(payload))
    return transcripts


def discover_chunk_providers(artifacts_root: Path) -> list[str]:
    provider_names: set[str] = set()
    chunks_root = artifacts_root / "transcribe" / "chunks"
    for path in chunks_root.glob("*/providers/*.json"):
        provider_names.add(path.stem)
    return sorted(provider_names)
