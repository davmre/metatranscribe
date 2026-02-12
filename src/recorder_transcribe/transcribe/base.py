from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from recorder_transcribe.models import ProviderTranscript


class TranscriptionProvider(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio_id: str, audio_path: Path, language_hint: str) -> ProviderTranscript:
        raise NotImplementedError
