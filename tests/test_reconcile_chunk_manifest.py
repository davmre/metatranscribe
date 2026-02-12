import json
from pathlib import Path

import pytest

from recorder_transcribe.models import ProviderTranscript
from recorder_transcribe.orchestrator import (
    _load_transcription_chunk_windows,
    _manifest_duration_sec,
    _validate_chunk_provider_completeness,
)


def test_load_transcription_chunk_windows(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts" / "a"
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = [
        {"index": 0, "start_sec": 0.0, "end_sec": 300.0},
        {"index": 1, "start_sec": 300.0, "end_sec": 600.0},
    ]
    (artifacts / "transcribe_chunks.json").write_text(json.dumps(payload), encoding="utf-8")

    windows = _load_transcription_chunk_windows(artifacts)
    assert len(windows) == 2
    assert windows[0].start_sec == 0.0
    assert windows[1].end_sec == 600.0


def test_manifest_duration_uses_last_window_end(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts" / "a"
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = [
        {"index": 0, "start_sec": 0.0, "end_sec": 300.0},
        {"index": 1, "start_sec": 300.0, "end_sec": 620.0},
    ]
    (artifacts / "transcribe_chunks.json").write_text(json.dumps(payload), encoding="utf-8")

    windows = _load_transcription_chunk_windows(artifacts)
    assert _manifest_duration_sec(windows) == 620.0


def test_validate_chunk_provider_completeness_raises_on_missing_provider() -> None:
    chunk_transcripts = [
        ProviderTranscript(
            provider_name="deepgram",
            audio_id="a",
            language="en",
            duration_sec=100.0,
            segments=[],
            raw_text="",
            confidence_summary=None,
            raw_payload_path="",
        )
    ]
    with pytest.raises(RuntimeError, match="Missing provider transcript artifacts"):
        _validate_chunk_provider_completeness("a", 1, ["deepgram", "openai"], chunk_transcripts)
