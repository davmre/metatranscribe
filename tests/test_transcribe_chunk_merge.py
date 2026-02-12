from pathlib import Path

from recorder_transcribe.models import ProviderTranscript, Segment
from recorder_transcribe.orchestrator import _with_global_segment_offsets
from recorder_transcribe.preprocess.audio_prep import AudioChunk
from recorder_transcribe.transcribe.runner import (
    discover_chunk_providers,
    load_provider_chunk_transcripts,
    save_provider_chunk_transcript,
)


def test_with_global_segment_offsets_offsets_timed_segments() -> None:
    transcript = ProviderTranscript(
        provider_name="deepgram",
        audio_id="a__chunk_001",
        language="en",
        duration_sec=10,
        segments=[Segment(start_sec=0.5, end_sec=1.5, text="two", confidence=0.9)],
        raw_text="two",
        confidence_summary=None,
        raw_payload_path="",
    )
    chunk = AudioChunk(path=Path("chunk.wav"), start_sec=10, end_sec=20)

    shifted = _with_global_segment_offsets("a", transcript, chunk)
    assert shifted.audio_id == "a"
    assert shifted.segments[0].start_sec == 10.5
    assert shifted.segments[0].end_sec == 11.5


def test_chunk_provider_artifact_round_trip(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "a"
    transcript = ProviderTranscript(
        provider_name="openai",
        audio_id="a",
        language="en",
        duration_sec=20.0,
        segments=[Segment(start_sec=0.0, end_sec=1.0, text="hello")],
        raw_text="hello",
        confidence_summary=None,
        raw_payload_path="",
    )

    path = save_provider_chunk_transcript(transcript, artifacts_root, 2)
    loaded = load_provider_chunk_transcripts(artifacts_root, 2)
    providers = discover_chunk_providers(artifacts_root)

    assert path.name == "openai.json"
    assert len(loaded) == 1
    assert loaded[0].provider_name == "openai"
    assert loaded[0].raw_payload_path == str(path)
    assert providers == ["openai"]
