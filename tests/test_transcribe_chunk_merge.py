from recorder_transcribe.models import ProviderTranscript, Segment
from recorder_transcribe.orchestrator import _merge_provider_chunk_transcripts
from recorder_transcribe.preprocess.audio_prep import AudioChunk


def test_merge_provider_chunks_offsets_timed_segments() -> None:
    chunks = [AudioChunk(path=None, start_sec=0, end_sec=10), AudioChunk(path=None, start_sec=10, end_sec=20)]
    chunk_results = [
        ProviderTranscript(
            provider_name="deepgram",
            audio_id="a__chunk_000",
            language="en",
            duration_sec=10,
            segments=[Segment(start_sec=1, end_sec=2, text="one", confidence=0.8)],
            raw_text="one",
            confidence_summary=None,
            raw_payload_path="",
        ),
        ProviderTranscript(
            provider_name="deepgram",
            audio_id="a__chunk_001",
            language="en",
            duration_sec=10,
            segments=[Segment(start_sec=0.5, end_sec=1.5, text="two", confidence=0.9)],
            raw_text="two",
            confidence_summary=None,
            raw_payload_path="",
        ),
    ]

    merged = _merge_provider_chunk_transcripts("a", "deepgram", chunks, chunk_results, "en", 20)
    assert len(merged.segments) == 2
    assert merged.segments[0].start_sec == 1
    assert merged.segments[1].start_sec == 10.5


def test_merge_provider_chunks_uses_chunk_window_for_untimed_text() -> None:
    chunks = [AudioChunk(path=None, start_sec=30, end_sec=40)]
    chunk_results = [
        ProviderTranscript(
            provider_name="openai",
            audio_id="a__chunk_000",
            language="en",
            duration_sec=10,
            segments=[Segment(start_sec=0, end_sec=0, text="untimed text")],
            raw_text="untimed text",
            confidence_summary=None,
            raw_payload_path="",
        )
    ]

    merged = _merge_provider_chunk_transcripts("a", "openai", chunks, chunk_results, "en", 40)
    assert len(merged.segments) == 1
    assert merged.segments[0].start_sec == 30
    assert merged.segments[0].end_sec == 40
