from recorder_transcribe.models import CanonicalSegment, CanonicalTranscript, ProviderTranscript, Segment
from recorder_transcribe.reconcile.chunking import (
    build_windows,
    merge_chunk_canonicals,
    slice_transcripts_for_window,
)


def test_build_windows_with_overlap() -> None:
    windows = build_windows(duration_sec=620, chunk_seconds=300, overlap_seconds=20)
    assert len(windows) == 3
    assert windows[0].start_sec == 0
    assert windows[1].start_sec == 280


def test_slice_transcripts_for_window_filters_timed_segments() -> None:
    t = ProviderTranscript(
        provider_name="deepgram",
        audio_id="a",
        language="en",
        duration_sec=100,
        segments=[
            Segment(start_sec=0, end_sec=5, text="a"),
            Segment(start_sec=30, end_sec=35, text="b"),
        ],
        raw_text="a b",
        confidence_summary=None,
        raw_payload_path="",
    )
    out = slice_transcripts_for_window([t], windows := type("W", (), {"start_sec": 20, "end_sec": 40})())
    assert len(out) == 1
    assert len(out[0].segments) == 1
    assert out[0].segments[0].text == "b"


def test_merge_chunk_canonicals_dedupes_overlap() -> None:
    c1 = CanonicalTranscript(
        audio_id="a",
        title="t",
        language="en",
        duration_sec=60,
        segments=[CanonicalSegment(start_sec=0, end_sec=5, text="hello", evidence_refs=["x"])],
        silence_markers=[],
        final_text="hello",
        provenance={},
    )
    c2 = CanonicalTranscript(
        audio_id="a",
        title="t",
        language="en",
        duration_sec=60,
        segments=[
            CanonicalSegment(start_sec=0, end_sec=5, text="hello", evidence_refs=["x"]),
            CanonicalSegment(start_sec=6, end_sec=10, text="world", evidence_refs=["x"]),
        ],
        silence_markers=[],
        final_text="hello world",
        provenance={},
    )

    merged = merge_chunk_canonicals("a", [c1, c2], 60)
    assert len(merged.segments) == 2
    assert merged.segments[1].text == "world"
