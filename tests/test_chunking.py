from recorder_transcribe.models import CanonicalSegment, CanonicalTranscript
from recorder_transcribe.reconcile.chunking import merge_chunk_canonicals


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
