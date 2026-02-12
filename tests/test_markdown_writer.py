from recorder_transcribe.models import CanonicalSegment, CanonicalTranscript, SilenceMarker
from recorder_transcribe.output.markdown_writer import render_markdown


def _sample_canonical() -> CanonicalTranscript:
    return CanonicalTranscript(
        audio_id="abc123",
        title="Walk Notes",
        language="en",
        duration_sec=240,
        segments=[
            CanonicalSegment(start_sec=0, end_sec=10, text="First thought.", evidence_refs=["openai"]),
            CanonicalSegment(start_sec=11, end_sec=18, text="Second sentence.", evidence_refs=["openai"]),
            CanonicalSegment(start_sec=200, end_sec=210, text="Third thought.", evidence_refs=["deepgram"]),
        ],
        silence_markers=[
            SilenceMarker(start_sec=18, end_sec=200, duration_sec=182, label="[3 minutes of silence]")
        ],
        final_text="First thought. Second sentence. Third thought.",
        provenance={"providers": ["openai", "deepgram"]},
    )


def test_markdown_paragraph_mode_contains_silence_and_compacted_ranges() -> None:
    canonical = _sample_canonical()
    md = render_markdown(
        canonical,
        source_file="original.m4a",
        providers=["openai", "deepgram"],
        reconciler_model="gpt-5",
        pipeline_version="0.1.0",
        output_style="paragraph",
        paragraph_gap_seconds=20,
        paragraph_max_chars=500,
    )

    assert "## Transcript" in md
    assert "[00:00:00 - 00:00:18] First thought. Second sentence." in md
    assert "[3 minutes of silence]" in md
    assert "Detailed timing" in md


def test_markdown_timestamped_mode_preserves_per_segment_lines() -> None:
    canonical = _sample_canonical()
    md = render_markdown(
        canonical,
        source_file="original.m4a",
        providers=["openai", "deepgram"],
        reconciler_model="gpt-5",
        pipeline_version="0.1.0",
        output_style="timestamped",
    )

    assert "[00:00:00 - 00:00:10] First thought." in md
    assert "[00:00:11 - 00:00:18] Second sentence." in md
    assert "[3 minutes of silence]" in md
