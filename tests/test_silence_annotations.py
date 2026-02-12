from recorder_transcribe.models import CanonicalSegment
from recorder_transcribe.postprocess.silence_annotations import build_silence_markers


def test_build_silence_markers_threshold() -> None:
    segments = [
        CanonicalSegment(start_sec=0, end_sec=10, text="a", evidence_refs=[]),
        CanonicalSegment(start_sec=70, end_sec=80, text="b", evidence_refs=[]),
        CanonicalSegment(start_sec=140, end_sec=150, text="c", evidence_refs=[]),
    ]
    markers = build_silence_markers(segments, gap_threshold_sec=60)
    assert len(markers) == 2
    assert markers[0].duration_sec == 60
    assert "silence" in markers[0].label


def test_build_silence_markers_ignores_short_gap() -> None:
    segments = [
        CanonicalSegment(start_sec=0, end_sec=10, text="a", evidence_refs=[]),
        CanonicalSegment(start_sec=65, end_sec=80, text="b", evidence_refs=[]),
    ]
    markers = build_silence_markers(segments, gap_threshold_sec=90)
    assert markers == []
