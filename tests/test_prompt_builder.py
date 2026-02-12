import json

from recorder_transcribe.models import ProviderTranscript, Segment
from recorder_transcribe.reconcile.prompt_builder import build_reconciliation_prompt


def test_prompt_builder_compacts_segments_and_marks_timing() -> None:
    deepgram = ProviderTranscript(
        provider_name="deepgram",
        audio_id="a",
        language="en",
        duration_sec=12.0,
        raw_text="hello world this is a test",
        raw_payload_path="",
        confidence_summary=None,
        segments=[
            Segment(start_sec=0.0, end_sec=1.0, text="hello", confidence=0.9),
            Segment(start_sec=1.1, end_sec=2.0, text="world", confidence=0.9),
            Segment(start_sec=5.0, end_sec=6.0, text="next", confidence=0.9),
        ],
    )
    openai = ProviderTranscript(
        provider_name="openai",
        audio_id="a",
        language="en",
        duration_sec=0.0,
        raw_text="full transcript only",
        raw_payload_path="",
        confidence_summary=None,
        segments=[Segment(start_sec=0.0, end_sec=0.0, text="full transcript only")],
    )

    prompt = build_reconciliation_prompt("a", [deepgram, openai], "en")
    payload = json.loads(prompt)
    assert payload["audio_id"] == "a"
    providers = {p["provider"]: p for p in payload["evidence"]}

    assert providers["deepgram"]["segment_count_original"] == 3
    assert providers["deepgram"]["segment_count_compact"] == 2
    assert providers["deepgram"]["has_timed_segments"] is True
    assert providers["openai"]["has_timed_segments"] is False
