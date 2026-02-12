from __future__ import annotations

import json

from recorder_transcribe.models import ProviderTranscript, Segment

MAX_PROVIDER_RAW_TEXT_CHARS = 4000
SEGMENT_MERGE_GAP_SECONDS = 1.2
MERGED_SEGMENT_MAX_CHARS = 280
MAX_MERGED_SEGMENTS = 120


def build_reconciliation_prompt(audio_id: str, transcripts: list[ProviderTranscript], language_hint: str) -> str:
    provider_payload = []
    for transcript in transcripts:
        compact_segments = _compact_segments(transcript.segments)
        has_timing = any(seg.end_sec > seg.start_sec for seg in compact_segments)
        raw_text = transcript.raw_text.strip()
        if len(raw_text) > MAX_PROVIDER_RAW_TEXT_CHARS:
            raw_text = raw_text[:MAX_PROVIDER_RAW_TEXT_CHARS].rstrip() + " ..."

        provider_payload.append(
            {
                "provider": transcript.provider_name,
                "language": transcript.language,
                "duration_sec": transcript.duration_sec,
                "has_timed_segments": has_timing,
                # "raw_text": raw_text,
                "segment_count_original": len(transcript.segments),
                "segment_count_compact": len(compact_segments),
                "segments": [
                    {
                        "start_sec": s.start_sec,
                        "end_sec": s.end_sec,
                        "text": s.text,
                        "confidence": s.confidence,
                    }
                    for s in compact_segments
                ],
            }
        )

    schema = {
        "audio_id": "string",
        "title": "string",
        "language": "string",
        "duration_sec": "number",
        "segments": [
            {
                "start_sec": "number",
                "end_sec": "number",
                "text": "string",
                "evidence_refs": ["provider names that support this phrasing"],
            }
        ],
        "silence_markers": [],
        "final_text": "string",
        "provenance": {"providers": ["string"], "notes": "string"},
    }

    instructions = {
        "task": "Reconcile multiple transcripts into one best technical transcript.",
        "constraints": [
            "Return strict JSON only.",
            "Keep segments in chronological order.",
            "If uncertain, preserve token in square brackets.",
            "Set audio_id exactly as provided.",
            "Prefer timed segments from providers that supply them.",
            "Use untimed provider text as lexical support for wording/terminology.",
        ],
        "audio_id": audio_id,
        "language_hint": language_hint,
        "output_schema": schema,
        "evidence": provider_payload,
    }

    return json.dumps(instructions)


def _compact_segments(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s.start_sec, s.end_sec))
    merged: list[Segment] = []

    for seg in ordered:
        text = seg.text.strip()
        if not text:
            continue

        if not merged:
            merged.append(seg.model_copy())
            continue

        prev = merged[-1]
        gap = seg.start_sec - prev.end_sec
        combined_len = len(prev.text.strip()) + 1 + len(text)
        should_merge = (
            gap <= SEGMENT_MERGE_GAP_SECONDS
            and combined_len <= MERGED_SEGMENT_MAX_CHARS
            and (seg.end_sec > seg.start_sec)
            and (prev.end_sec > prev.start_sec)
        )
        if not should_merge:
            merged.append(seg.model_copy())
            continue

        prev.text = f"{prev.text.strip()} {text}".strip()
        prev.end_sec = max(prev.end_sec, seg.end_sec)
        if prev.confidence is not None and seg.confidence is not None:
            prev.confidence = (prev.confidence + seg.confidence) / 2.0
        elif seg.confidence is not None:
            prev.confidence = seg.confidence

    return merged[:MAX_MERGED_SEGMENTS]
