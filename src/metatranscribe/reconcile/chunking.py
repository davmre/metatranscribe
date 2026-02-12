from __future__ import annotations

from dataclasses import dataclass

from metatranscribe.models import CanonicalSegment, CanonicalTranscript


@dataclass(slots=True)
class ChunkWindow:
    index: int
    start_sec: float
    end_sec: float


def merge_chunk_canonicals(audio_id: str, chunks: list[CanonicalTranscript], duration_sec: float) -> CanonicalTranscript:
    all_segments: list[CanonicalSegment] = []
    for chunk in chunks:
        all_segments.extend(chunk.segments)

    deduped = _dedupe_segments(all_segments)
    final_text = "\n".join(seg.text for seg in deduped if seg.text.strip())
    title = chunks[0].title if chunks else "Transcript"
    language = chunks[0].language if chunks else "en"

    return CanonicalTranscript(
        audio_id=audio_id,
        title=title,
        language=language,
        duration_sec=duration_sec,
        segments=deduped,
        silence_markers=[],
        final_text=final_text,
        provenance={
            "chunk_count": len(chunks),
            "merge": "mechanical",
            "providers": sorted({ref for c in chunks for s in c.segments for ref in s.evidence_refs}),
        },
    )


def _dedupe_segments(segments: list[CanonicalSegment]) -> list[CanonicalSegment]:
    ordered = sorted(segments, key=lambda s: (s.start_sec, s.end_sec, s.text))
    result: list[CanonicalSegment] = []
    seen: set[tuple[int, int, str]] = set()

    for seg in ordered:
        text = seg.text.strip()
        if not text:
            continue
        key = (int(round(seg.start_sec * 10)), int(round(seg.end_sec * 10)), _normalize_text(text)[:120])
        if key in seen:
            continue
        seen.add(key)

        if result and seg.start_sec < result[-1].start_sec:
            continue

        if result and seg.start_sec <= result[-1].end_sec and _normalize_text(text) == _normalize_text(result[-1].text):
            if seg.end_sec > result[-1].end_sec:
                result[-1].end_sec = seg.end_sec
            continue

        result.append(seg.model_copy())

    return result


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())
