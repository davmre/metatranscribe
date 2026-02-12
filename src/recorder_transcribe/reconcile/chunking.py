from __future__ import annotations

from dataclasses import dataclass

from recorder_transcribe.models import CanonicalSegment, CanonicalTranscript, ProviderTranscript, Segment


@dataclass(slots=True)
class ChunkWindow:
    index: int
    start_sec: float
    end_sec: float


def estimate_duration_sec(transcripts: list[ProviderTranscript]) -> float:
    duration = 0.0
    for transcript in transcripts:
        duration = max(duration, transcript.duration_sec)
        if transcript.segments:
            duration = max(duration, max(seg.end_sec for seg in transcript.segments))
    return duration


def build_windows(duration_sec: float, chunk_seconds: int, overlap_seconds: int) -> list[ChunkWindow]:
    if duration_sec <= 0:
        return [ChunkWindow(index=0, start_sec=0.0, end_sec=float(chunk_seconds))]

    chunk = max(30, chunk_seconds)
    overlap = max(0, min(overlap_seconds, chunk // 2))
    step = max(1, chunk - overlap)

    windows: list[ChunkWindow] = []
    start = 0.0
    idx = 0
    while start < duration_sec:
        end = min(duration_sec, start + chunk)
        windows.append(ChunkWindow(index=idx, start_sec=start, end_sec=end))
        idx += 1
        start += step

    if not windows:
        windows.append(ChunkWindow(index=0, start_sec=0.0, end_sec=float(chunk)))
    return windows


def slice_transcripts_for_window(transcripts: list[ProviderTranscript], window: ChunkWindow) -> list[ProviderTranscript]:
    sliced: list[ProviderTranscript] = []
    for transcript in transcripts:
        window_segments = [
            seg.model_copy()
            for seg in transcript.segments
            if seg.end_sec >= window.start_sec and seg.start_sec <= window.end_sec
        ]
        raw_text = " ".join(seg.text.strip() for seg in window_segments if seg.text.strip()).strip()

        sliced.append(
            ProviderTranscript(
                provider_name=transcript.provider_name,
                audio_id=transcript.audio_id,
                language=transcript.language,
                duration_sec=max(0.0, window.end_sec - window.start_sec),
                segments=window_segments,
                raw_text=raw_text,
                confidence_summary=transcript.confidence_summary,
                raw_payload_path=transcript.raw_payload_path,
            )
        )

    return sliced


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
