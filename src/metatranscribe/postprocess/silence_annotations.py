from __future__ import annotations

from metatranscribe.models import CanonicalSegment, SilenceMarker


def _format_gap_label(duration_sec: float) -> str:
    rounded = int(round(duration_sec / 15.0) * 15)
    minutes = rounded // 60
    seconds = rounded % 60
    if minutes and seconds:
        return f"[{minutes}m {seconds}s of silence]"
    if minutes:
        unit = "minute" if minutes == 1 else "minutes"
        return f"[{minutes} {unit} of silence]"
    return f"[{seconds} seconds of silence]"


def build_silence_markers(segments: list[CanonicalSegment], gap_threshold_sec: int) -> list[SilenceMarker]:
    markers: list[SilenceMarker] = []
    if not segments:
        return markers

    ordered = sorted(segments, key=lambda s: s.start_sec)
    for prev, nxt in zip(ordered, ordered[1:]):
        gap = nxt.start_sec - prev.end_sec
        if gap >= gap_threshold_sec:
            markers.append(
                SilenceMarker(
                    start_sec=prev.end_sec,
                    end_sec=nxt.start_sec,
                    duration_sec=gap,
                    label=_format_gap_label(gap),
                )
            )

    return markers
