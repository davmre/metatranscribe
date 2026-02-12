from __future__ import annotations

import json
import mimetypes
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def normalize_audio(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(input_path, output_path)
        return output_path

    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ]
    )
    return output_path


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def guess_audio_content_type(audio_path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(audio_path))
    return guessed or "application/octet-stream"


def probe_duration_sec(audio_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0

    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_path),
        ]
    )
    payload = json.loads(result.stdout or "{}")
    return float(payload.get("format", {}).get("duration", 0.0) or 0.0)


@dataclass(slots=True)
class AudioChunk:
    path: Path
    start_sec: float
    end_sec: float
    core_start_sec: float | None = None
    core_end_sec: float | None = None

    def __post_init__(self) -> None:
        if self.core_start_sec is None:
            self.core_start_sec = self.start_sec
        if self.core_end_sec is None:
            self.core_end_sec = self.end_sec


def split_audio_into_chunks(
    audio_path: Path,
    output_dir: Path,
    duration_sec: float,
    chunk_seconds: int,
    overlap_seconds: int = 0,
) -> list[AudioChunk]:
    if duration_sec <= 0 or chunk_seconds <= 0 or duration_sec <= chunk_seconds:
        return [AudioChunk(path=audio_path, start_sec=0.0, end_sec=duration_sec)]

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return [AudioChunk(path=audio_path, start_sec=0.0, end_sec=duration_sec)]

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[AudioChunk] = []
    idx = 0
    start = 0.0
    while start < duration_sec:
        core_start = start
        core_end = min(duration_sec, start + chunk_seconds)
        extract_start = max(0.0, core_start - overlap_seconds)
        extract_end = min(duration_sec, core_end + overlap_seconds)
        out_path = output_dir / f"chunk_{idx:03d}.wav"
        _run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{extract_start:.3f}",
                "-t",
                f"{max(0.1, extract_end - extract_start):.3f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(out_path),
            ]
        )
        chunks.append(
            AudioChunk(
                path=out_path,
                start_sec=extract_start,
                end_sec=extract_end,
                core_start_sec=core_start,
                core_end_sec=core_end,
            )
        )
        idx += 1
        start = core_end

    return chunks
