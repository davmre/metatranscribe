import subprocess
from pathlib import Path

from metatranscribe.preprocess.audio_prep import split_audio_into_chunks


def test_split_audio_into_chunks_without_overlap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("metatranscribe.preprocess.audio_prep.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "metatranscribe.preprocess.audio_prep._run",
        lambda cmd: subprocess.CompletedProcess(cmd, 0, "", ""),
    )

    chunks = split_audio_into_chunks(
        audio_path=tmp_path / "audio.wav",
        output_dir=tmp_path / "chunks",
        duration_sec=1300.0,
        chunk_seconds=600,
        overlap_seconds=0,
    )

    assert len(chunks) == 3
    assert chunks[0].core_start_sec == 0.0
    assert chunks[0].core_end_sec == 600.0
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 600.0
    assert chunks[1].core_start_sec == 600.0
    assert chunks[1].core_end_sec == 1200.0
    assert chunks[1].start_sec == 600.0
    assert chunks[1].end_sec == 1200.0
    assert chunks[2].core_start_sec == 1200.0
    assert chunks[2].core_end_sec == 1300.0
    assert chunks[2].start_sec == 1200.0
    assert chunks[2].end_sec == 1300.0


def test_split_audio_into_chunks_with_overlap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("metatranscribe.preprocess.audio_prep.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "metatranscribe.preprocess.audio_prep._run",
        lambda cmd: subprocess.CompletedProcess(cmd, 0, "", ""),
    )

    chunks = split_audio_into_chunks(
        audio_path=tmp_path / "audio.wav",
        output_dir=tmp_path / "chunks",
        duration_sec=1300.0,
        chunk_seconds=600,
        overlap_seconds=5,
    )

    assert len(chunks) == 3
    assert chunks[0].core_start_sec == 0.0
    assert chunks[0].core_end_sec == 600.0
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 605.0
    assert chunks[1].core_start_sec == 600.0
    assert chunks[1].core_end_sec == 1200.0
    assert chunks[1].start_sec == 595.0
    assert chunks[1].end_sec == 1205.0
    assert chunks[2].core_start_sec == 1200.0
    assert chunks[2].core_end_sec == 1300.0
    assert chunks[2].start_sec == 1195.0
    assert chunks[2].end_sec == 1300.0
