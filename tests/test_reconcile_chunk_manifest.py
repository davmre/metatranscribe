import json
from pathlib import Path

from recorder_transcribe.orchestrator import _load_transcription_chunk_windows


def test_load_transcription_chunk_windows(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts" / "a"
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = [
        {"index": 0, "start_sec": 0.0, "end_sec": 300.0},
        {"index": 1, "start_sec": 300.0, "end_sec": 600.0},
    ]
    (artifacts / "transcribe_chunks.json").write_text(json.dumps(payload), encoding="utf-8")

    windows = _load_transcription_chunk_windows(artifacts)
    assert len(windows) == 2
    assert windows[0].start_sec == 0.0
    assert windows[1].end_sec == 600.0
