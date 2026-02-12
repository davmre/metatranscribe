from pathlib import Path

from metatranscribe.models import AudioFileRecord
from metatranscribe.state.store import StateStore


def test_insert_audio_dedupes_by_hash(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "pipeline.db")
    record1 = AudioFileRecord(audio_id="id1", source_path="a", source_hash="h")
    record2 = AudioFileRecord(audio_id="id2", source_path="b", source_hash="h")

    assert store.insert_audio_if_new(record1) is True
    assert store.insert_audio_if_new(record2) is False


def test_status_update_sets_timestamp(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "pipeline.db")
    record = AudioFileRecord(audio_id="id1", source_path="a", source_hash="h")
    store.insert_audio_if_new(record)

    store.update_status("id1", "transcribed")
    refreshed = store.get_record("id1")
    assert refreshed is not None
    assert refreshed.transcribed_at is not None
    assert refreshed.status == "transcribed"
