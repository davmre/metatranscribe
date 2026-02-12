from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from metatranscribe.models import AudioFileRecord
from metatranscribe.state.store import StateStore

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".ogg", ".aac", ".flac"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_new_files(inbox_dir: Path, raw_root: Path, store: StateStore) -> list[str]:
    inbox_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    ingested_ids: list[str] = []
    for path in sorted(inbox_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        source_hash = hash_file(path)
        audio_id = source_hash[:16]
        record = AudioFileRecord(
            audio_id=audio_id,
            source_path=str(path.resolve()),
            source_hash=source_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            ingested_at=datetime.now(timezone.utc).isoformat(),
            status="ingested",
        )

        if not store.insert_audio_if_new(record):
            continue

        audio_dir = raw_root / audio_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        destination = audio_dir / f"original{path.suffix.lower()}"
        shutil.copy2(path, destination)
        ingested_ids.append(audio_id)

    return ingested_ids


def get_original_audio_path(raw_root: Path, audio_id: str) -> Path:
    audio_dir = raw_root / audio_id
    candidates = sorted(audio_dir.glob("original.*"))
    if not candidates:
        raise FileNotFoundError(f"No original audio file for {audio_id}")
    return candidates[0]
