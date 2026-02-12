from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metatranscribe.models import AudioFileRecord


class StateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS audio_files (
                    audio_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_hash TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    ingested_at TEXT,
                    transcribed_at TEXT,
                    reconciled_at TEXT,
                    exported_at TEXT,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS provider_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_id TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    summary_json TEXT,
                    UNIQUE(audio_id, provider_name)
                );
                """
            )

    def insert_audio_if_new(self, record: AudioFileRecord) -> bool:
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO audio_files (
                        audio_id, source_path, source_hash, created_at, ingested_at,
                        transcribed_at, reconciled_at, exported_at, status, retry_count, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.audio_id,
                        record.source_path,
                        record.source_hash,
                        record.created_at,
                        record.ingested_at,
                        record.transcribed_at,
                        record.reconciled_at,
                        record.exported_at,
                        record.status,
                        record.retry_count,
                        record.error,
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def get_records(self, status: str | None = None) -> list[AudioFileRecord]:
        query = "SELECT * FROM audio_files"
        params: tuple[Any, ...] = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [AudioFileRecord(**dict(row)) for row in rows]

    def get_record(self, audio_id: str) -> AudioFileRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM audio_files WHERE audio_id = ?", (audio_id,)).fetchone()
        return AudioFileRecord(**dict(row)) if row else None

    def update_status(self, audio_id: str, status: str, error: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        fields = ["status = ?", "error = ?"]
        values: list[Any] = [status, error]
        if status == "transcribed":
            fields.append("transcribed_at = ?")
            values.append(now)
        elif status == "reconciled":
            fields.append("reconciled_at = ?")
            values.append(now)
        elif status == "exported":
            fields.append("exported_at = ?")
            values.append(now)
        elif status == "ingested":
            fields.append("ingested_at = ?")
            values.append(now)

        values.append(audio_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE audio_files SET {', '.join(fields)} WHERE audio_id = ?", values)

    def increment_retry(self, audio_id: str) -> int:
        with self._connect() as conn:
            conn.execute(
                "UPDATE audio_files SET retry_count = retry_count + 1 WHERE audio_id = ?",
                (audio_id,),
            )
            retry = conn.execute(
                "SELECT retry_count FROM audio_files WHERE audio_id = ?",
                (audio_id,),
            ).fetchone()[0]
        return int(retry)

    def save_provider_run(
        self,
        audio_id: str,
        provider_name: str,
        artifact_path: str,
        summary: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO provider_runs (audio_id, provider_name, created_at, artifact_path, summary_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(audio_id, provider_name)
                DO UPDATE SET created_at = excluded.created_at,
                              artifact_path = excluded.artifact_path,
                              summary_json = excluded.summary_json
                """,
                (
                    audio_id,
                    provider_name,
                    datetime.now(timezone.utc).isoformat(),
                    artifact_path,
                    json.dumps(summary or {}),
                ),
            )
