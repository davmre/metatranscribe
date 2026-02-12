#!/usr/bin/env python3
from recorder_transcribe.config import load_settings
from recorder_transcribe.ingest.manual_inbox import ingest_new_files
from recorder_transcribe.state.store import StateStore


def main() -> None:
    settings = load_settings()
    store = StateStore(settings.state_db_path)
    ids = ingest_new_files(settings.inbox_dir, settings.data_root / "raw", store)
    print(f"ingested={len(ids)}")
    for audio_id in ids:
        print(audio_id)


if __name__ == "__main__":
    main()
