#!/usr/bin/env python3
import argparse

from metatranscribe.config import load_settings, validate_provider_credentials
from metatranscribe.logging_utils import configure_logging
from metatranscribe.orchestrator import transcribe_step
from metatranscribe.state.store import StateStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-id", default=None)
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.logs_dir / "pipeline.log", settings.log_level)
    validate_provider_credentials(settings)
    store = StateStore(settings.state_db_path)

    if args.audio_id:
        transcribe_step(settings, store, args.audio_id)
        print(f"transcribed={args.audio_id}")
        return

    for record in store.get_records(status="ingested"):
        transcribe_step(settings, store, record.audio_id)
        print(f"transcribed={record.audio_id}")


if __name__ == "__main__":
    main()
