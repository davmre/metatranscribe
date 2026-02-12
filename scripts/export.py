#!/usr/bin/env python3
import argparse

from metatranscribe.config import load_settings
from metatranscribe.orchestrator import export_step
from metatranscribe.state.store import StateStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-id", default=None)
    args = parser.parse_args()

    settings = load_settings()
    store = StateStore(settings.state_db_path)

    if args.audio_id:
        export_step(settings, store, args.audio_id)
        print(f"exported={args.audio_id}")
        return

    for record in store.get_records(status="reconciled"):
        export_step(settings, store, record.audio_id)
        print(f"exported={record.audio_id}")


if __name__ == "__main__":
    main()
