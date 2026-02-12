#!/usr/bin/env python3
import argparse

from recorder_transcribe.config import load_settings, validate_reconciler_credentials
from recorder_transcribe.orchestrator import reconcile_step
from recorder_transcribe.state.store import StateStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-id", default=None)
    parser.add_argument("--dry-run", default=False, action='store_true')
    args = parser.parse_args()

    settings = load_settings()
    validate_reconciler_credentials(settings)
    store = StateStore(settings.state_db_path)

    if args.audio_id:
        reconcile_step(settings, store, args.audio_id, dry_run=args.dry_run)
        print(f"reconciled={args.audio_id}")
        return

    for record in store.get_records(status="transcribed"):
        reconcile_step(settings, store, record.audio_id, dry_run=args.dry_run)
        print(f"reconciled={record.audio_id}")


if __name__ == "__main__":
    main()
