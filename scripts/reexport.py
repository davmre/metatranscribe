#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from recorder_transcribe.config import load_settings, validate_polish_credentials, validate_reconciler_credentials
from recorder_transcribe.logging_utils import configure_logging
from recorder_transcribe.orchestrator import export_step
from recorder_transcribe.state.store import StateStore


def _discover_audio_ids_from_canonical(artifacts_root: Path) -> list[str]:
    audio_ids: list[str] = []
    if not artifacts_root.exists():
        return audio_ids

    for canonical_path in sorted(artifacts_root.glob("*/canonical.json")):
        audio_ids.append(canonical_path.parent.name)
    return audio_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run export/polish from existing canonical artifacts")
    parser.add_argument("--audio-id", default=None, help="Single audio_id to re-export")
    parser.add_argument("--all", action="store_true", help="Re-export all audio_ids with canonical.json")
    parser.add_argument("--dry-run", default=False, action="store_true", help="Write polish prompts only; no model call or output writes")
    args = parser.parse_args()

    if not args.audio_id and not args.all:
        parser.error("Provide --audio-id <id> or --all")

    settings = load_settings()
    configure_logging(settings.logs_dir / "pipeline.log", settings.log_level)
    validate_reconciler_credentials(settings)
    validate_polish_credentials(settings)

    store = StateStore(settings.state_db_path)

    if args.audio_id:
        export_step(settings, store, args.audio_id, dry_run=args.dry_run)
        if args.dry_run:
            print(f"dry_run_reexported={args.audio_id}")
        else:
            print(f"reexported={args.audio_id}")
        return

    audio_ids = _discover_audio_ids_from_canonical(settings.output_root / "artifacts")
    if not audio_ids:
        print("reexported=0")
        return

    count = 0
    for audio_id in audio_ids:
        export_step(settings, store, audio_id, dry_run=args.dry_run)
        if args.dry_run:
            print(f"dry_run_reexported={audio_id}")
        else:
            print(f"reexported={audio_id}")
        count += 1

    if args.dry_run:
        print(f"dry_run_reexported_total={count}")
    else:
        print(f"reexported_total={count}")


if __name__ == "__main__":
    main()
