#!/usr/bin/env python3
from recorder_transcribe.config import (
    load_settings,
    validate_polish_credentials,
    validate_provider_credentials,
    validate_reconciler_credentials,
)
from recorder_transcribe.logging_utils import configure_logging
from recorder_transcribe.orchestrator import run_pipeline


def main() -> None:
    settings = load_settings()
    configure_logging(settings.logs_dir / "pipeline.log", settings.log_level)
    validate_provider_credentials(settings)
    validate_reconciler_credentials(settings)
    if settings.enable_polish_pass:
        validate_polish_credentials(settings)

    succeeded, failed = run_pipeline(settings)
    print(f"succeeded={succeeded} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
