from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    data_root: Path
    output_root: Path
    logs_dir: Path
    inbox_dir: Path
    state_db_path: Path
    transcribe_providers: list[str]
    language_hint: str
    silence_gap_seconds: int
    max_retries: int
    openai_api_key: str | None
    anthropic_api_key: str | None
    deepgram_api_key: str | None
    transcribe_openai_model: str
    transcribe_deepgram_model: str
    transcribe_chunk_seconds: int
    transcribe_chunk_overlap_seconds: int
    reconciler_model: str
    reconciler_provider: str
    polish_provider: str
    polish_model: str
    polish_long_silence_seconds: int
    export_publish_dir: Path | None
    log_level: str
    pipeline_version: str

    def ensure_dirs(self) -> None:
        for path in (
            self.data_root,
            self.output_root,
            self.logs_dir,
            self.inbox_dir,
            self.data_root / "raw",
            self.data_root / "processed",
            self.data_root / "state",
            self.output_root / "final",
            self.output_root / "artifacts",
        ):
            path.mkdir(parents=True, exist_ok=True)
        if self.export_publish_dir:
            self.export_publish_dir.mkdir(parents=True, exist_ok=True)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_settings(dotenv_path: str | None = None) -> Settings:
    load_dotenv(dotenv_path=dotenv_path)

    data_root = Path(os.getenv("DATA_ROOT", "./data")).resolve()
    output_root = Path(os.getenv("OUTPUT_ROOT", "./outputs")).resolve()
    logs_dir = Path(os.getenv("LOGS_DIR", "./logs")).resolve()
    inbox_dir = Path(os.getenv("INBOX_DIR", str(data_root / "inbox"))).resolve()

    settings = Settings(
        data_root=data_root,
        output_root=output_root,
        logs_dir=logs_dir,
        inbox_dir=inbox_dir,
        state_db_path=Path(os.getenv("STATE_DB_PATH", str(data_root / "state" / "pipeline.db"))).resolve(),
        transcribe_providers=_parse_csv(os.getenv("TRANSCRIBE_PROVIDERS", "openai,deepgram")),
        language_hint=os.getenv("LANGUAGE_HINT", "en"),
        silence_gap_seconds=int(os.getenv("SILENCE_GAP_SECONDS", "90")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
        transcribe_openai_model=os.getenv("TRANSCRIBE_OPENAI_MODEL", "gpt-4o-transcribe"),
        transcribe_deepgram_model=os.getenv("TRANSCRIBE_DEEPGRAM_MODEL", "nova-3"),
        transcribe_chunk_seconds=int(os.getenv("TRANSCRIBE_CHUNK_SECONDS", "540")),
        transcribe_chunk_overlap_seconds=int(os.getenv("TRANSCRIBE_CHUNK_OVERLAP_SECONDS", "0")),
        reconciler_model=os.getenv("RECONCILER_MODEL", "gpt-5"),
        reconciler_provider=os.getenv("RECONCILER_PROVIDER", "openai"),
        polish_provider=os.getenv("POLISH_PROVIDER", os.getenv("RECONCILER_PROVIDER", "openai")),
        polish_model=os.getenv("POLISH_MODEL", os.getenv("RECONCILER_MODEL", "gpt-5")),
        polish_long_silence_seconds=int(os.getenv("POLISH_LONG_SILENCE_SECONDS", "90")),
        export_publish_dir=Path(os.environ["EXPORT_PUBLISH_DIR"]).resolve()
        if os.getenv("EXPORT_PUBLISH_DIR", "").strip()
        else None,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        pipeline_version=os.getenv("PIPELINE_VERSION", "0.1.0"),
    )
    _validate_transcribe_chunk_settings(settings)
    settings.ensure_dirs()
    return settings


def _validate_transcribe_chunk_settings(settings: Settings) -> None:
    overlap = settings.transcribe_chunk_overlap_seconds
    chunk = settings.transcribe_chunk_seconds
    if overlap < 0:
        raise ValueError("TRANSCRIBE_CHUNK_OVERLAP_SECONDS must be >= 0")
    if chunk > 0 and overlap >= chunk:
        raise ValueError("TRANSCRIBE_CHUNK_OVERLAP_SECONDS must be < TRANSCRIBE_CHUNK_SECONDS")


def validate_provider_credentials(settings: Settings) -> None:
    if not settings.transcribe_providers:
        raise ValueError("TRANSCRIBE_PROVIDERS must contain at least one provider")

    missing: list[str] = []
    if "openai" in settings.transcribe_providers and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if "deepgram" in settings.transcribe_providers and not settings.deepgram_api_key:
        missing.append("DEEPGRAM_API_KEY")

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def validate_reconciler_credentials(settings: Settings) -> None:
    provider = settings.reconciler_provider.strip().lower()
    _validate_llm_provider_credentials(provider, settings.openai_api_key, settings.anthropic_api_key, "reconciler")


def validate_polish_credentials(settings: Settings) -> None:
    provider = settings.polish_provider.strip().lower()
    _validate_llm_provider_credentials(provider, settings.openai_api_key, settings.anthropic_api_key, "polish")


def resolve_llm_api_key(settings: Settings, provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for openai provider")
        return settings.openai_api_key
    if normalized == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for anthropic provider")
        return settings.anthropic_api_key
    raise ValueError("Provider must be one of: openai, anthropic")


def _validate_llm_provider_credentials(
    provider: str,
    openai_api_key: str | None,
    anthropic_api_key: str | None,
    context: str,
) -> None:
    if provider == "openai":
        if not openai_api_key:
            raise ValueError(f"OPENAI_API_KEY required for openai {context} provider")
        return
    if provider == "anthropic":
        if not anthropic_api_key:
            raise ValueError(f"ANTHROPIC_API_KEY required for anthropic {context} provider")
        return
    raise ValueError(f"{context.upper()}_PROVIDER must be one of: openai, anthropic")
