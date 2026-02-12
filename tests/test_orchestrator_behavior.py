import json
from pathlib import Path

from metatranscribe.config import Settings
from metatranscribe.models import AudioFileRecord, CanonicalTranscript, ProviderTranscript, Segment
from metatranscribe.orchestrator import export_step, reconcile_step, run_pipeline
from metatranscribe.state.store import StateStore


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path / "data",
        output_root=tmp_path / "outputs",
        logs_dir=tmp_path / "logs",
        inbox_dir=tmp_path / "data" / "inbox",
        state_db_path=tmp_path / "data" / "state" / "pipeline.db",
        transcribe_providers=["openai"],
        language_hint="en",
        silence_gap_seconds=90,
        max_retries=3,
        openai_api_key="test-key",
        anthropic_api_key=None,
        deepgram_api_key=None,
        transcribe_openai_model="gpt-4o-transcribe",
        transcribe_deepgram_model="nova-3",
        transcribe_chunk_seconds=540,
        reconciler_model="gpt-5",
        reconciler_provider="openai",
        polish_provider="openai",
        polish_model="gpt-5",
        polish_long_silence_seconds=90,
        log_level="INFO",
        pipeline_version="0.1.0",
    )


def test_reconcile_dry_run_writes_prompt_only_and_preserves_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    store = StateStore(settings.state_db_path)
    audio_id = "a1"
    store.insert_audio_if_new(
        AudioFileRecord(audio_id=audio_id, source_path="dummy.wav", source_hash="hash-a1", status="transcribed")
    )

    artifacts_root = settings.output_root / "artifacts" / audio_id
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "transcribe_chunks.json").write_text(
        json.dumps([{"index": 0, "start_sec": 0.0, "end_sec": 12.0}], indent=2),
        encoding="utf-8",
    )
    provider = ProviderTranscript(
        provider_name="openai",
        audio_id=audio_id,
        language="en",
        duration_sec=12.0,
        segments=[Segment(start_sec=0.0, end_sec=2.0, text="hello")],
        raw_text="hello",
        confidence_summary=None,
        raw_payload_path="",
    )
    provider_path = artifacts_root / "transcribe" / "chunks" / "000" / "providers" / "openai.json"
    provider_path.parent.mkdir(parents=True, exist_ok=True)
    provider_path.write_text(json.dumps(provider.model_dump(), indent=2), encoding="utf-8")

    reconcile_step(settings, store, audio_id, dry_run=True)

    assert (artifacts_root / "reconcile" / "chunks" / "000" / "request_prompt.json").exists()
    assert not (artifacts_root / "reconcile" / "chunks" / "000" / "response_raw.txt").exists()
    assert not (artifacts_root / "canonical.json").exists()
    refreshed = store.get_record(audio_id)
    assert refreshed is not None
    assert refreshed.status == "transcribed"


def test_export_dry_run_writes_prompt_only_and_preserves_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    store = StateStore(settings.state_db_path)
    audio_id = "a2"
    store.insert_audio_if_new(
        AudioFileRecord(audio_id=audio_id, source_path="dummy.wav", source_hash="hash-a2", status="reconciled")
    )

    raw_dir = settings.data_root / "raw" / audio_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "original.wav").write_bytes(b"wav")

    canonical = CanonicalTranscript(
        audio_id=audio_id,
        title="Title",
        language="en",
        duration_sec=12.0,
        segments=[],
        silence_markers=[],
        final_text="",
        provenance={},
    )
    canonical_path = settings.output_root / "artifacts" / audio_id / "canonical.json"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(json.dumps(canonical.model_dump(), indent=2), encoding="utf-8")

    export_step(settings, store, audio_id, dry_run=True)

    assert (settings.output_root / "artifacts" / audio_id / "polish" / "request_prompt.json").exists()
    assert not (settings.output_root / "artifacts" / audio_id / "polish" / "response_raw.md").exists()
    assert not (settings.output_root / "final" / f"{audio_id}.md").exists()
    assert not (settings.output_root / "final" / f"{audio_id}.json").exists()
    refreshed = store.get_record(audio_id)
    assert refreshed is not None
    assert refreshed.status == "reconciled"


def test_run_pipeline_skips_failed_records_at_retry_cap(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings.max_retries = 3
    settings.ensure_dirs()
    store = StateStore(settings.state_db_path)
    audio_id = "a3"
    store.insert_audio_if_new(
        AudioFileRecord(
            audio_id=audio_id,
            source_path="dummy.wav",
            source_hash="hash-a3",
            status="failed",
            retry_count=3,
        )
    )

    called = {"transcribe": False}

    def _fake_transcribe_step(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        called["transcribe"] = True

    monkeypatch.setattr("metatranscribe.orchestrator.transcribe_step", _fake_transcribe_step)
    succeeded, failed = run_pipeline(settings)

    assert succeeded == 0
    assert failed == 0
    assert called["transcribe"] is False
    refreshed = store.get_record(audio_id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.retry_count == 3
