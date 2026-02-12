import json
from datetime import datetime
from pathlib import Path

from metatranscribe.config import Settings
from metatranscribe.models import AudioFileRecord, CanonicalTranscript, ProviderTranscript, Segment
from metatranscribe.orchestrator import export_step, reconcile_step, run_pipeline, transcribe_step
from metatranscribe.preprocess.audio_prep import AudioChunk
from metatranscribe.output.polish_markdown import PolishResult
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
        export_publish_dir=None,
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


def test_export_publishes_markdown_copy_with_collision_suffix(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    publish_dir = tmp_path / "published"
    settings.export_publish_dir = publish_dir
    settings.ensure_dirs()
    store = StateStore(settings.state_db_path)
    audio_id = "a4"
    store.insert_audio_if_new(
        AudioFileRecord(audio_id=audio_id, source_path="dummy.wav", source_hash="hash-a4", status="reconciled")
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

    monkeypatch.setattr(
        "metatranscribe.orchestrator.render_polished_markdown",
        lambda **kwargs: PolishResult(markdown="# Heading\n\nBody\n", suggested_name="Weekly Sync"),
    )

    collision_name = f"{datetime.now().date().isoformat()}_weekly_sync.md"
    publish_dir.mkdir(parents=True, exist_ok=True)
    (publish_dir / collision_name).write_text("existing", encoding="utf-8")

    export_step(settings, store, audio_id, dry_run=False)

    assert (settings.output_root / "final" / f"{audio_id}.md").exists()
    assert (publish_dir / collision_name).exists()
    assert (publish_dir / collision_name.replace(".md", "_2.md")).exists()
    publish_artifact = settings.output_root / "artifacts" / audio_id / "polish" / "publish_target.json"
    assert publish_artifact.exists()
    refreshed = store.get_record(audio_id)
    assert refreshed is not None
    assert refreshed.status == "exported"


def test_export_frontmatter_source_file_uses_original_inbox_basename(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    store = StateStore(settings.state_db_path)
    audio_id = "a5"
    store.insert_audio_if_new(
        AudioFileRecord(
            audio_id=audio_id,
            source_path=str(tmp_path / "data" / "inbox" / "Feb 1 at 4-19 PM.m4a"),
            source_hash="hash-a5",
            status="reconciled",
        )
    )

    raw_dir = settings.data_root / "raw" / audio_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "original.m4a").write_bytes(b"m4a")

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

    monkeypatch.setattr(
        "metatranscribe.orchestrator.render_polished_markdown",
        lambda **kwargs: PolishResult(markdown="# Heading\n\nBody\n", suggested_name="Weekly Sync"),
    )

    export_step(settings, store, audio_id, dry_run=False)

    markdown = (settings.output_root / "final" / f"{audio_id}.md").read_text(encoding="utf-8")
    assert 'source_file: "Feb 1 at 4-19 PM.m4a"' in markdown


def test_export_frontmatter_source_file_falls_back_to_raw_original_name(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    store = StateStore(settings.state_db_path)
    audio_id = "a6"

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

    monkeypatch.setattr(
        "metatranscribe.orchestrator.render_polished_markdown",
        lambda **kwargs: PolishResult(markdown="# Heading\n\nBody\n", suggested_name="Weekly Sync"),
    )

    export_step(settings, store, audio_id, dry_run=False)

    markdown = (settings.output_root / "final" / f"{audio_id}.md").read_text(encoding="utf-8")
    assert 'source_file: "original.wav"' in markdown


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


def test_transcribe_step_cleans_processed_audio_dir_on_success(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    store = StateStore(settings.state_db_path)
    audio_id = "a7"
    store.insert_audio_if_new(
        AudioFileRecord(audio_id=audio_id, source_path="dummy.wav", source_hash="hash-a7", status="ingested")
    )

    raw_dir = settings.data_root / "raw" / audio_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    original = raw_dir / "original.wav"
    original.write_bytes(b"wav")

    def _fake_normalize_audio(_input_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"normalized")
        return output_path

    def _fake_split_audio_into_chunks(
        *,
        audio_path: Path,
        output_dir: Path,
        duration_sec: float,
        chunk_seconds: int,
    ) -> list[AudioChunk]:
        del audio_path, duration_sec, chunk_seconds
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = output_dir / "chunk_000.wav"
        chunk_path.write_bytes(b"chunk")
        return [AudioChunk(path=chunk_path, start_sec=0.0, end_sec=12.0)]

    class _FakeProvider:
        name = "openai"

        def transcribe(self, audio_id: str, audio_path: Path, language_hint: str) -> ProviderTranscript:
            del audio_path, language_hint
            return ProviderTranscript(
                provider_name=self.name,
                audio_id=audio_id,
                language="en",
                duration_sec=12.0,
                segments=[Segment(start_sec=0.0, end_sec=1.0, text="hello")],
                raw_text="hello",
                confidence_summary=None,
                raw_payload_path="",
            )

    monkeypatch.setattr("metatranscribe.orchestrator.get_original_audio_path", lambda *_: original)
    monkeypatch.setattr("metatranscribe.orchestrator.has_ffmpeg", lambda: True)
    monkeypatch.setattr("metatranscribe.orchestrator.normalize_audio", _fake_normalize_audio)
    monkeypatch.setattr("metatranscribe.orchestrator.probe_duration_sec", lambda *_: 12.0)
    monkeypatch.setattr("metatranscribe.orchestrator.split_audio_into_chunks", _fake_split_audio_into_chunks)
    monkeypatch.setattr("metatranscribe.orchestrator.build_providers", lambda *_: [_FakeProvider()])

    transcribe_step(settings, store, audio_id)

    assert not (settings.data_root / "processed" / audio_id).exists()
    provider_artifact = (
        settings.output_root / "artifacts" / audio_id / "transcribe" / "chunks" / "000" / "providers" / "openai.json"
    )
    assert provider_artifact.exists()
    refreshed = store.get_record(audio_id)
    assert refreshed is not None
    assert refreshed.status == "transcribed"
