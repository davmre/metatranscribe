from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from metatranscribe.config import (
    Settings,
    resolve_llm_api_key,
    validate_polish_credentials,
    validate_reconciler_credentials,
)
from metatranscribe.ingest.manual_inbox import get_original_audio_path, ingest_new_files
from metatranscribe.models import CanonicalTranscript, ProviderTranscript, Segment
from metatranscribe.output.io import write_outputs
from metatranscribe.output.polish_markdown import render_polished_markdown
from metatranscribe.output.publish import build_publish_filename, copy_published_markdown
from metatranscribe.postprocess.silence_annotations import build_silence_markers
from metatranscribe.preprocess.audio_prep import (
    AudioChunk,
    has_ffmpeg,
    normalize_audio,
    probe_duration_sec,
    split_audio_into_chunks,
)
from metatranscribe.reconcile.chunking import (
    ChunkWindow,
    merge_chunk_canonicals,
)
from metatranscribe.reconcile.llm_reconciler import LLMReconciler, save_canonical_transcript
from metatranscribe.state.store import StateStore
from metatranscribe.transcribe.runner import (
    build_providers,
    discover_chunk_providers,
    load_provider_chunk_transcripts,
    save_provider_chunk_transcript,
)

logger = logging.getLogger(__name__)


def ingest_step(settings: Settings, store: StateStore) -> list[str]:
    ids = ingest_new_files(settings.inbox_dir, settings.data_root / "raw", store)
    logger.info("Ingested %d new files", len(ids))
    return ids


def transcribe_step(settings: Settings, store: StateStore, audio_id: str) -> None:
    logger.info("Transcription step started audio_id=%s", audio_id)
    original_path = get_original_audio_path(settings.data_root / "raw", audio_id)
    output_suffix = ".wav" if has_ffmpeg() else original_path.suffix.lower()
    processed_audio = settings.data_root / "processed" / audio_id / f"audio{output_suffix}"
    normalize_audio(original_path, processed_audio)

    duration = probe_duration_sec(processed_audio)
    chunks = split_audio_into_chunks(
        audio_path=processed_audio,
        output_dir=settings.data_root / "processed" / audio_id / "chunks",
        duration_sec=duration,
        chunk_seconds=settings.transcribe_chunk_seconds,
        overlap_seconds=settings.transcribe_chunk_overlap_seconds,
    )
    logger.info(
        "Prepared transcription chunks audio_id=%s count=%d duration_sec=%.2f chunk_sec=%d overlap_sec=%d",
        audio_id,
        len(chunks),
        duration,
        settings.transcribe_chunk_seconds,
        settings.transcribe_chunk_overlap_seconds,
    )
    _write_transcription_chunk_manifest(settings.output_root / "artifacts" / audio_id, chunks)
    providers = build_providers(settings)
    if any(p.name == "openai" for p in providers):
        oversized = [c for c in chunks if c.path.exists() and c.path.stat().st_size > 25 * 1024 * 1024]
        if oversized:
            example = oversized[0]
            size_mb = example.path.stat().st_size / (1024 * 1024)
            raise RuntimeError(
                "OpenAI transcription chunk exceeds 25 MB upload limit "
                f"(example chunk={example.path.name} size_mb={size_mb:.1f}). "
                "Reduce TRANSCRIBE_CHUNK_SECONDS and ensure ffmpeg is installed so splitting can run."
            )
    artifacts_root = settings.output_root / "artifacts" / audio_id

    for provider in providers:
        if provider.name == "openai":
            model_hint = settings.transcribe_openai_model
        elif provider.name == "deepgram":
            model_hint = settings.transcribe_deepgram_model
        else:
            model_hint = "default"
        logger.info(
            "Calling transcription provider audio_id=%s provider=%s model_hint=%s",
            audio_id,
            provider.name,
            model_hint,
        )
        provider_segments = 0
        for chunk_index, chunk in enumerate(chunks):
            logger.info(
                "Transcribing chunk audio_id=%s provider=%s chunk=%d core_start=%.2f core_end=%.2f extract_start=%.2f extract_end=%.2f",
                audio_id,
                provider.name,
                chunk_index,
                chunk.core_start_sec,
                chunk.core_end_sec,
                chunk.start_sec,
                chunk.end_sec,
            )
            chunk_audio_id = f"{audio_id}__chunk_{chunk_index:03d}"
            transcript = provider.transcribe(chunk_audio_id, chunk.path, settings.language_hint)
            transcript = _with_global_segment_offsets(audio_id, transcript, chunk)
            save_provider_chunk_transcript(transcript, artifacts_root, chunk_index)
            provider_segments += len(transcript.segments)

        store.save_provider_run(
            audio_id,
            provider.name,
            str(artifacts_root / "transcribe" / "chunks"),
            summary={"duration_sec": duration, "segments": provider_segments, "chunks": len(chunks)},
        )

    store.update_status(audio_id, "transcribed")
    _cleanup_processed_audio(settings.data_root / "processed" / audio_id, audio_id)
    logger.info("Transcription step completed audio_id=%s providers=%d", audio_id, len(providers))


def reconcile_step(settings: Settings, store: StateStore, audio_id: str, dry_run: bool = False) -> CanonicalTranscript:
    logger.info(
        "Reconciliation step started audio_id=%s provider=%s model=%s",
        audio_id,
        settings.reconciler_provider,
        settings.reconciler_model,
    )
    if dry_run:
        logger.info("Reconciliation running in dry-run mode audio_id=%s", audio_id)
    artifacts_root = settings.output_root / "artifacts" / audio_id
    windows = _load_transcription_chunk_windows(artifacts_root)
    if not windows:
        if _has_legacy_provider_artifacts(artifacts_root):
            raise RuntimeError(
                f"Legacy merged provider transcripts detected for {audio_id}. "
                "This pipeline now requires chunk-based artifacts; rerun transcribe."
            )
        raise RuntimeError(f"Missing transcription chunk manifest for {audio_id}")

    expected_providers = discover_chunk_providers(artifacts_root)
    if not expected_providers:
        if _has_legacy_provider_artifacts(artifacts_root):
            raise RuntimeError(
                f"Legacy merged provider transcripts detected for {audio_id}. "
                "This pipeline now requires chunk-based artifacts; rerun transcribe."
            )
        raise RuntimeError(f"No chunk provider transcripts found for {audio_id}")
    validate_reconciler_credentials(settings)
    reconciler_provider = settings.reconciler_provider.strip().lower()
    api_key = resolve_llm_api_key(settings, reconciler_provider)

    reconcile_artifacts_dir = artifacts_root / "reconcile"
    duration_sec = _manifest_duration_sec(windows)
    logger.info(
        "Chunked reconciliation planned audio_id=%s windows=%d duration_sec=%.2f",
        audio_id,
        len(windows),
        duration_sec,
    )
    chunk_results: list[CanonicalTranscript] = []
    for window in windows:
        chunk_id = f"{audio_id}__chunk_{window.index:03d}"
        chunk_dir = reconcile_artifacts_dir / "chunks" / f"{window.index:03d}"
        window_transcripts = load_provider_chunk_transcripts(artifacts_root, window.index)
        _validate_chunk_provider_completeness(audio_id, window.index, expected_providers, window_transcripts)
        logger.info(
            "Reconciling chunk audio_id=%s chunk=%d start=%.2f end=%.2f providers=%d",
            audio_id,
            window.index,
            window.start_sec,
            window.end_sec,
            len(window_transcripts),
        )
        reconciler = LLMReconciler(
            api_key,
            settings.reconciler_model,
            reconciler_provider,
            artifacts_dir=chunk_dir,
        )
        chunk_canonical = reconciler.reconcile(chunk_id, window_transcripts, settings.language_hint, dry_run=dry_run)
        if not dry_run:
            chunk_canonical.audio_id = audio_id
            (chunk_dir / "canonical.json").write_text(json.dumps(chunk_canonical.model_dump(), indent=2), encoding="utf-8")
            chunk_results.append(chunk_canonical)

    if dry_run:
        logger.info("Reconciliation dry-run completed audio_id=%s chunks=%d", audio_id, len(windows))
        return CanonicalTranscript(
            audio_id=audio_id,
            title="Dry-run preview",
            language=settings.language_hint,
            duration_sec=duration_sec,
            segments=[],
            silence_markers=[],
            final_text="",
            provenance={"dry_run": True, "chunk_count": len(windows)},
        )

    canonical = merge_chunk_canonicals(audio_id, chunk_results, duration_sec)

    canonical.silence_markers = build_silence_markers(canonical.segments, settings.silence_gap_seconds)
    canonical_path = settings.output_root / "artifacts" / audio_id / "canonical.json"
    save_canonical_transcript(canonical, canonical_path)
    store.update_status(audio_id, "reconciled")
    logger.info("Reconciliation step completed audio_id=%s segments=%d", audio_id, len(canonical.segments))
    return canonical


def export_step(settings: Settings, store: StateStore, audio_id: str, dry_run: bool = False) -> None:
    logger.info("Export step started audio_id=%s", audio_id)
    if dry_run:
        logger.info("Export running in dry-run mode audio_id=%s", audio_id)
    canonical_path = settings.output_root / "artifacts" / audio_id / "canonical.json"
    payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical = CanonicalTranscript.model_validate(payload)

    source_file = _resolve_source_filename(settings, store, audio_id)
    providers = discover_chunk_providers(settings.output_root / "artifacts" / audio_id)
    validate_polish_credentials(settings)
    polish_provider = settings.polish_provider.strip().lower()
    polish_api_key = resolve_llm_api_key(settings, polish_provider)
    polish_result = render_polished_markdown(
        canonical=canonical,
        provider=polish_provider,
        model=settings.polish_model,
        api_key=polish_api_key,
        long_silence_seconds=settings.polish_long_silence_seconds,
        artifacts_dir=settings.output_root / "artifacts" / audio_id / "polish",
        dry_run=dry_run,
    )
    if dry_run:
        logger.info("Export dry-run completed audio_id=%s", audio_id)
        return
    markdown = _render_markdown_with_frontmatter(
        body=polish_result.markdown,
        canonical=canonical,
        source_file=source_file,
        providers=providers,
        reconciler_model=settings.reconciler_model,
        pipeline_version=settings.pipeline_version,
    )

    final_md = settings.output_root / "final" / f"{audio_id}.md"
    final_json = settings.output_root / "final" / f"{audio_id}.json"
    write_outputs(canonical, final_md, final_json, markdown)
    published_path: Path | None = None
    if settings.export_publish_dir:
        publish_filename = build_publish_filename(datetime.now().date(), polish_result.suggested_name)
        published_path = copy_published_markdown(final_md, settings.export_publish_dir, publish_filename)
        _write_publish_artifact(
            settings.output_root / "artifacts" / audio_id / "polish",
            publish_filename,
            published_path.name,
            polish_result.suggested_name,
        )
    store.update_status(audio_id, "exported")
    logger.info(
        "Export step completed audio_id=%s output=%s published_copy=%s",
        audio_id,
        final_md,
        str(published_path) if published_path else "none",
    )


def _resolve_source_filename(settings: Settings, store: StateStore, audio_id: str) -> str:
    record = store.get_record(audio_id)
    if record and record.source_path.strip():
        return Path(record.source_path).name
    return get_original_audio_path(settings.data_root / "raw", audio_id).name


def run_pipeline(settings: Settings) -> tuple[int, int]:
    store = StateStore(settings.state_db_path)
    ingest_step(settings, store)

    records = [r for r in store.get_records() if r.status in {"ingested", "failed", "transcribed", "reconciled"}]
    succeeded = 0
    failed = 0

    for record in records:
        logger.info("Processing audio_id=%s current_status=%s", record.audio_id, record.status)
        try:
            if record.status == "failed" and record.retry_count >= settings.max_retries:
                logger.warning(
                    "Skipping failed audio_id=%s retry_count=%d max_retries=%d",
                    record.audio_id,
                    record.retry_count,
                    settings.max_retries,
                )
                continue
            if record.status in {"ingested", "failed"}:
                transcribe_step(settings, store, record.audio_id)
                record = store.get_record(record.audio_id) or record
            if record.status == "transcribed":
                reconcile_step(settings, store, record.audio_id)
                record = store.get_record(record.audio_id) or record
            if record.status == "reconciled":
                export_step(settings, store, record.audio_id)

            succeeded += 1
            logger.info("Processing succeeded audio_id=%s", record.audio_id)
        except Exception as exc:  # noqa: BLE001
            retry = store.increment_retry(record.audio_id)
            store.update_status(record.audio_id, "failed", error=str(exc))
            failed += 1
            logger.exception("Failed audio_id=%s retry=%s", record.audio_id, retry)

    return succeeded, failed


def _render_markdown_with_frontmatter(
    body: str,
    canonical: CanonicalTranscript,
    source_file: str,
    providers: list[str],
    reconciler_model: str,
    pipeline_version: str,
) -> str:
    frontmatter = {
        "audio_id": canonical.audio_id,
        "source_file": source_file,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": canonical.duration_sec,
        "language": canonical.language,
        "providers": providers,
        "reconciler_model": reconciler_model,
        "pipeline_version": pipeline_version,
    }
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}: {json.dumps(value)}")
        elif isinstance(value, (float, int)):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f'{key}: "{str(value).replace(chr(34), chr(39))}"')
    lines.extend(["---", "", body.strip(), ""])
    return "\n".join(lines)


def _write_publish_artifact(
    polish_artifacts_dir: Path,
    requested_filename: str,
    copied_filename: str,
    suggested_name: str,
) -> None:
    polish_artifacts_dir.mkdir(parents=True, exist_ok=True)
    (polish_artifacts_dir / "publish_target.json").write_text(
        json.dumps(
            {
                "requested_filename": requested_filename,
                "copied_filename": copied_filename,
                "suggested_name": suggested_name,
                "collision_resolved": requested_filename != copied_filename,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _cleanup_processed_audio(processed_dir: Path, audio_id: str) -> None:
    if not processed_dir.exists():
        return
    try:
        shutil.rmtree(processed_dir)
        logger.info("Cleaned processed audio files audio_id=%s path=%s", audio_id, processed_dir)
    except OSError:
        logger.warning(
            "Failed to clean processed audio files audio_id=%s path=%s",
            audio_id,
            processed_dir,
            exc_info=True,
        )


def _write_transcription_chunk_manifest(artifacts_root: Path, chunks: list[AudioChunk]) -> None:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "index": idx,
            "start_sec": chunk.core_start_sec,
            "end_sec": chunk.core_end_sec,
            "extract_start_sec": chunk.start_sec,
            "extract_end_sec": chunk.end_sec,
        }
        for idx, chunk in enumerate(chunks)
    ]
    (artifacts_root / "transcribe_chunks.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_transcription_chunk_windows(artifacts_root: Path) -> list[ChunkWindow]:
    path = artifacts_root / "transcribe_chunks.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    windows: list[ChunkWindow] = []
    for item in payload:
        windows.append(
            ChunkWindow(
                index=int(item.get("index", len(windows))),
                start_sec=float(item.get("start_sec", 0.0)),
                end_sec=float(item.get("end_sec", 0.0)),
            )
        )
    return windows


def _with_global_segment_offsets(audio_id: str, transcript: ProviderTranscript, chunk: AudioChunk) -> ProviderTranscript:
    global_segments: list[Segment] = []
    for seg in transcript.segments:
        global_segments.append(
            Segment(
                start_sec=seg.start_sec + chunk.start_sec,
                end_sec=seg.end_sec + chunk.start_sec,
                text=seg.text,
                confidence=seg.confidence,
                speaker=seg.speaker,
            )
        )

    duration = transcript.duration_sec
    if duration <= 0.0:
        duration = max(0.0, chunk.end_sec - chunk.start_sec)

    return transcript.model_copy(
        update={
            "audio_id": audio_id,
            "duration_sec": duration,
            "segments": global_segments,
            "raw_payload_path": "",
        }
    )


def _manifest_duration_sec(windows: list[ChunkWindow]) -> float:
    if not windows:
        return 0.0
    return max(window.end_sec for window in windows)


def _has_legacy_provider_artifacts(artifacts_root: Path) -> bool:
    provider_dir = artifacts_root / "providers"
    return provider_dir.exists() and any(provider_dir.glob("*.json"))


def _validate_chunk_provider_completeness(
    audio_id: str,
    chunk_index: int,
    expected_providers: list[str],
    chunk_transcripts: list[ProviderTranscript],
) -> None:
    found = {transcript.provider_name for transcript in chunk_transcripts}
    missing = sorted(set(expected_providers) - found)
    if missing:
        raise RuntimeError(
            f"Missing provider transcript artifacts for audio_id={audio_id} chunk={chunk_index:03d}: "
            f"{', '.join(missing)}"
        )
