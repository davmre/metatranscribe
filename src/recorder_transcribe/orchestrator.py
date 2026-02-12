from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from recorder_transcribe.config import (
    Settings,
    resolve_llm_api_key,
    validate_polish_credentials,
    validate_reconciler_credentials,
)
from recorder_transcribe.ingest.manual_inbox import get_original_audio_path, ingest_new_files
from recorder_transcribe.models import CanonicalTranscript, ProviderTranscript, Segment
from recorder_transcribe.output.polish_markdown import render_polished_markdown
from recorder_transcribe.output.markdown_writer import render_markdown, write_outputs
from recorder_transcribe.postprocess.silence_annotations import build_silence_markers
from recorder_transcribe.preprocess.audio_prep import (
    AudioChunk,
    has_ffmpeg,
    normalize_audio,
    probe_duration_sec,
    split_audio_into_chunks,
)
from recorder_transcribe.reconcile.chunking import (
    ChunkWindow,
    build_windows,
    estimate_duration_sec,
    merge_chunk_canonicals,
    slice_transcripts_for_window,
)
from recorder_transcribe.reconcile.llm_reconciler import LLMReconciler, save_canonical_transcript
from recorder_transcribe.state.store import StateStore
from recorder_transcribe.transcribe.runner import build_providers, load_provider_transcripts, save_provider_transcript

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
    )
    logger.info(
        "Prepared transcription chunks audio_id=%s count=%d duration_sec=%.2f chunk_sec=%d",
        audio_id,
        len(chunks),
        duration,
        settings.transcribe_chunk_seconds,
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
    artifacts_dir = settings.output_root / "artifacts" / audio_id / "providers"

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
        chunk_results = []
        for chunk_index, chunk in enumerate(chunks):
            logger.info(
                "Transcribing chunk audio_id=%s provider=%s chunk=%d start=%.2f end=%.2f",
                audio_id,
                provider.name,
                chunk_index,
                chunk.start_sec,
                chunk.end_sec,
            )
            chunk_audio_id = f"{audio_id}__chunk_{chunk_index:03d}"
            chunk_results.append(provider.transcribe(chunk_audio_id, chunk.path, settings.language_hint))

        transcript = _merge_provider_chunk_transcripts(
            audio_id=audio_id,
            provider_name=provider.name,
            chunks=chunks,
            chunk_transcripts=chunk_results,
            language_hint=settings.language_hint,
            total_duration_sec=duration,
        )
        path = save_provider_transcript(transcript, artifacts_dir)
        transcript.raw_payload_path = str(path)
        path.write_text(json.dumps(transcript.model_dump(), indent=2), encoding="utf-8")
        store.save_provider_run(
            audio_id,
            transcript.provider_name,
            str(path),
            summary={"duration_sec": transcript.duration_sec, "segments": len(transcript.segments)},
        )

    store.update_status(audio_id, "transcribed")
    logger.info("Transcription step completed audio_id=%s providers=%d", audio_id, len(providers))


def reconcile_step(settings: Settings, store: StateStore, audio_id: str, dry_run: bool = False) -> CanonicalTranscript:
    logger.info(
        "Reconciliation step started audio_id=%s provider=%s model=%s",
        audio_id,
        settings.reconciler_provider,
        settings.reconciler_model,
    )
    provider_dir = settings.output_root / "artifacts" / audio_id / "providers"
    transcripts = load_provider_transcripts(provider_dir)
    if not transcripts:
        raise RuntimeError(f"No provider transcripts found for {audio_id}")
    validate_reconciler_credentials(settings)
    reconciler_provider = settings.reconciler_provider.strip().lower()
    api_key = resolve_llm_api_key(settings, reconciler_provider)

    reconcile_artifacts_dir = settings.output_root / "artifacts" / audio_id / "reconcile"
    duration_sec = estimate_duration_sec(transcripts)
    windows = _load_transcription_chunk_windows(settings.output_root / "artifacts" / audio_id)
    if windows and settings.use_transcribe_chunks_for_reconcile:
        logger.info(
            "Using transcription chunk windows for reconciliation audio_id=%s windows=%d",
            audio_id,
            len(windows),
        )
    else:
        windows = build_windows(
            duration_sec=duration_sec,
            chunk_seconds=settings.reconcile_chunk_seconds,
            overlap_seconds=settings.reconcile_chunk_overlap_seconds,
        )
    logger.info(
        "Chunked reconciliation planned audio_id=%s windows=%d duration_sec=%.2f chunk_sec=%d overlap_sec=%d",
        audio_id,
        len(windows),
        duration_sec,
        settings.reconcile_chunk_seconds,
        settings.reconcile_chunk_overlap_seconds,
    )
    chunk_results: list[CanonicalTranscript] = []
    for window in windows:
        chunk_id = f"{audio_id}__chunk_{window.index:03d}"
        chunk_dir = reconcile_artifacts_dir / "chunks" / f"{window.index:03d}"
        window_transcripts = slice_transcripts_for_window(transcripts, window)
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

    canonical = merge_chunk_canonicals(audio_id, chunk_results, duration_sec)

    canonical.silence_markers = build_silence_markers(canonical.segments, settings.silence_gap_seconds)
    canonical_path = settings.output_root / "artifacts" / audio_id / "canonical.json"
    save_canonical_transcript(canonical, canonical_path)
    store.update_status(audio_id, "reconciled")
    logger.info("Reconciliation step completed audio_id=%s segments=%d", audio_id, len(canonical.segments))
    return canonical


def export_step(settings: Settings, store: StateStore, audio_id: str) -> None:
    logger.info("Export step started audio_id=%s", audio_id)
    canonical_path = settings.output_root / "artifacts" / audio_id / "canonical.json"
    payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical = CanonicalTranscript.model_validate(payload)

    source_file = get_original_audio_path(settings.data_root / "raw", audio_id).name
    providers = [t.provider_name for t in load_provider_transcripts(settings.output_root / "artifacts" / audio_id / "providers")]
    if settings.enable_polish_pass:
        validate_polish_credentials(settings)
        polish_provider = settings.polish_provider.strip().lower()
        polish_api_key = resolve_llm_api_key(settings, polish_provider)
        markdown_body = render_polished_markdown(
            canonical=canonical,
            provider=polish_provider,
            model=settings.polish_model,
            api_key=polish_api_key,
            long_silence_seconds=settings.polish_long_silence_seconds,
            artifacts_dir=settings.output_root / "artifacts" / audio_id / "polish",
        )
        markdown = _render_markdown_with_frontmatter(
            body=markdown_body,
            canonical=canonical,
            source_file=source_file,
            providers=providers,
            reconciler_model=settings.reconciler_model,
            pipeline_version=settings.pipeline_version,
        )
    else:
        markdown = render_markdown(
            canonical,
            source_file=source_file,
            providers=providers,
            reconciler_model=settings.reconciler_model,
            pipeline_version=settings.pipeline_version,
            output_style=settings.output_style,
            paragraph_gap_seconds=settings.paragraph_gap_seconds,
            paragraph_max_chars=settings.paragraph_max_chars,
        )

    final_md = settings.output_root / "final" / f"{audio_id}.md"
    final_json = settings.output_root / "final" / f"{audio_id}.json"
    write_outputs(canonical, final_md, final_json, markdown)
    store.update_status(audio_id, "exported")
    logger.info("Export step completed audio_id=%s output=%s", audio_id, final_md)


def run_pipeline(settings: Settings) -> tuple[int, int]:
    store = StateStore(settings.state_db_path)
    ingest_step(settings, store)

    records = [r for r in store.get_records() if r.status in {"ingested", "failed", "transcribed", "reconciled"}]
    succeeded = 0
    failed = 0

    for record in records:
        logger.info("Processing audio_id=%s current_status=%s", record.audio_id, record.status)
        try:
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


def run_summary(succeeded: int, failed: int) -> dict[str, str | int]:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "succeeded": succeeded,
        "failed": failed,
    }


def _merge_provider_chunk_transcripts(
    audio_id: str,
    provider_name: str,
    chunks: list[AudioChunk],
    chunk_transcripts: list[ProviderTranscript],
    language_hint: str,
    total_duration_sec: float,
) -> ProviderTranscript:
    merged_segments: list[Segment] = []
    merged_text_parts: list[str] = []
    language = language_hint
    confidence_values: list[float] = []

    for chunk, chunk_result in zip(chunks, chunk_transcripts):
        language = chunk_result.language or language
        chunk_text = chunk_result.raw_text.strip()
        if chunk_text:
            merged_text_parts.append(chunk_text)

        timed = [s for s in chunk_result.segments if s.end_sec > s.start_sec]
        if timed:
            for seg in timed:
                merged_segments.append(
                    Segment(
                        start_sec=seg.start_sec + chunk.start_sec,
                        end_sec=seg.end_sec + chunk.start_sec,
                        text=seg.text,
                        confidence=seg.confidence,
                        speaker=seg.speaker,
                    )
                )
                if seg.confidence is not None:
                    confidence_values.append(seg.confidence)
        elif chunk_text:
            merged_segments.append(
                Segment(
                    start_sec=chunk.start_sec,
                    end_sec=max(chunk.end_sec, chunk.start_sec + 0.1),
                    text=chunk_text,
                    confidence=None,
                    speaker=None,
                )
            )

    merged_segments.sort(key=lambda s: (s.start_sec, s.end_sec))
    confidence_summary: dict[str, float] = {}
    if confidence_values:
        confidence_summary["segment_confidence_mean"] = sum(confidence_values) / len(confidence_values)

    return ProviderTranscript(
        provider_name=provider_name,
        audio_id=audio_id,
        language=language,
        duration_sec=total_duration_sec,
        segments=merged_segments,
        raw_text="\n".join(merged_text_parts).strip(),
        confidence_summary=confidence_summary or None,
        raw_payload_path="",
    )


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


def _write_transcription_chunk_manifest(artifacts_root: Path, chunks: list[AudioChunk]) -> None:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "index": idx,
            "start_sec": chunk.start_sec,
            "end_sec": chunk.end_sec,
            "path": str(chunk.path),
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
