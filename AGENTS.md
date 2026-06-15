# AGENTS.md

## Purpose
This repository implements a modular transcription pipeline for arbitrary audio files, optimized for long-form notes and high-quality final transcripts.

The pipeline is designed to:
- ingest audio files from a local inbox,
- transcribe with multiple providers,
- reconcile transcripts in chunks,
- mechanically merge to canonical timeline JSON,
- run a final LLM polish pass to produce readable Markdown.

## High-Level Flow
1. `ingest`:
- Scan `data/inbox/`, hash files, dedupe into sqlite state.
- Copy originals into `data/raw/<audio_id>/original.*`.

2. `transcribe`:
- Normalize audio (`ffmpeg` if available).
- Split into transcription chunks (`TRANSCRIBE_CHUNK_SECONDS`) with optional overlap (`TRANSCRIBE_CHUNK_OVERLAP_SECONDS`).
- Call each provider per chunk.
- Save per-chunk provider artifacts to `outputs/artifacts/<audio_id>/transcribe/chunks/<idx>/providers/*.json`.
- Save chunk manifest to `outputs/artifacts/<audio_id>/transcribe_chunks.json`.

3. `reconcile`:
- Load provider transcripts directly from transcription chunk artifacts.
- Reconcile in chunks using the transcription chunk boundaries.
- Save per-chunk reconcile artifacts under `outputs/artifacts/<audio_id>/reconcile/chunks/*`.
- Mechanically merge chunk canonicals into `outputs/artifacts/<audio_id>/canonical.json`.

4. `export`:
- Load canonical timeline JSON.
- Optional LLM polish pass creates human-readable markdown body.
- Write final outputs:
  - `outputs/final/<audio_id>.md`
  - `outputs/final/<audio_id>.json`

## Entry Points (Scripts)
- `scripts/ingest.py`
- `scripts/transcribe.py`
- `scripts/reconcile.py`
- `scripts/export.py`
- `scripts/reexport.py` (re-run export/polish only from existing canonical artifacts)
- `scripts/run_pipeline.py` (full orchestrated run)
- `scripts/serve_web.py` (password-protected web UI for upload/browse; serves via Waitress)

## Core Modules
- `src/metatranscribe/orchestrator.py`
  - Main step orchestration and status transitions.
- `src/metatranscribe/config.py`
  - Env-backed settings and provider credential validation.
- `src/metatranscribe/state/store.py`
  - sqlite state store (`audio_files`, `provider_runs`).
- `src/metatranscribe/transcribe/*`
  - Provider adapters and transcript normalization.
- `src/metatranscribe/reconcile/*`
  - Prompt construction, LLM reconcile client, chunking + merge utilities.
- `src/metatranscribe/output/*`
  - Markdown rendering and polish pass.
- `src/metatranscribe/preprocess/audio_prep.py`
  - ffmpeg normalization, probing, and chunk splitting.
- `src/metatranscribe/web/*`
  - Flask app factory, session-cookie auth, background pipeline worker, and transcript read helpers.

## Artifacts and Debugging
For each `audio_id`, inspect:
- Provider outputs:
  - `outputs/artifacts/<audio_id>/transcribe/chunks/<idx>/providers/*.json`
- Transcription chunk boundaries:
  - `outputs/artifacts/<audio_id>/transcribe_chunks.json`
- Reconciliation I/O:
  - `outputs/artifacts/<audio_id>/reconcile/chunks/<idx>/request_prompt.json`
  - `outputs/artifacts/<audio_id>/reconcile/chunks/<idx>/response_raw.txt`
  - `outputs/artifacts/<audio_id>/reconcile/chunks/<idx>/response_parsed.json`
- Canonical timeline:
  - `outputs/artifacts/<audio_id>/canonical.json`
- Polish I/O:
  - `outputs/artifacts/<audio_id>/polish/request_prompt.json`
  - `outputs/artifacts/<audio_id>/polish/response_raw.md`

Logs:
- `logs/pipeline.log`
- Set `LOG_LEVEL=DEBUG` for verbose progress and API timing.

## Important Environment Variables
Transcription:
- `TRANSCRIBE_PROVIDERS` (e.g. `openai,deepgram`)
- `TRANSCRIBE_OPENAI_MODEL`
- `TRANSCRIBE_DEEPGRAM_MODEL`
- `TRANSCRIBE_CHUNK_SECONDS`
- `TRANSCRIBE_CHUNK_OVERLAP_SECONDS`

Reconciliation:
- `RECONCILER_PROVIDER` (`openai|anthropic`)
- `RECONCILER_MODEL`

Polish:
- `POLISH_PROVIDER`
- `POLISH_MODEL`
- `POLISH_LONG_SILENCE_SECONDS`

General:
- `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`
- `DATA_ROOT`, `OUTPUT_ROOT`, `STATE_DB_PATH`, `INBOX_DIR`, `LOGS_DIR`

## Test and Validation
Run:
```bash
.venv/bin/python -m pytest -q
```

Current suite validates:
- provider error handling and parameter behavior,
- chunking/merge behavior,
- reconciler parsing and artifact writing,
- state store behavior,
- markdown output rendering.

## Extension Guidelines
When modifying pipeline behavior:
1. Preserve `canonical.json` as the debuggable source of truth.
2. Keep external API calls logged with model/provider and elapsed time.
3. Prefer additive config flags over hard behavioral switches.
4. Add/adjust tests for any change in chunking, merging, or rendering logic.
5. Keep artifact contracts stable unless there is a clear migration path.
