# Recorder Transcribe Pipeline

Modular pipeline to process Google Pixel Recorder exports and generate high-quality, timestamped transcripts.

## Features
- Manual inbox ingest (`data/inbox`) with SHA256 dedupe
- Multi-provider transcription (OpenAI, Deepgram, optional local Whisper)
- Chunked transcription for long recordings (prevents oversized API uploads)
- Chunked LLM reconciliation into canonical transcript timeline (OpenAI or Anthropic)
- Mechanical chunk merge (no extra LLM merge pass)
- Optional single LLM polish pass to generate final Markdown
- Canonical JSON + polished Markdown outputs per note
- Idempotent sqlite state tracking and rerunnable steps

## Quickstart
1. Create and activate a Python 3.11+ virtualenv.
2. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Configure environment:
   ```bash
   cp .env.example .env
   ```
4. Put exported audio files into `data/inbox/`.

## Run Modules Individually
```bash
python scripts/ingest.py
python scripts/transcribe.py --audio-id <audio_id>
python scripts/reconcile.py --audio-id <audio_id>
python scripts/export.py --audio-id <audio_id>
python scripts/reexport.py --audio-id <audio_id>
```

Batch mode by status:
```bash
python scripts/transcribe.py
python scripts/reconcile.py
python scripts/export.py
python scripts/reexport.py --all
```

`reexport.py` reruns only final export/polish using existing `canonical.json` artifacts.

## Run End-to-End
```bash
python scripts/run_pipeline.py
```

## Output
- Final markdown: `outputs/final/<audio_id>.md`
- Final JSON: `outputs/final/<audio_id>.json`
- Provider chunk artifacts: `outputs/artifacts/<audio_id>/transcribe/chunks/<idx>/providers/*.json`
- Canonical artifact: `outputs/artifacts/<audio_id>/canonical.json`
- Reconciliation I/O artifacts: `outputs/artifacts/<audio_id>/reconcile/*`
- Polish I/O artifacts: `outputs/artifacts/<audio_id>/polish/*`

## Cron Example
Runs every 4 hours:
```cron
0 */4 * * * cd /Users/dave/code/recorder_transcribe && /Users/dave/code/recorder_transcribe/.venv/bin/python scripts/run_pipeline.py >> logs/cron.log 2>&1
```

## Notes
- `ffmpeg`/`ffprobe` are optional but recommended for normalization and accurate durations.
- Transcription chunk size is configurable with `TRANSCRIBE_CHUNK_SECONDS` (default 540s).
- Reconciliation always uses transcription chunk boundaries from `transcribe_chunks.json`.
- Reconciliation provider is configurable via `RECONCILER_PROVIDER=openai|anthropic`.
- Enable/disable polished final output with `ENABLE_POLISH_PASS=true|false`.
- Polish provider/model are configurable via `POLISH_PROVIDER` and `POLISH_MODEL`.
- Increase verbosity with `LOG_LEVEL=DEBUG` in `.env`.
- Final markdown format is configurable with `OUTPUT_STYLE=paragraph|timestamped`.
- Silence markers are generated algorithmically from timeline gaps (default threshold 90 seconds).
