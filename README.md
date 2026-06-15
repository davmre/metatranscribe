# MetaTranscribe Pipeline

Modular pipeline to process arbitrary audio files and generate high-quality, timestamped transcripts.

## Features
- Manual inbox ingest (`data/inbox`) with SHA256 dedupe
- Multi-provider transcription (OpenAI, Deepgram, optional local Whisper)
- Chunked transcription for long recordings (prevents oversized API uploads)
- Chunked LLM reconciliation into canonical transcript timeline (OpenAI or Anthropic)
- Mechanical chunk merge (no extra LLM merge pass)
- Optional single LLM polish pass to generate final Markdown
- Canonical JSON + polished Markdown outputs per note
- Idempotent sqlite state tracking and rerunnable steps
- Optional password-protected web interface for uploading and browsing transcripts from a phone

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
python scripts/reconcile.py --audio-id <audio_id> --dry-run
python scripts/export.py --audio-id <audio_id>
python scripts/reexport.py --audio-id <audio_id>
python scripts/reexport.py --audio-id <audio_id> --dry-run
```

Batch mode by status:
```bash
python scripts/transcribe.py
python scripts/reconcile.py
python scripts/export.py
python scripts/reexport.py --all
python scripts/reexport.py --all --dry-run
```

`reexport.py` reruns only final export/polish using existing `canonical.json` artifacts.

## Run End-to-End
```bash
python scripts/run_pipeline.py
```

## Output
- Final markdown: `outputs/final/<audio_id>.md`
- Final JSON: `outputs/final/<audio_id>.json`
- Optional published markdown copy: `<EXPORT_PUBLISH_DIR>/YYYY-MM-DD_<llm_suggested_name>.md`
- Provider chunk artifacts: `outputs/artifacts/<audio_id>/transcribe/chunks/<idx>/providers/*.json`
- Canonical artifact: `outputs/artifacts/<audio_id>/canonical.json`
- Reconciliation I/O artifacts: `outputs/artifacts/<audio_id>/reconcile/*`
- Polish I/O artifacts: `outputs/artifacts/<audio_id>/polish/*`

## Web Interface
A minimal, mobile-friendly web UI lets you upload a recording, watch its status, and browse
previous transcripts. It is single-user and protected by one shared password.

1. Install the web extra:
   ```bash
   pip install -e ".[dev,web]"
   ```
2. Set the web variables in `.env` (see `.env.example`):
   - `WEB_PASSWORD` — the shared login password.
   - `WEB_SECRET_KEY` — a fixed random string used to sign session cookies. **Keep it stable**;
     changing it logs you out. Generate one with `python -c "import secrets; print(secrets.token_hex(32))"`.
   - `WEB_HOST` (default `127.0.0.1`), `WEB_PORT` (default `8080`), `WEB_MAX_UPLOAD_MB` (default `512`).
   - `WEB_SESSION_COOKIE_SECURE` (default `true`) — set `false` only for local `http://` testing.
3. Run it (served by Waitress, a production WSGI server):
   ```bash
   python scripts/serve_web.py
   ```

Uploads land in `data/inbox/` and are processed by a single background worker that runs the
normal pipeline (`ingest → transcribe → reconcile → export`); the page polls and refreshes when
a transcript is ready. The worker also resumes any in-flight records on startup.

### Public deployment
The app speaks plain HTTP and is meant to run **behind a reverse proxy (nginx/Caddy) that
terminates HTTPS** — a password and session cookie cross the wire, so TLS is required. Keep
`WEB_HOST=127.0.0.1` and let the proxy forward to it. Remember to:
- Raise the proxy's max body size to at least `WEB_MAX_UPLOAD_MB` (e.g. nginx `client_max_body_size`)
  and increase read/proxy timeouts for large uploads.
- Run under a process manager (e.g. systemd) so the server restarts on crash.
- `GET /healthz` is unauthenticated for uptime checks.

## Cron Example
Runs every 4 hours:
```cron
0 */4 * * * cd /Users/dave/code/metatranscribe && /Users/dave/code/metatranscribe/.venv/bin/python scripts/run_pipeline.py >> logs/cron.log 2>&1
```

## Notes
- `ffmpeg`/`ffprobe` are optional but recommended for normalization and accurate durations.
- Transcription chunk size is configurable with `TRANSCRIBE_CHUNK_SECONDS` (default 540s).
- Optional chunk overlap is configurable with `TRANSCRIBE_CHUNK_OVERLAP_SECONDS` (default 0s, suggested 5s to protect boundary words).
- Reconciliation always uses transcription chunk boundaries from `transcribe_chunks.json`.
- Reconciliation provider is configurable via `RECONCILER_PROVIDER=openai|anthropic`.
- `scripts/reconcile.py --dry-run` writes per-chunk `request_prompt.json` artifacts only (no LLM call, no canonical write, no status update).
- Export always runs the polish pass to generate human-readable Markdown.
- Polish provider/model are configurable via `POLISH_PROVIDER` and `POLISH_MODEL`.
- If `EXPORT_PUBLISH_DIR` is set, export also copies markdown to that folder using `YYYY-MM-DD_<llm_suggested_name>.md` and appends `_2`, `_3`, etc. on collisions.
- `scripts/reexport.py --dry-run` writes `polish/request_prompt.json` only (no LLM call, no final output writes, no status update).
- Increase verbosity with `LOG_LEVEL=DEBUG` in `.env`.
- Silence markers are generated algorithmically from timeline gaps (default threshold 90 seconds).
- Failed records stop auto-retrying once `retry_count >= MAX_RETRIES`.
