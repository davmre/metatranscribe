import io
import json
from pathlib import Path

import pytest

from metatranscribe.config import Settings
from metatranscribe.state.store import StateStore
from metatranscribe.web.app import create_app


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
        transcribe_chunk_overlap_seconds=0,
        reconciler_model="gpt-5",
        reconciler_provider="openai",
        polish_provider="openai",
        polish_model="gpt-5",
        polish_long_silence_seconds=90,
        export_publish_dir=None,
        log_level="INFO",
        pipeline_version="0.1.0",
        web_password="hunter2",
        web_secret_key="test-secret-key",
        web_host="127.0.0.1",
        web_port=8080,
        web_max_upload_mb=512,
        web_session_cookie_secure=False,
    )


@pytest.fixture
def app(tmp_path):
    settings = _settings(tmp_path)
    application = create_app(settings, start_worker=False)
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client):
    return client.post("/login", data={"password": "hunter2"}, follow_redirects=False)


def test_healthz_is_public(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.data == b"ok"


def test_index_requires_login(client):
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_login_then_index_shows_upload_form(client):
    _login(client)
    resp = client.get("/")
    assert resp.status_code == 200
    assert b'type="file"' in resp.data
    assert b'accept="audio/' in resp.data


def test_upload_ingests_and_creates_record(client, app, monkeypatch):
    enqueued = {"count": 0}
    monkeypatch.setattr(
        "metatranscribe.web.app.PipelineWorker.enqueue",
        lambda self: enqueued.__setitem__("count", enqueued["count"] + 1),
    )
    _login(client)
    data = {"audio": (io.BytesIO(b"fake audio bytes"), "memo.m4a")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "/transcript/" in resp.headers["Location"]
    assert enqueued["count"] == 1

    store = StateStore(_settings_from_app(app))
    records = store.get_records()
    assert len(records) == 1
    assert records[0].status == "ingested"


def test_upload_rejects_unsupported_type(client):
    _login(client)
    data = {"audio": (io.BytesIO(b"not audio"), "notes.txt")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Unsupported file type" in resp.data


def test_transcript_detail_renders_exported_markdown(client, app):
    settings = app.config["SETTINGS"]
    audio_id = "abc123"
    store = StateStore(settings.state_db_path)
    from metatranscribe.models import AudioFileRecord

    store.insert_audio_if_new(
        AudioFileRecord(audio_id=audio_id, source_path="memo.m4a", source_hash="h1", status="exported")
    )
    final_dir = settings.output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / f"{audio_id}.json").write_text(
        json.dumps({"title": "My Memo", "duration_sec": 65.0}), encoding="utf-8"
    )
    (final_dir / f"{audio_id}.md").write_text(
        "---\naudio_id: abc123\n---\n\n# Heading\n\nHello world body.\n", encoding="utf-8"
    )

    _login(client)
    resp = client.get(f"/transcript/{audio_id}")
    assert resp.status_code == 200
    assert b"My Memo" in resp.data
    assert b"<h1>Heading</h1>" in resp.data
    assert b"Hello world body." in resp.data
    # Copy/download affordances present, with the raw markdown source embedded.
    assert b"id=\"copy-btn\"" in resp.data
    assert b"/transcript/abc123/download" in resp.data

    download = client.get(f"/transcript/{audio_id}/download")
    assert download.status_code == 200
    assert download.mimetype == "text/markdown"
    assert "My_Memo.md" in download.headers["Content-Disposition"]
    body = download.data.decode("utf-8")
    assert "# Heading" in body
    assert "Hello world body." in body
    assert "audio_id: abc123" not in body  # frontmatter stripped


def test_download_404_when_not_ready(client, app):
    settings = app.config["SETTINGS"]
    store = StateStore(settings.state_db_path)
    from metatranscribe.models import AudioFileRecord

    store.insert_audio_if_new(
        AudioFileRecord(audio_id="pend01", source_path="x.m4a", source_hash="hp", status="ingested")
    )
    _login(client)
    assert client.get("/transcript/pend01/download").status_code == 404


def test_transcript_markdown_is_sanitized(client, app):
    settings = app.config["SETTINGS"]
    audio_id = "xss001"
    store = StateStore(settings.state_db_path)
    from metatranscribe.models import AudioFileRecord

    store.insert_audio_if_new(
        AudioFileRecord(audio_id=audio_id, source_path="x.m4a", source_hash="h2", status="exported")
    )
    final_dir = settings.output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / f"{audio_id}.json").write_text(json.dumps({"title": "X", "duration_sec": 1.0}), encoding="utf-8")
    (final_dir / f"{audio_id}.md").write_text("---\n---\n\n<script>alert(1)</script>\n", encoding="utf-8")

    _login(client)
    resp = client.get(f"/transcript/{audio_id}")
    assert resp.status_code == 200
    assert b"<script>alert(1)</script>" not in resp.data
    assert b"&lt;script&gt;" in resp.data


def test_status_endpoint_returns_json(client, app):
    settings = app.config["SETTINGS"]
    audio_id = "stat01"
    store = StateStore(settings.state_db_path)
    from metatranscribe.models import AudioFileRecord

    store.insert_audio_if_new(
        AudioFileRecord(audio_id=audio_id, source_path="x.m4a", source_hash="h3", status="transcribed")
    )
    _login(client)
    resp = client.get(f"/status/{audio_id}")
    assert resp.status_code == 200
    assert resp.get_json() == {"audio_id": audio_id, "status": "transcribed", "error": None}


def _settings_from_app(app) -> Path:
    return app.config["SETTINGS"].state_db_path
