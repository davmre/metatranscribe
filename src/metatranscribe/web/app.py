from __future__ import annotations

import logging
import os
import tempfile
from datetime import timedelta
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from metatranscribe.config import Settings, load_settings, validate_web_credentials
from metatranscribe.ingest.manual_inbox import AUDIO_EXTENSIONS, ingest_new_files
from metatranscribe.state.store import StateStore
from metatranscribe.web import transcripts
from metatranscribe.web.auth import check_password, login_required
from metatranscribe.web.worker import PipelineWorker

logger = logging.getLogger(__name__)


def _format_duration(seconds: float | None) -> str:
    if not seconds or seconds <= 0:
        return ""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _store(app: Flask) -> StateStore:
    return app.config["STATE_STORE"]


def _settings(app: Flask) -> Settings:
    return app.config["SETTINGS"]


def _worker(app: Flask) -> PipelineWorker:
    return app.config["WORKER"]


def _unique_inbox_path(inbox_dir: Path, filename: str) -> Path:
    candidate = inbox_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    counter = 2
    while True:
        candidate = inbox_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def create_app(settings: Settings | None = None, *, start_worker: bool = True) -> Flask:
    settings = settings or load_settings()
    validate_web_credentials(settings)
    settings.ensure_dirs()

    app = Flask(__name__)
    app.config.update(
        SETTINGS=settings,
        WEB_PASSWORD=settings.web_password,
        SECRET_KEY=settings.web_secret_key,
        MAX_CONTENT_LENGTH=settings.web_max_upload_mb * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.web_session_cookie_secure,
        PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    )
    app.config["STATE_STORE"] = StateStore(settings.state_db_path)
    worker = PipelineWorker(settings)
    app.config["WORKER"] = worker
    if start_worker:
        worker.start()

    app.jinja_env.filters["duration"] = _format_duration

    _register_routes(app)
    return app


def _register_routes(app: Flask) -> None:
    @app.route("/healthz")
    def healthz():
        return "ok", 200

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            if check_password(request.form.get("password", "")):
                session.clear()
                session["authed"] = True
                session.permanent = True
                target = request.args.get("next") or url_for("index")
                if not target.startswith("/"):
                    target = url_for("index")
                return redirect(target)
            flash("Incorrect password.")
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/")
    @login_required
    def index():
        items = transcripts.list_items(_settings(app), _store(app))
        return render_template("index.html", items=items)

    @app.route("/upload", methods=["POST"])
    @login_required
    def upload():
        settings = _settings(app)
        file = request.files.get("audio")
        if file is None or not file.filename:
            flash("No file selected.")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        if not filename or Path(filename).suffix.lower() not in AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(AUDIO_EXTENSIONS))
            flash(f"Unsupported file type. Supported audio formats: {allowed}.")
            return redirect(url_for("index"))

        settings.inbox_dir.mkdir(parents=True, exist_ok=True)
        destination = _unique_inbox_path(settings.inbox_dir, filename)
        # Write to a temp file in the same dir then atomically move into place so
        # ingest never picks up a partially written upload.
        fd, tmp_name = tempfile.mkstemp(dir=settings.inbox_dir, suffix=Path(filename).suffix)
        os.close(fd)
        tmp_path = Path(tmp_name)
        file.save(tmp_path)
        os.replace(tmp_path, destination)

        new_ids = ingest_new_files(settings.inbox_dir, settings.data_root / "raw", _store(app))
        _worker(app).enqueue()

        if new_ids:
            return redirect(url_for("transcript", audio_id=new_ids[0]))
        flash("This recording was already uploaded.")
        return redirect(url_for("index"))

    @app.route("/transcript/<audio_id>")
    @login_required
    def transcript(audio_id: str):
        detail = transcripts.load_detail(_settings(app), _store(app), audio_id)
        if detail is None:
            abort(404)
        return render_template("detail.html", detail=detail)

    @app.route("/status/<audio_id>")
    @login_required
    def status(audio_id: str):
        record = _store(app).get_record(audio_id)
        if record is None:
            abort(404)
        return jsonify({"audio_id": audio_id, "status": record.status, "error": record.error})

    @app.errorhandler(413)
    def too_large(_error):
        settings = _settings(app)
        return (
            render_template("error.html", message=f"File too large (limit {settings.web_max_upload_mb} MB)."),
            413,
        )
