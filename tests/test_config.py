import pytest

from metatranscribe.config import load_settings


def test_load_settings_export_publish_dir_unset(monkeypatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env.empty"
    dotenv_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("EXPORT_PUBLISH_DIR", raising=False)

    settings = load_settings(dotenv_path=str(dotenv_path))

    assert settings.export_publish_dir is None


def test_load_settings_export_publish_dir_set(monkeypatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env.empty"
    dotenv_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("EXPORT_PUBLISH_DIR", str(tmp_path / "published"))

    settings = load_settings(dotenv_path=str(dotenv_path))

    assert settings.export_publish_dir == (tmp_path / "published").resolve()
    assert settings.export_publish_dir.exists()


def test_load_settings_chunk_overlap_defaults_to_zero(monkeypatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env.empty"
    dotenv_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TRANSCRIBE_CHUNK_SECONDS", "540")
    monkeypatch.delenv("TRANSCRIBE_CHUNK_OVERLAP_SECONDS", raising=False)

    settings = load_settings(dotenv_path=str(dotenv_path))

    assert settings.transcribe_chunk_overlap_seconds == 0


def test_load_settings_chunk_overlap_rejects_negative(monkeypatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env.empty"
    dotenv_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TRANSCRIBE_CHUNK_SECONDS", "540")
    monkeypatch.setenv("TRANSCRIBE_CHUNK_OVERLAP_SECONDS", "-1")

    with pytest.raises(ValueError, match="TRANSCRIBE_CHUNK_OVERLAP_SECONDS must be >= 0"):
        load_settings(dotenv_path=str(dotenv_path))


def test_load_settings_chunk_overlap_rejects_chunk_size_or_more(monkeypatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env.empty"
    dotenv_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TRANSCRIBE_CHUNK_SECONDS", "540")
    monkeypatch.setenv("TRANSCRIBE_CHUNK_OVERLAP_SECONDS", "540")

    with pytest.raises(ValueError, match="TRANSCRIBE_CHUNK_OVERLAP_SECONDS must be < TRANSCRIBE_CHUNK_SECONDS"):
        load_settings(dotenv_path=str(dotenv_path))
