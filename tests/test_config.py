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
