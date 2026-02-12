import pytest

from metatranscribe.config import load_settings, validate_reconciler_credentials
from metatranscribe.reconcile.llm_reconciler import LLMReconciler


def test_validate_reconciler_credentials_openai(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env.empty"
    dotenv_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("RECONCILER_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = load_settings(dotenv_path=str(dotenv_path))
    validate_reconciler_credentials(settings)


def test_validate_reconciler_credentials_anthropic(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env.empty"
    dotenv_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("RECONCILER_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = load_settings(dotenv_path=str(dotenv_path))
    validate_reconciler_credentials(settings)


def test_validate_reconciler_credentials_missing_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env.empty"
    dotenv_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("RECONCILER_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = load_settings(dotenv_path=str(dotenv_path))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        validate_reconciler_credentials(settings)


def test_reconciler_unknown_provider_raises() -> None:
    reconciler = LLMReconciler(api_key="k", model="m", provider="not-real")
    with pytest.raises(ValueError, match="Unsupported reconciler provider"):
        reconciler._call_model("{}")
