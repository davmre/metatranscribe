from pathlib import Path

import pytest
from requests import HTTPError

from recorder_transcribe.transcribe.openai_adapter import OpenAITranscriptionProvider, _build_form_data


def test_build_form_data_whisper_uses_verbose_json() -> None:
    data = _build_form_data("whisper-1", "en")
    assert data["response_format"] == "verbose_json"
    assert data["timestamp_granularities[]"] == "segment"


def test_build_form_data_gpt_uses_json() -> None:
    data = _build_form_data("gpt-4o-transcribe", "en")
    assert data["response_format"] == "json"
    assert "timestamp_granularities[]" not in data


def test_transcribe_surfaces_response_body_on_http_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyResponse:
        status_code = 400
        text = '{"error":"bad param"}'

        def raise_for_status(self) -> None:
            raise HTTPError("400 Client Error", response=self)

        def json(self):
            return {}

    def fake_post(*args, **kwargs):  # noqa: ANN002, ANN003
        return DummyResponse()

    monkeypatch.setattr("recorder_transcribe.transcribe.openai_adapter.requests.post", fake_post)

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"wav")

    provider = OpenAITranscriptionProvider("k", "gpt-4o-transcribe")
    with pytest.raises(HTTPError, match="Response body"):
        provider.transcribe("id", audio, "en")
