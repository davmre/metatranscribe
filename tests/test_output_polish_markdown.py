import json
from pathlib import Path

from metatranscribe.models import CanonicalTranscript
from metatranscribe.output.polish_markdown import render_polished_markdown


def _canonical() -> CanonicalTranscript:
    return CanonicalTranscript(
        audio_id="a1",
        title="Project Update",
        language="en",
        duration_sec=12.0,
        segments=[],
        silence_markers=[],
        final_text="",
        provenance={},
    )


def test_render_polished_markdown_parses_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "metatranscribe.output.polish_markdown._call_model",
        lambda provider, model, api_key, prompt: json.dumps(
            {"suggested_name": "Weekly Sync", "markdown": "# Heading\n\nBody"}
        ),
    )

    result = render_polished_markdown(
        canonical=_canonical(),
        provider="openai",
        model="gpt-5",
        api_key="k",
        long_silence_seconds=90,
        artifacts_dir=tmp_path,
    )

    assert result.suggested_name == "Weekly Sync"
    assert result.markdown == "# Heading\n\nBody\n"
    parsed = json.loads((tmp_path / "response_parsed.json").read_text(encoding="utf-8"))
    assert parsed["suggested_name_used_fallback"] is False


def test_render_polished_markdown_parses_fenced_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "metatranscribe.output.polish_markdown._call_model",
        lambda provider, model, api_key, prompt: '```json\n{"suggested_name":"Team Notes","markdown":"Body"}\n```',
    )

    result = render_polished_markdown(
        canonical=_canonical(),
        provider="openai",
        model="gpt-5",
        api_key="k",
        long_silence_seconds=90,
        artifacts_dir=tmp_path,
    )

    assert result.suggested_name == "Team Notes"
    assert result.markdown.startswith("# Project Update")


def test_render_polished_markdown_fallbacks_on_invalid_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "metatranscribe.output.polish_markdown._call_model",
        lambda provider, model, api_key, prompt: "not-json response",
    )

    result = render_polished_markdown(
        canonical=_canonical(),
        provider="openai",
        model="gpt-5",
        api_key="k",
        long_silence_seconds=90,
        artifacts_dir=tmp_path,
    )

    assert result.suggested_name == "Project Update"
    assert result.markdown == "# Project Update\n\nnot-json response\n"
    parsed = json.loads((tmp_path / "response_parsed.json").read_text(encoding="utf-8"))
    assert parsed["suggested_name_used_fallback"] is True
