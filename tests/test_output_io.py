import json
from pathlib import Path

from metatranscribe.models import CanonicalTranscript
from metatranscribe.output.io import write_outputs


def test_write_outputs_writes_markdown_and_json(tmp_path: Path) -> None:
    canonical = CanonicalTranscript(
        audio_id="abc123",
        title="Title",
        language="en",
        duration_sec=12.0,
        segments=[],
        silence_markers=[],
        final_text="",
        provenance={},
    )
    md_path = tmp_path / "final" / "abc123.md"
    json_path = tmp_path / "final" / "abc123.json"

    write_outputs(canonical, md_path, json_path, "# Title\n\nBody\n")

    assert md_path.read_text(encoding="utf-8") == "# Title\n\nBody\n"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["audio_id"] == "abc123"
