from __future__ import annotations

import json
from pathlib import Path

from recorder_transcribe.models import CanonicalTranscript


def write_outputs(canonical: CanonicalTranscript, markdown_path: Path, json_path: Path, markdown_text: str) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    json_path.write_text(json.dumps(canonical.model_dump(), indent=2), encoding="utf-8")
