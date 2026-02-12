import json
from pathlib import Path

from recorder_transcribe.models import ProviderTranscript, Segment
from recorder_transcribe.reconcile.llm_reconciler import LLMReconciler


class _DummyReconciler(LLMReconciler):
    def _call_model(self, prompt: str) -> str:
        return '{"audio_id":"id","title":"t","language":"en","duration_sec":1,"segments":[],"final_text":"","provenance":{}}'


def test_reconciler_writes_io_artifacts(tmp_path: Path) -> None:
    reconciler = _DummyReconciler("k", "m", provider="openai", artifacts_dir=tmp_path)
    transcript = ProviderTranscript(
        provider_name="openai",
        audio_id="id",
        language="en",
        duration_sec=1,
        segments=[Segment(start_sec=0, end_sec=1, text="x")],
        raw_text="x",
        confidence_summary=None,
        raw_payload_path="",
    )

    canonical = reconciler.reconcile("id", [transcript], "en")
    assert canonical.audio_id == "id"

    assert (tmp_path / "request_prompt.json").exists()
    assert (tmp_path / "response_raw.txt").exists()
    parsed = json.loads((tmp_path / "response_parsed.json").read_text(encoding="utf-8"))
    assert parsed["audio_id"] == "id"
