import json

import pytest

from recorder_transcribe.reconcile.llm_reconciler import _extract_json_payload


def test_extract_json_payload_plain_object() -> None:
    payload = _extract_json_payload('{"a":1}')
    assert payload == {"a": 1}


def test_extract_json_payload_markdown_fence() -> None:
    payload = _extract_json_payload('```json\n{"a":1}\n```')
    assert payload == {"a": 1}


def test_extract_json_payload_with_prose_wrapper() -> None:
    payload = _extract_json_payload('Here is your JSON:\n{"a":1,"b":2}\nThanks')
    assert payload == {"a": 1, "b": 2}


def test_extract_json_payload_raises_on_empty() -> None:
    with pytest.raises(json.JSONDecodeError):
        _extract_json_payload("")
