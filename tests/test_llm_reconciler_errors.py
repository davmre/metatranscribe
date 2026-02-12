import pytest
from requests import HTTPError

from recorder_transcribe.llm.http_client import raise_for_status_with_body


class _DummyResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        raise HTTPError(f"{self.status_code} error", response=self)


def test_raise_for_status_with_body_includes_response_payload() -> None:
    response = _DummyResponse(404, '{"error":{"message":"model_not_found"}}')
    with pytest.raises(HTTPError, match="Response body"):
        raise_for_status_with_body(response)
