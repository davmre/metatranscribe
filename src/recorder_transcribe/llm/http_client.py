from __future__ import annotations

from typing import Any

import requests
from requests import HTTPError


def raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except HTTPError as exc:
        detail = response.text.strip()
        raise HTTPError(f"{exc}. Response body: {detail}", response=response) from exc


def post_openai_chat_completion(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout: int = 600,
    response_format: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if response_format is not None:
        body["response_format"] = response_format
    response = requests.post(url, headers=headers, json=body, timeout=timeout)
    raise_for_status_with_body(response)
    return response.json()


def extract_openai_message_text(payload: dict[str, Any]) -> str:
    return str(payload["choices"][0]["message"]["content"])


def post_anthropic_message(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int = 600,
) -> dict[str, Any]:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    response = requests.post(url, headers=headers, json=body, timeout=timeout)
    raise_for_status_with_body(response)
    return response.json()


def extract_anthropic_text(payload: dict[str, Any]) -> str:
    parts = payload.get("content", [])
    texts = [part.get("text", "") for part in parts if part.get("type") == "text"]
    return "\n".join(texts).strip()
