from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from metatranscribe.llm.http_client import (
    extract_anthropic_text,
    extract_openai_message_text,
    post_anthropic_message,
    post_openai_chat_completion,
)
from metatranscribe.models import CanonicalTranscript, ProviderTranscript
from metatranscribe.reconcile.prompt_builder import build_reconciliation_prompt


class LLMReconciler:
    def __init__(self, api_key: str, model: str, provider: str = "openai", artifacts_dir: Path | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.provider = provider.strip().lower()
        self.artifacts_dir = artifacts_dir

    def reconcile(
        self,
        audio_id: str,
        transcripts: list[ProviderTranscript],
        language_hint: str,
        dry_run: bool = False,
    ) -> CanonicalTranscript:
        if not transcripts:
            raise ValueError("Cannot reconcile empty transcript list")

        prompt = build_reconciliation_prompt(audio_id, transcripts, language_hint)
        logger = logging.getLogger(__name__)
        logger.info(
            "Reconciliation payload prepared audio_id=%s providers=%d total_provider_segments=%d prompt_chars=%d",
            audio_id,
            len(transcripts),
            sum(len(t.segments) for t in transcripts),
            len(prompt),
        )
        self._write_artifact("request_prompt.json", prompt)
            
        if dry_run:
            return CanonicalTranscript(audio_id=audio_id, title='', language='', duration_sec=0, segments=[], final_text='', provenance={})

        response_text = self._call_model(prompt)
        self._write_artifact("response_raw.txt", response_text)
        payload = self._parse_or_repair_json(prompt, response_text)
        self._write_artifact("response_parsed.json", json.dumps(payload, indent=2))

        canonical = CanonicalTranscript.model_validate(payload)
        
        if canonical.audio_id != audio_id:
            canonical.audio_id = audio_id
        return canonical

    def _call_model(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._call_openai(prompt)
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        raise ValueError(f"Unsupported reconciler provider '{self.provider}'")

    def _call_openai(self, prompt: str) -> str:
        logger = logging.getLogger(__name__)
        logger.info("OpenAI reconciliation request started model=%s", self.model)
        started = time.perf_counter()
        payload = post_openai_chat_completion(
            api_key=self.api_key,
            model=self.model,
            system_prompt="Let's reconcile some transcriptions!",
            user_prompt=prompt,
            temperature=0,
            response_format={"type": "json_object"},
        )
        elapsed = time.perf_counter() - started
        logger.info("OpenAI reconciliation request finished model=%s elapsed_sec=%.2f", self.model, elapsed)
        return extract_openai_message_text(payload)

    def _call_anthropic(self, prompt: str) -> str:
        logger = logging.getLogger(__name__)
        logger.info("Anthropic reconciliation request started model=%s", self.model)
        started = time.perf_counter()
        payload = post_anthropic_message(
            api_key=self.api_key,
            model=self.model,
            system_prompt="Let's reconcile some transcriptions! Please return strict JSON only.",
            user_prompt=prompt,
            temperature=0,
            max_tokens=8192,
        )
        elapsed = time.perf_counter() - started
        logger.info("Anthropic reconciliation request finished model=%s elapsed_sec=%.2f", self.model, elapsed)
        return extract_anthropic_text(payload)

    def _repair_json(self, original_prompt: str, bad_output: str) -> dict:
        logger = logging.getLogger(__name__)
        logger.warning("Reconciler returned non-JSON output; running repair pass")
        repair_prompt = json.dumps(
            {
                "task": "Fix invalid JSON from transcript reconciler.",
                "original_prompt": original_prompt,
                "invalid_output": bad_output,
                "requirement": "Return valid JSON only.",
            }
        )
        repaired = self._call_model(repair_prompt)
        self._write_artifact("response_repair_raw.txt", repaired)
        try:
            return _extract_json_payload(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Failed to parse reconciler output as JSON after repair. "
                f"Raw repaired output (first 500 chars): {repaired[:500]!r}"
            ) from exc

    def _parse_or_repair_json(self, original_prompt: str, output_text: str) -> dict:
        try:
            return _extract_json_payload(output_text)
        except json.JSONDecodeError:
            return self._repair_json(original_prompt, output_text)

    def _write_artifact(self, filename: str, content: str) -> None:
        if not self.artifacts_dir:
            return
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / filename).write_text(content, encoding="utf-8")


def save_canonical_transcript(canonical: CanonicalTranscript, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonical.model_dump(), indent=2), encoding="utf-8")


def _extract_json_payload(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise json.JSONDecodeError("Empty response", raw, 0)

    # Common case: model wraps response in ```json ... ``` fences.
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        raise json.JSONDecodeError("Top-level JSON is not an object", raw, 0)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first balanced JSON object.
    start = raw.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", raw, 0)

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : idx + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                raise json.JSONDecodeError("Top-level JSON is not an object", candidate, 0)

    raise json.JSONDecodeError("Unbalanced JSON object", raw, start)
