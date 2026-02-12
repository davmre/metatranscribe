from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Segment(BaseModel):
    start_sec: float
    end_sec: float
    text: str
    confidence: float | None = None
    speaker: str | None = None

    @model_validator(mode="after")
    def _validate_times(self) -> "Segment":
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be >= start_sec")
        return self


class ProviderTranscript(BaseModel):
    provider_name: str
    audio_id: str
    language: str
    duration_sec: float
    segments: list[Segment] = Field(default_factory=list)
    raw_text: str
    confidence_summary: dict[str, float] | None = None
    raw_payload_path: str


class CanonicalSegment(BaseModel):
    start_sec: float
    end_sec: float
    text: str
    evidence_refs: list[str] = Field(default_factory=list)


class SilenceMarker(BaseModel):
    start_sec: float
    end_sec: float
    duration_sec: float
    label: str


class CanonicalTranscript(BaseModel):
    audio_id: str
    title: str
    language: str
    duration_sec: float
    segments: list[CanonicalSegment]
    silence_markers: list[SilenceMarker] = Field(default_factory=list)
    final_text: str
    provenance: dict[str, Any]


class AudioFileRecord(BaseModel):
    audio_id: str
    source_path: str
    source_hash: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ingested_at: str | None = None
    transcribed_at: str | None = None
    reconciled_at: str | None = None
    exported_at: str | None = None
    status: str = "ingested"
    retry_count: int = 0
    error: str | None = None


class PipelineSummary(BaseModel):
    run_id: str
    started_at: str
    completed_at: str
    processed: int
    succeeded: int
    failed: int
