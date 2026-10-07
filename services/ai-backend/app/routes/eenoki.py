"""EENOKI semantic auto-edit routes."""

import asyncio
import json
import logging
import time
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eenoki", tags=["eenoki"])


class AutoCutSegment(BaseModel):
    id: str | int
    text: str
    start: float
    end: float


class AutoCutRequest(BaseModel):
    segments: list[AutoCutSegment]
    objective: str = (
        "Create a concise, natural social-media video while preserving "
        "the speaker's intended meaning, technical accuracy, authority, and CTA."
    )


class AutoCutDecision(BaseModel):
    segment_id: str
    action: Literal["KEEP", "CUT", "REVIEW"]
    reason: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AutoCutResponse(BaseModel):
    decisions: list[AutoCutDecision]
    cut_ranges: list[dict]
    summary: str


SYSTEM_PROMPT = """
You are the semantic video editor for EENOKI INFRA CONSULTANCY LLP.

The speaker commonly uses natural Malayalam mixed with English technical terms.
Your job is to tighten spoken videos WITHOUT changing the intended meaning.

For EVERY transcript segment, return exactly one decision.

KEEP:
- technical explanations
- useful examples
- hook/problem statement
- important context
- conclusions
- EENOKI service positioning
- useful CTA
- sentences needed for continuity

CUT only when clearly unnecessary:
- obvious false starts immediately restarted
- exact or near-exact repeated ideas
- abandoned/incomplete speech with no useful meaning
- pure verbal filler with no content
- accidental unrelated speech

REVIEW:
- anything ambiguous
- anything that may affect meaning if removed
- technical content you are not confident about

Rules:
1. Never invent facts.
2. Never rewrite technical claims.
3. Never cut merely because grammar is informal.
4. Malayalam-English code switching is normal.
5. When uncertain, KEEP or REVIEW — never CUT.
6. Do not remove the only CTA or company/service identification.
7. Judge every segment using the full transcript context.

Return JSON only:
{
  "decisions": [
    {
      "segment_id": "0",
      "action": "KEEP",
      "reason": "short reason",
      "confidence": 0.95
    }
  ],
  "summary": "short overall editing assessment"
}
"""


def _gemini_generate_json(prompt: str) -> dict:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("Gemini API key is not configured")

    url = (
        f"{settings.GEMINI_API_BASE_URL}/models/"
        f"{settings.GEMINI_EDIT_MODEL}:generateContent"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"{SYSTEM_PROMPT}\n\n{prompt}",
                    }
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }

    timeout = httpx.Timeout(120.0, connect=30.0)

    retryable_statuses = {429, 500, 502, 503, 504}
    last_error = None

    with httpx.Client(timeout=timeout) as client:
        for attempt in range(4):
            try:
                response = client.post(
                    url,
                    headers={
                        "x-goog-api-key": settings.GEMINI_API_KEY,
                        "Content-Type": "application/json; charset=utf-8",
                    },
                    json=payload,
                )

                if response.status_code in retryable_statuses:
                    last_error = httpx.HTTPStatusError(
                        f"Retryable Gemini error: {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                    if attempt < 3:
                        wait_seconds = 2 ** (attempt + 1)
                        logger.warning(
                            "Gemini returned %s; retrying in %ss",
                            response.status_code,
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                        continue

                response.raise_for_status()
                data = response.json()
                break

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc

                if attempt < 3:
                    wait_seconds = 2 ** (attempt + 1)
                    logger.warning(
                        "Gemini connection error; retrying in %ss",
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                raise
        else:
            if last_error:
                raise last_error
            raise RuntimeError("Gemini request failed after retries")

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError("Gemini returned no content")

    text = parts[0].get("text", "")
    if not text:
        raise RuntimeError("Gemini returned empty content")

    return json.loads(text)


@router.post("/auto-cut/plan", response_model=AutoCutResponse)
async def auto_cut_plan(request: AutoCutRequest) -> AutoCutResponse:
    if not request.segments:
        raise HTTPException(
            status_code=400,
            detail="No transcript segments provided.",
        )

    transcript_payload = [
        {
            "id": str(seg.id),
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        }
        for seg in request.segments
    ]

    prompt = (
        f"Editing objective:\n{request.objective}\n\n"
        "Transcript segments:\n"
        f"{json.dumps(transcript_payload, ensure_ascii=False, indent=2)}"
    )

    try:
        result = await asyncio.to_thread(
            _gemini_generate_json,
            prompt,
        )
        logger.info(
            "EENOKI Gemini auto-cut plan completed for %d segments",
            len(request.segments),
        )
    except Exception as exc:
        logger.exception("EENOKI Gemini auto-cut analysis failed")

        safe_decisions = [
            AutoCutDecision(
                segment_id=str(seg.id),
                action="KEEP",
                reason="Gemini analysis unavailable; preserved for safety.",
                confidence=0.0,
            )
            for seg in request.segments
        ]

        return AutoCutResponse(
            decisions=safe_decisions,
            cut_ranges=[],
            summary=f"Gemini unavailable. No automatic cuts applied. {exc}",
        )

    raw_decisions = result.get("decisions", [])
    decisions: list[AutoCutDecision] = []

    valid_ids = {str(seg.id) for seg in request.segments}

    for item in raw_decisions:
        try:
            segment_id = str(item.get("segment_id", ""))
            action = str(item.get("action", "REVIEW")).upper()

            if segment_id not in valid_ids:
                continue

            if action not in {"KEEP", "CUT", "REVIEW"}:
                action = "REVIEW"

            decisions.append(
                AutoCutDecision(
                    segment_id=segment_id,
                    action=action,
                    reason=str(item.get("reason", "No reason provided")),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        except Exception:
            continue

    decided_ids = {d.segment_id for d in decisions}

    for seg in request.segments:
        sid = str(seg.id)
        if sid not in decided_ids:
            decisions.append(
                AutoCutDecision(
                    segment_id=sid,
                    action="KEEP",
                    reason="No reliable Gemini decision returned; preserved for safety.",
                    confidence=0.0,
                )
            )

    segment_map = {
        str(seg.id): seg
        for seg in request.segments
    }

    cut_ranges = [
        {
            "segment_id": d.segment_id,
            "start": segment_map[d.segment_id].start,
            "end": segment_map[d.segment_id].end,
            "reason": d.reason,
            "confidence": d.confidence,
        }
        for d in decisions
        if d.action == "CUT"
    ]

    return AutoCutResponse(
        decisions=decisions,
        cut_ranges=cut_ranges,
        summary=str(
            result.get(
                "summary",
                "Gemini semantic edit plan generated.",
            )
        ),
    )
