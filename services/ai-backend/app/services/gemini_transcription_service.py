"""Gemini 3.5 Transcribe adapter for OpenCut."""

import logging
import mimetypes

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def _seconds(value: str | None) -> float:
    if not value:
        return 0.0
    value = str(value).strip()
    if value.endswith("s"):
        value = value[:-1]
    try:
        return float(value)
    except ValueError:
        return 0.0


def _build_segments(words: list[dict]) -> list[dict]:
    """Group Gemini word timestamps into OpenCut-friendly segments."""
    if not words:
        return []

    groups = []
    current = []

    for word in words:
        if current:
            gap = word["start"] - current[-1]["end"]
            segment_duration = word["end"] - current[0]["start"]

            if gap > 0.8 or segment_duration > 7.5 or len(current) >= 14:
                groups.append(current)
                current = []

        current.append(word)

    if current:
        groups.append(current)

    segments = []

    for index, group in enumerate(groups):
        segments.append(
            {
                "id": index,
                "text": " ".join(w["word"] for w in group).strip(),
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "words": group,
                "avg_logprob": 0.0,
                "no_speech_prob": 0.0,
                "speaker": None,
            }
        )

    return segments


def transcribe_with_gemini(
    contents: bytes,
    filename: str,
    content_type: str | None = None,
    language: str | None = None,
) -> dict:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("Gemini API key is not configured.")

    mime_type = content_type or mimetypes.guess_type(filename)[0] or "audio/mp4"

    # Windows/browser uploads sometimes report generic MIME types.
    if filename.lower().endswith(".m4a"):
        mime_type = "audio/mp4"

    api_root = settings.GEMINI_API_BASE_URL.removesuffix("/v1beta")
    upload_endpoint = f"{api_root}/upload/v1beta/files"
    interaction_endpoint = f"{settings.GEMINI_API_BASE_URL}/interactions"

    api_headers = {
        "x-goog-api-key": settings.GEMINI_API_KEY,
    }

    timeout = httpx.Timeout(300.0, connect=30.0)

    with httpx.Client(timeout=timeout) as client:
        # 1. Start resumable Gemini Files upload.
        start_headers = {
            **api_headers,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(contents)),
            "X-Goog-Upload-Header-Content-Type": mime_type,
        }

        start_resp = client.post(
            upload_endpoint,
            headers=start_headers,
            json={"file": {"display_name": filename}},
        )
        start_resp.raise_for_status()

        upload_url = start_resp.headers.get("x-goog-upload-url")
        if not upload_url:
            raise RuntimeError("Gemini did not return an upload URL.")

        # 2. Upload and finalize audio.
        upload_resp = client.post(
            upload_url,
            headers={
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
                "Content-Type": mime_type,
            },
            content=contents,
        )
        upload_resp.raise_for_status()

        file_data = upload_resp.json().get("file", {})
        file_uri = file_data.get("uri")

        if not file_uri:
            raise RuntimeError("Gemini Files API did not return a file URI.")

        transcription_config = {
            "language_codes": (
                [language]
                if language and language.lower() not in {"auto", "detect"}
                else []
            ),
            "mode": {
                "type": "verbatim",
                "timestamp_granularities": ["word"],
            },
        }

        # 3. Gemini 3.5 Transcribe.
        body = {
            "model": settings.GEMINI_TRANSCRIBE_MODEL,
            "input": [
                {
                    "type": "audio",
                    "uri": file_uri,
                    "mime_type": mime_type,
                }
            ],
            "generation_config": {
                "transcription_config": transcription_config,
            },
        }

        response = client.post(
            interaction_endpoint,
            headers=api_headers,
            json=body,
        )
        response.raise_for_status()

        result = response.json()

    full_text = ""
    annotations = []

    for step in result.get("steps", []):
        if step.get("type") != "model_output":
            continue

        for item in step.get("content", []):
            if item.get("type") == "text":
                full_text = item.get("text", "") or ""
                annotations.extend(item.get("annotations", []) or [])

    words = []

    for annotation in annotations:
        if annotation.get("type") != "word_info":
            continue

        word_text = (annotation.get("text") or "").strip()
        if not word_text:
            continue

        words.append(
            {
                "word": word_text,
                "start": _seconds(annotation.get("start_offset")),
                "end": _seconds(annotation.get("end_offset")),
                "probability": 1.0,
            }
        )

    if not words:
        raise RuntimeError("Gemini returned no word timestamps.")

    segments = _build_segments(words)
    duration = max((word["end"] for word in words), default=0.0)

    return {
        "text": full_text.strip(),
        "segments": segments,
        "language": language or "",
        "duration": duration,
    }