"""Gemini 3.5 Transcribe adapter for OpenCut."""

import json
import logging
import mimetypes
import time

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

        if response.is_error:
            logger.error(
                "Gemini interactions error %s: %s",
                response.status_code,
                response.text,
            )

            if (
                response.status_code == 400
                and "Thinking is not enabled for this model" in response.text
            ):
                logger.warning(
                    "Gemini Interactions API regression detected; "
                    "retrying transcription via GenerateContent API."
                )

                generate_content_endpoint = (
                    f"{settings.GEMINI_API_BASE_URL}/models/"
                    f"{settings.GEMINI_TRANSCRIBE_MODEL}:generateContent"
                )

                language_map = {
                    "ml": "ml-IN",
                    "en": "en-IN",
                    "hi": "hi-IN",
                    "ta": "ta-IN",
                    "kn": "kn-IN",
                    "te": "te-IN",
                }

                audio_transcription_config = {
                    "wordTimestamp": True,
                    "mode": "VERBATIM",
                }

                if language and language.lower() not in {"auto", "detect"}:
                    normalized_language = language_map.get(
                        language.lower(),
                        language,
                    )
                    audio_transcription_config["languageCodes"] = [
                        normalized_language
                    ]

                generate_body = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "fileData": {
                                        "fileUri": file_uri,
                                        "mimeType": mime_type,
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "audioTranscriptionConfig": audio_transcription_config,
                    },
                }

                fallback_response = client.post(
                    generate_content_endpoint,
                    headers=api_headers,
                    json=generate_body,
                )

                if fallback_response.is_error:
                    logger.error(
                        "Gemini GenerateContent error %s: %s",
                        fallback_response.status_code,
                        fallback_response.text,
                    )

                    if (
                        fallback_response.status_code == 400
                        and "Thinking is not enabled for this model"
                        in fallback_response.text
                    ):
                        logger.warning(
                            "Gemini Transcribe model unavailable; "
                            "retrying audio transcription with %s.",
                            settings.GEMINI_EDIT_MODEL,
                        )

                        flash_endpoint = (
                            f"{settings.GEMINI_API_BASE_URL}/models/"
                            f"{settings.GEMINI_EDIT_MODEL}:generateContent"
                        )

                        requested_language = (
                            language
                            if language
                            and language.lower() not in {"auto", "detect"}
                            else "auto-detect"
                        )

                        flash_prompt = f"""
Transcribe this audio faithfully.

The spoken language is Malayalam or Malayalam mixed with English
technical words. Requested language: {requested_language}.

Rules:
- Preserve Malayalam speech in Malayalam script.
- Preserve naturally spoken English technical words in English.
- Do not translate.
- Do not summarize.
- Do not correct the meaning.
- Do not invent missing speech.
- Split the transcript into natural short segments.
- Give start and end timestamps as numeric seconds.
- Timestamps must be in chronological order.
- Return JSON only.

Return exactly this structure:
{{
  "language": "ml",
  "text": "complete transcript",
  "segments": [
    {{
      "text": "segment text",
      "start": 0.0,
      "end": 4.2
    }}
  ]
}}
""".strip()

                        flash_body = {
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [
                                        {
                                            "text": flash_prompt,
                                        },
                                        {
                                            "fileData": {
                                                "fileUri": file_uri,
                                                "mimeType": mime_type,
                                            }
                                        },
                                    ],
                                }
                            ],
                            "generationConfig": {
                                "responseMimeType": "application/json",
                                "temperature": 0.0,
                            },
                        }

                        flash_retryable_statuses = {
                            429,
                            500,
                            502,
                            503,
                            504,
                        }

                        flash_response = None

                        for flash_attempt in range(4):
                            flash_response = client.post(
                                flash_endpoint,
                                headers=api_headers,
                                json=flash_body,
                            )

                            if (
                                flash_response.status_code
                                not in flash_retryable_statuses
                            ):
                                break

                            logger.warning(
                                "Gemini Flash returned %s "
                                "on attempt %d/4.",
                                flash_response.status_code,
                                flash_attempt + 1,
                            )

                            if flash_attempt < 3:
                                wait_seconds = 2 ** (flash_attempt + 1)

                                logger.warning(
                                    "Retrying Gemini Flash in %ss.",
                                    wait_seconds,
                                )

                                time.sleep(wait_seconds)

                        if flash_response is None:
                            raise RuntimeError(
                                "Gemini Flash request did not run."
                            )

                        if flash_response.is_error:
                            logger.error(
                                "Gemini Flash audio fallback error %s: %s",
                                flash_response.status_code,
                                flash_response.text,
                            )

                        flash_response.raise_for_status()
                        flash_result = flash_response.json()

                        flash_parts = []

                        for candidate in flash_result.get("candidates", []):
                            content = candidate.get("content", {})

                            for part in content.get("parts", []):
                                part_text = part.get("text")
                                if part_text:
                                    flash_parts.append(part_text)

                        flash_json_text = "\n".join(flash_parts).strip()

                        if not flash_json_text:
                            raise RuntimeError(
                                "Gemini Flash returned empty transcription."
                            )

                        flash_payload = json.loads(flash_json_text)

                        raw_segments = flash_payload.get("segments", [])

                        if not raw_segments:
                            raise RuntimeError(
                                "Gemini Flash returned no transcript segments."
                            )

                        flash_segments = []
                        all_words = []

                        for index, raw_segment in enumerate(raw_segments):
                            segment_text = (
                                raw_segment.get("text") or ""
                            ).strip()

                            if not segment_text:
                                continue

                            try:
                                start = float(raw_segment.get("start", 0.0))
                                end = float(raw_segment.get("end", start))
                            except (TypeError, ValueError):
                                continue

                            if end <= start:
                                continue

                            tokens = segment_text.split()
                            words = []

                            if tokens:
                                segment_duration = end - start
                                word_duration = (
                                    segment_duration / len(tokens)
                                )

                                for word_index, token in enumerate(tokens):
                                    word_start = (
                                        start
                                        + word_duration * word_index
                                    )
                                    word_end = (
                                        start
                                        + word_duration
                                        * (word_index + 1)
                                    )

                                    word = {
                                        "word": token,
                                        "start": round(word_start, 3),
                                        "end": round(word_end, 3),
                                        "probability": 1.0,
                                    }

                                    words.append(word)
                                    all_words.append(word)

                            flash_segments.append(
                                {
                                    "id": len(flash_segments),
                                    "text": segment_text,
                                    "start": start,
                                    "end": end,
                                    "words": words,
                                    "avg_logprob": 0.0,
                                    "no_speech_prob": 0.0,
                                    "speaker": None,
                                }
                            )

                        if not flash_segments:
                            raise RuntimeError(
                                "Gemini Flash produced no usable segments."
                            )

                        flash_text = (
                            flash_payload.get("text") or ""
                        ).strip()

                        if not flash_text:
                            flash_text = " ".join(
                                segment["text"]
                                for segment in flash_segments
                            ).strip()

                        flash_duration = max(
                            (
                                segment["end"]
                                for segment in flash_segments
                            ),
                            default=0.0,
                        )

                        detected_language = (
                            flash_payload.get("language")
                            or language
                            or ""
                        )

                        logger.info(
                            "Gemini Flash audio transcription completed "
                            "with %d segments and %d words.",
                            len(flash_segments),
                            len(all_words),
                        )

                        return {
                            "text": flash_text,
                            "segments": flash_segments,
                            "language": detected_language,
                            "duration": flash_duration,
                        }

                fallback_response.raise_for_status()
                fallback_result = fallback_response.json()

                fallback_words = []
                fallback_text_parts = []

                for candidate in fallback_result.get("candidates", []):
                    content = candidate.get("content", {})

                    for part in content.get("parts", []):
                        text_part = (part.get("text") or "").strip()
                        if text_part:
                            fallback_text_parts.append(text_part)

                        transcription = part.get("audioTranscription")
                        if not transcription:
                            continue

                        for word_info in transcription.get("words", []):
                            word_text = (word_info.get("word") or "").strip()
                            if not word_text:
                                continue

                            fallback_words.append(
                                {
                                    "word": word_text,
                                    "start": _seconds(
                                        word_info.get("startOffset")
                                    ),
                                    "end": _seconds(
                                        word_info.get("endOffset")
                                    ),
                                    "probability": 1.0,
                                }
                            )

                if not fallback_words:
                    raise RuntimeError(
                        "Gemini GenerateContent returned no word timestamps."
                    )

                fallback_segments = _build_segments(fallback_words)

                fallback_duration = max(
                    (word["end"] for word in fallback_words),
                    default=0.0,
                )

                fallback_text = "\n".join(
                    fallback_text_parts
                ).strip()

                if not fallback_text:
                    fallback_text = " ".join(
                        word["word"] for word in fallback_words
                    ).strip()

                logger.info(
                    "Gemini GenerateContent transcription completed "
                    "with %d words.",
                    len(fallback_words),
                )

                return {
                    "text": fallback_text,
                    "segments": fallback_segments,
                    "language": language or "",
                    "duration": fallback_duration,
                }

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