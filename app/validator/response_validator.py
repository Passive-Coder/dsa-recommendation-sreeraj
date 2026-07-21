"""Validate and repair raw LLM output."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from app.config.settings import get_settings
from app.logging.logger import get_logger
from app.models.response_schemas import ErrorCategory, ReasoningQuality

logger = get_logger(__name__)

SUPPORTED_SCHEMA_VERSION = "1.0"

BANNED_PHRASES = [
    "check your logic",
    "review your algorithm",
    "handle edge cases",
    "maintain state",
    "consider duplicates",
    "review your loop conditions",
    "your algorithm is incorrect",
]

BANNED_PHRASE_FALLBACK = (
    "Feedback could not be verified — please try again or review the judge output."
)
HALLUCINATION_FALLBACK = (
    "Feedback could not be verified against your submission — please try again."
)


class _LLMResponsePayload(BaseModel):
    """Subset of AnalyzeResponse expected directly from the LLM."""

    feedback_text: str
    hint_text: str
    error_category: ErrorCategory
    reasoning_quality: ReasoningQuality
    concept_gaps: list[str]


def _strip_code_fence(raw_text: str) -> str | None:
    """Return code-fence contents when the entire response is fenced."""

    stripped = raw_text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return None

    lines = stripped.splitlines()
    if len(lines) < 3:
        return None
    return "\n".join(lines[1:-1]).strip()


def _extract_first_json_object(raw_text: str) -> str | None:
    """Extract the first balanced JSON object substring."""

    start = raw_text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw_text)):
        char = raw_text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start : index + 1]

    return None


def _candidate_texts(raw_text: str) -> list[str]:
    """Return parse candidates in validation order without duplicates."""

    candidates = [raw_text]
    fenced = _strip_code_fence(raw_text)
    if fenced is not None:
        candidates.append(fenced)
    extracted = _extract_first_json_object(raw_text)
    if extracted is not None:
        candidates.append(extracted)

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _validate_candidate(candidate: str) -> _LLMResponsePayload | None:
    """Parse and validate one raw JSON candidate."""

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    schema_version = data.get("schema_version")
    if schema_version is not None and schema_version != SUPPORTED_SCHEMA_VERSION:
        return None

    data = _normalize_candidate_data(data)

    if data is None:
        return None

    try:
        payload = _LLMResponsePayload.model_validate(data)
    except ValidationError:
        return None

    if not payload.feedback_text.strip() or not payload.hint_text.strip():
        return None

    return payload.model_copy(
        update={
            "feedback_text": payload.feedback_text.strip(),
            "hint_text": payload.hint_text.strip(),
        },
    )


def _normalize_candidate_data(data: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize repairable LLM fields before Pydantic validation."""

    data = data.copy()
    # Internal-only field: it improves model reasoning, but may contain false starts or
    # unreviewed solution fragments. It must never reach the backend or the student.
    data.pop("_reasoning_scratchpad", None)

    concept_gaps = data.get("concept_gaps")
    if not isinstance(concept_gaps, list) or not all(
        isinstance(item, str) for item in concept_gaps
    ):
        return None

    settings = get_settings()
    if len(concept_gaps) > settings.max_concept_gaps:
        logger.warning(
            "llm output concept_gaps truncated",
            extra={
                "original_count": len(concept_gaps),
                "max_concept_gaps": settings.max_concept_gaps,
            },
        )
        data = data | {"concept_gaps": concept_gaps[: settings.max_concept_gaps]}

    return data


def _extract_identifiers(text: str) -> set[str]:
    """Extract code identifiers (backtick-wrapped or camelCase/snake_case) from text."""

    # Backticks are also commonly used for concrete values from failed cases
    # (for example, `9` or `[0, 1]`). Only treat a backticked token as a code
    # reference when it is actually a valid identifier.
    backtick_matches = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text)
    identifier_matches = re.findall(
        r"\b[a-z]+(?:_[a-z0-9]+)+|[a-z]+[A-Z][a-zA-Z0-9]*\b", text
    )
    return set(backtick_matches + identifier_matches)


def validate_llm_output(
    raw_text: str,
    submission_id: str,
    source_code: str,
) -> _LLMResponsePayload | None:
    """Validate raw LLM output into response fields, returning None on failure."""

    for candidate in _candidate_texts(raw_text):
        payload = _validate_candidate(candidate)
        if payload is not None:
            if _check_banned_phrases(payload, submission_id):
                return None
            if _check_hallucinated_references(payload, submission_id, source_code):
                return None
            return payload

    logger.warning(
        "llm output validation failed",
        extra={"submission_id": submission_id},
    )
    return None


def _check_banned_phrases(payload: _LLMResponsePayload, submission_id: str) -> bool:
    """Log a warning and return True if any banned phrase is detected.

    Returning True causes validate_llm_output to reject the response entirely,
    triggering the llm_output_invalid fallback path in the orchestrator.
    """

    combined_text = (payload.feedback_text + " " + payload.hint_text).lower()
    for phrase in BANNED_PHRASES:
        if phrase in combined_text:
            logger.warning(
                "banned phrase used in llm output",
                extra={
                    "submission_id": submission_id,
                    "matched_phrase": phrase,
                },
            )
            return True
    return False


def _check_hallucinated_references(
    payload: _LLMResponsePayload, submission_id: str, source_code: str
) -> bool:
    """Log a warning and return True if identifiers not in source code are detected.

    Checks both feedback_text and hint_text — either field referencing a nonexistent
    identifier triggers the same enforcement (return True → response rejected).
    """
    candidates = _extract_identifiers(payload.feedback_text) | _extract_identifiers(
        payload.hint_text
    )

    for candidate in candidates:
        if candidate not in source_code:
            logger.warning(
                "possible hallucinated reference",
                extra={
                    "submission_id": submission_id,
                    "mismatched_identifier": candidate,
                    "possible_hallucinated_reference": True,
                },
            )
            return True
    return False
