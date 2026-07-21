"""LLM response validator tests."""

import logging
from types import SimpleNamespace

from app.validator import response_validator
from app.validator.response_validator import validate_llm_output

VALID_LLM_JSON = """
{
  "feedback_text": "Your loop skips the target at the right edge.",
  "hint_text": "Check how the upper bound is updated.",
  "error_category": "off_by_one",
  "reasoning_quality": "strong",
  "concept_gaps": ["binary search", "bounds"]
}
"""


def test_clean_json_parses() -> None:
    """Clean JSON validates into structured fields."""

    result = validate_llm_output(VALID_LLM_JSON, "sub_1", "def source(): pass")

    assert result is not None
    assert result.feedback_text.startswith("Your loop")
    assert result.error_category == "off_by_one"


def test_reasoning_scratchpad_is_stripped() -> None:
    """Internal scratchpad improves model output but never reaches validated data."""

    raw = """
    {
      "_reasoning_scratchpad": "Private chain of reasoning that is not returned.",
      "feedback_text": "Your bounds drop a valid candidate.",
      "hint_text": "Trace the interval after each comparison.",
      "error_category": "off_by_one",
      "reasoning_quality": "strong",
      "concept_gaps": ["bounds"]
    }
    """

    result = validate_llm_output(raw, "sub_1", "def source(): pass")

    assert result is not None
    assert "_reasoning_scratchpad" not in result.model_dump()


def test_missing_reasoning_scratchpad_still_parses() -> None:
    """Older model output without a scratchpad remains valid."""

    result = validate_llm_output(VALID_LLM_JSON, "sub_1", "def source(): pass")

    assert result is not None
    assert result.hint_text.startswith("Check")


def test_markdown_fenced_json_is_repaired() -> None:
    """Markdown fenced JSON is stripped and parsed."""

    result = validate_llm_output(f"```json\n{VALID_LLM_JSON}\n```", "sub_1", "def source(): pass")

    assert result is not None
    assert result.hint_text.startswith("Check")


def test_balanced_json_substring_is_extracted() -> None:
    """Surrounding prose is repaired with balanced object extraction."""

    raw = f"Here is the JSON:\n{VALID_LLM_JSON}\nThanks."

    result = validate_llm_output(raw, "sub_1", "def source(): pass")

    assert result is not None
    assert result.reasoning_quality == "strong"


def test_garbage_returns_none() -> None:
    """Unparseable output returns None."""

    assert validate_llm_output("not json at all", "sub_1", "def source(): pass") is None


def test_oversized_concept_gaps_are_truncated(monkeypatch) -> None:  
    """Overproduced concept gaps are capped by settings."""

    monkeypatch.setattr(
        response_validator,
        "get_settings",
        lambda: SimpleNamespace(max_concept_gaps=2),
    )
    raw = """
    {
      "feedback_text": "Good feedback",
      "hint_text": "Good hint",
      "error_category": "unknown",
      "reasoning_quality": "partial",
      "concept_gaps": ["a", "b", "c"]
    }
    """

    result = validate_llm_output(raw, "sub_1", "def source(): pass")

    assert result is not None
    assert result.concept_gaps == ["a", "b"]


def test_empty_feedback_is_invalid() -> None:
    """Empty feedback text is treated as invalid output."""

    raw = """
    {
      "feedback_text": "   ",
      "hint_text": "Good hint",
      "error_category": "unknown",
      "reasoning_quality": "partial",
      "concept_gaps": []
    }
    """

    assert validate_llm_output(raw, "sub_1", "def source(): pass") is None


def test_wrong_schema_version_is_invalid() -> None:
    """Unsupported schema versions are rejected when present."""

    raw = VALID_LLM_JSON.replace("{", '{"schema_version": "2.0",', 1)

    assert validate_llm_output(raw, "sub_1", "def source(): pass") is None


def test_banned_phrase_enforced_and_logs_warning(caplog) -> None:  
    """Banned phrase detection enforces rejection (result is None) AND logs a warning.

    Previously this was log-only. Now the response is rejected as a safety gate.
    """

    raw = VALID_LLM_JSON.replace(
        "Your loop skips the target at the right edge.",
        "Check your logic. Your loop skips the target."
    )

    with caplog.at_level(logging.WARNING):
        result = validate_llm_output(raw, "sub_1", "def source(): pass")

    assert result is None
    assert any("banned phrase used in llm output" in rec.message for rec in caplog.records)


def test_no_banned_phrase_logs_nothing(caplog) -> None:  
    """Valid feedback without banned phrases does not log a warning."""

    with caplog.at_level(logging.WARNING):
        result = validate_llm_output(VALID_LLM_JSON, "sub_1", "def source(): pass")

    assert result is not None
    assert not any("banned phrase used in llm output" in rec.message for rec in caplog.records)


def test_hallucinated_reference_enforced_and_logs_warning(caplog) -> None:  
    """Hallucinated reference detection enforces rejection AND logs a warning.

    Previously this was log-only. Now the response is rejected as a safety gate.
    """

    raw = VALID_LLM_JSON.replace(
        "Your loop skips the target at the right edge.",
        "Your loop skips the `missing_var` at the right edge."
    )

    with caplog.at_level(logging.WARNING):
        result = validate_llm_output(raw, "sub_1", "def source(): pass")

    assert result is None
    assert any("possible hallucinated reference" in rec.message for rec in caplog.records)


def test_valid_reference_does_not_log_warning(caplog) -> None:  
    """References that appear in source code do not trigger a warning."""

    raw = VALID_LLM_JSON.replace(
        "Your loop skips the target at the right edge.",
        "Your loop skips the `target` at the right edge."
    )

    with caplog.at_level(logging.WARNING):
        result = validate_llm_output(raw, "sub_1", "def source(target): pass")

    assert result is not None
    assert not any("possible hallucinated reference" in rec.message for rec in caplog.records)


def test_backticked_numeric_value_is_not_treated_as_identifier(caplog) -> None:
    """Concrete failed-case values may be quoted without becoming code references."""

    raw = VALID_LLM_JSON.replace(
        "Your loop skips the target at the right edge.",
        "The function returns the wrong pair when the target is `9`.",
    )

    with caplog.at_level(logging.WARNING):
        result = validate_llm_output(raw, "sub_1", "def source(nums, target): pass")

    assert result is not None
    assert not any("possible hallucinated reference" in rec.message for rec in caplog.records)


def test_hallucinated_reference_in_hint_text_enforced(caplog) -> None:  
    """Hallucinated identifier specifically in hint_text (feedback clean) is now caught.

    Before P2-2, only feedback_text was checked — this case would have passed silently.
    """
    raw = VALID_LLM_JSON.replace(
        "Check how the upper bound is updated.",
        "Try calling `missing_func` to fix the bound."
    )

    with caplog.at_level(logging.WARNING):
        result = validate_llm_output(raw, "sub_1", "def source(): pass")
    assert result is None
    assert any("possible hallucinated reference" in rec.message for rec in caplog.records)
