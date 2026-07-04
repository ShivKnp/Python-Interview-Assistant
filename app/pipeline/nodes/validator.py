"""
Node ②: Query Validation
Checks length, detects/strips PII, basic abuse filtering, and language hints.
"""

from typing import Optional

from app.pipeline.state import PipelineState
from app.observability.events import publish
from app.observability.metrics import metrics
from app.utils.pii import strip_pii
from app.pipeline.decorators import pipeline_node, trace_entry, error_trace


NODE_LABEL = "Query Validation"
NODE_NAME = "validator"

# Simple abuse word list (extend as needed)
_ABUSE_WORDS = {"ignore all", "ignore previous", "disregard your instructions",
                "jailbreak", "dan mode", "developer mode override"}

MIN_LENGTH = 3
MAX_LENGTH = 2000


def _check_injection(text: str) -> Optional[str]:
    """Detect prompt injection attempts."""
    lower = text.lower()
    for pattern in _ABUSE_WORDS:
        if pattern in lower:
            return f"Possible prompt injection detected: '{pattern}'"
    return None


@pipeline_node(NODE_NAME, NODE_LABEL)
async def validator_node(state: PipelineState) -> dict:
    query = state.get("raw_query", "").strip()

    # Length checks
    if len(query) < MIN_LENGTH:
        err = f"Query too short (min {MIN_LENGTH} characters)"
        metrics.record_validation_failure()
        return {
            "is_valid_query": False,
            "validation_error": err,
            "error_type": "validation_error",
            "error_message": err,
            **error_trace(err)
        }

    if len(query) > MAX_LENGTH:
        query = query[:MAX_LENGTH]  # truncate silently rather than reject

    # Injection check
    injection_msg = _check_injection(query)
    if injection_msg:
        metrics.record_validation_failure()
        return {
            "is_valid_query": False,
            "validation_error": injection_msg,
            "error_type": "validation_error",
            "error_message": injection_msg,
            **error_trace(injection_msg)
        }

    # PII strip
    cleaned, pii_warnings = strip_pii(query)
    detail = "Valid"
    if pii_warnings:
        detail = f"Valid (PII stripped: {', '.join(pii_warnings)})"

    return {
        "is_valid_query": True,
        "cleaned_query": cleaned,
        **trace_entry(detail=detail)
    }