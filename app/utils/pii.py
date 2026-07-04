"""
PII Detection Utilities — shared regex patterns for email, phone, and credit card detection.
Used by validator_node and guardrails_node.
"""

import re

# Email pattern
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Phone pattern (various formats)
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s\-().]{7,}\d)\b")

# Credit card pattern (4 groups of 4 digits with optional spaces/dashes)
CARD_RE = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")


def strip_pii(text: str) -> tuple[str, list[str]]:
    """
    Remove PII from text and return (cleaned_text, list_of_warnings).
    
    Args:
        text: Input text that may contain PII
        
    Returns:
        Tuple of (cleaned_text, warnings_list)
    """
    warnings: list[str] = []
    
    if EMAIL_RE.search(text):
        text = EMAIL_RE.sub("[EMAIL]", text)
        warnings.append("email address detected and removed")
    
    if PHONE_RE.search(text):
        text = PHONE_RE.sub("[PHONE]", text)
        warnings.append("phone number detected and removed")
    
    if CARD_RE.search(text):
        text = CARD_RE.sub("[CARD]", text)
        warnings.append("possible card number detected and removed")
    
    return text, warnings


def contains_pii(text: str) -> bool:
    """Check if text contains any PII patterns."""
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text) or CARD_RE.search(text))


def redact_pii(text: str) -> str:
    """Redact PII in text (for output sanitization). Returns redacted text."""
    text = EMAIL_RE.sub("[EMAIL REDACTED]", text)
    text = PHONE_RE.sub("[PHONE REDACTED]", text)
    text = CARD_RE.sub("[CARD REDACTED]", text)
    return text