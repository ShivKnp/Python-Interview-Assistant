"""
Node ⑪: Safety / Guardrails
Scans the generated answer for harmful content, off-topic responses,
and PII leakage. Replaces unsafe content with a safe refusal message.
"""

import re
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.pipeline.state import PipelineState
from app.observability.events import publish
from app.observability.metrics import metrics
from app.utils.pii import EMAIL_RE, PHONE_RE, redact_pii

NODE_LABEL = "Safety Guardrails"
NODE_NAME = "guardrails"

# Hard-coded harmful terms (supplement with LLM safety check)
_HARMFUL_PATTERNS = [
    r"\b(hack|exploit|malware|ransomware|rootkit|keylogger)\b.*\b(python|script|code)\b",
    r"\b(steal|exfiltrate|extract).*(password|credential|token|secret)\b",
    r"\bos\.system\s*\(\s*[\"']rm\s+-rf\b",
]
_HARMFUL_RE = [re.compile(p, re.IGNORECASE) for p in _HARMFUL_PATTERNS]

SAFETY_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a safety classifier for a Python programming assistant.
Review the response and classify it. Respond with ONLY one word:
- "safe" — response is appropriate for a programming assistant
- "unsafe" — response promotes harmful activities, leaks credentials, or is deeply inappropriate

Do NOT flag normal Python code, security best practices, or pentesting concepts for educational purposes."""),
    ("human", "Response to review:\n{answer}"),
])

_SAFE_REFUSAL = (
    "I'm sorry, I can't help with that request. "
    "Please ask me Python programming questions — I'm here to help you code!"
)


async def guardrails_node(state: PipelineState, llm) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")

    await publish(response_id, {"type": "node_start", "node": NODE_NAME, "label": NODE_LABEL})

    answer = state.get("generation", "")

    # ── Regex-based fast checks ────────────────────────────────────────────
    for pattern in _HARMFUL_RE:
        if pattern.search(answer):
            latency = int((time.perf_counter() - t0) * 1000)
            metrics.record_safety_block()
            await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                        "status": "warning", "latency_ms": latency, "detail": "Blocked: harmful pattern"})
            return _unsafe_result(state, latency)

    # ── PII leakage in output ──────────────────────────────────────────────
    answer = redact_pii(answer)

    # ── LLM-based safety check ─────────────────────────────────────────────
    try:
        chain = SAFETY_CHECK_PROMPT | llm | StrOutputParser()
        verdict = await chain.ainvoke({"answer": answer[:2000]})
        is_safe = "safe" in verdict.strip().lower()
    except Exception:
        is_safe = True  # fail-open

    latency = int((time.perf_counter() - t0) * 1000)

    if not is_safe:
        metrics.record_safety_block()
        await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                    "status": "warning", "latency_ms": latency, "detail": "Blocked: LLM safety check"})
        return _unsafe_result(state, latency)

    await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                "status": "success", "latency_ms": latency, "detail": "Content safe ✓"})
    return {
        "is_safe": True,
        "generation": answer,  # may have had PII stripped
        "pipeline_trace": state.get("pipeline_trace", []) + [{
            "node": NODE_NAME, "label": NODE_LABEL,
            "status": "success", "latency_ms": latency, "detail": "Content safe ✓"
        }]
    }


def _unsafe_result(state: PipelineState, latency: int) -> dict:
    return {
        "is_safe": False,
        "safety_message": _SAFE_REFUSAL,
        "generation": _SAFE_REFUSAL,
        "pipeline_trace": state.get("pipeline_trace", []) + [{
            "node": NODE_NAME, "label": NODE_LABEL,
            "status": "warning", "latency_ms": latency, "detail": "Blocked: safety violation"
        }]
    }
