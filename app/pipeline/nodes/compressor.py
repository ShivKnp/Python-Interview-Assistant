"""
Node ⑦: Context Compression
Trims the combined context to fit within a token budget, preserving the most
relevant passages per document. Uses character-level approximation (1 token ≈ 4 chars).
"""

import time

from langchain_core.documents import Document

from app.config import get_settings
from app.pipeline.state import PipelineState
from app.observability.events import publish

NODE_LABEL = "Context Compression"
NODE_NAME = "compressor"

CHARS_PER_TOKEN = 4  # rough approximation


def _build_context(docs: list[Document], char_budget: int) -> str:
    """
    Build a context string from docs, truncating individual docs if needed
    to stay within the overall character budget.
    """
    context_parts: list[str] = []
    used_chars = 0

    for doc in docs:
        source = doc.metadata.get("source", "knowledge_base")
        qid = doc.metadata.get("question_id", "?")
        score = doc.metadata.get("q_score", "?")
        title = doc.metadata.get("title", "")

        if source == "web":
            header = f"[Web Source: {doc.metadata.get('url', '')}]"
        elif source == "user_doc":
            header = f"[User Document: {doc.metadata.get('filename', title)}]"
        else:
            header = f"[Stack Overflow Q#{qid} | Score: {score} | {title}]"

        remaining = char_budget - used_chars - len(header) - 10
        if remaining <= 0:
            break

        content = doc.page_content[:remaining]
        part = f"{header}\n{content}"
        context_parts.append(part)
        used_chars += len(part)

    return "\n\n---\n\n".join(context_parts)


async def compressor_node(state: PipelineState) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")
    settings = get_settings()

    intent = state.get("intent", "rag")
    if intent == "out_of_topic":
        await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                    "status": "success", "latency_ms": 0, "detail": "Skipped (out of topic)"})
        return {
            "compressed_context": "",
            "pipeline_trace": state.get("pipeline_trace", []) + [{
                "node": NODE_NAME, "label": NODE_LABEL,
                "status": "success", "latency_ms": 0, "detail": "Skipped (out of topic)"
            }]
        }

    docs = state.get("reranked_docs", [])
    token_budget = max(600, settings.GENERATION_CONTEXT_TOKEN_BUDGET)
    char_budget = token_budget * CHARS_PER_TOKEN
    compressed = _build_context(docs, char_budget)

    latency = int((time.perf_counter() - t0) * 1000)
    token_est = len(compressed) // CHARS_PER_TOKEN
    detail = f"~{token_est} tokens across {len(docs)} docs"

    await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                "status": "success", "latency_ms": latency, "detail": detail})
    return {
        "compressed_context": compressed,
        "pipeline_trace": state.get("pipeline_trace", []) + [{
            "node": NODE_NAME, "label": NODE_LABEL,
            "status": "success", "latency_ms": latency, "detail": detail
        }]
    }
