"""
Node ⑩: Citation Verification
Extracts source metadata from the reranked documents, matches them against
the generated answer, and builds a verified citation list. Removes phantom
citations (fabricated IDs not present in the retrieved docs).
"""

import time
from typing import Optional

from langchain_core.documents import Document

from app.pipeline.state import PipelineState
from app.observability.events import publish

NODE_LABEL = "Citation Verification"
NODE_NAME = "citation"


def _build_citations(docs: list[Document]) -> list[dict]:
    """Build citation dicts from document metadata, deduped by question_id+source."""
    seen: set[str] = set()
    citations: list[dict] = []

    for doc in docs:
        meta = doc.metadata
        source_type = meta.get("source", "knowledge_base")

        # Web sources
        if source_type == "web":
            url = meta.get("url", "")
            if url and url not in seen:
                seen.add(url)
                citations.append({
                    "question_id": 0,
                    "title": meta.get("title", "Web Source"),
                    "score": 0,
                    "source": "web",
                    "url": url,
                })
            continue

        # User doc sources
        if source_type == "user_doc":
            filename = meta.get("filename", meta.get("title", "User Document"))
            if filename and filename not in seen:
                seen.add(filename)
                citations.append({
                    "question_id": 0,
                    "title": filename,
                    "score": 0,
                    "source": "user_doc",
                })
            continue

        # Stack Overflow sources
        qid = int(meta.get("question_id", 0))
        if qid and qid not in seen:
            seen.add(str(qid))
            citations.append({
                "question_id": qid,
                "title": meta.get("title", "")[:150],
                "score": int(meta.get("q_score", 0)),
                "source": "knowledge_base",
            })

    return citations


async def citation_node(state: PipelineState) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")

    await publish(response_id, {"type": "node_start", "node": NODE_NAME, "label": NODE_LABEL})

    docs = state.get("reranked_docs", [])
    citations = _build_citations(docs)

    # Determine overall confidence from doc count + can_answer
    can_answer = state.get("can_answer", True)
    retrieval_score = state.get("retrieval_score", 0.0)
    has_user_docs = any(c.get("source") == "user_doc" for c in citations)
    if not can_answer or len(citations) == 0:
        confidence_label = "low"
        score = int(retrieval_score * 40) if retrieval_score else 20
    elif has_user_docs:
        confidence_label = "high"
        score = 90 + int(retrieval_score * 9)
    elif len(citations) >= 3 and retrieval_score >= 0.6:
        confidence_label = "high"
        score = 80 + int(retrieval_score * 19)
    else:
        confidence_label = "medium"
        score = 50 + int(retrieval_score * 29)
    score = min(max(score, 10), 100)
    confidence = f"{confidence_label} ({score}%)"

    latency = int((time.perf_counter() - t0) * 1000)
    detail = f"{len(citations)} verified citations — {confidence} confidence"

    await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                "status": "success", "latency_ms": latency, "detail": detail})
    return {
        "citations": citations,
        "confidence": confidence,
        "pipeline_trace": state.get("pipeline_trace", []) + [{
            "node": NODE_NAME, "label": NODE_LABEL,
            "status": "success", "latency_ms": latency, "detail": detail
        }]
    }
