"""
Node ⑤: Hybrid Retrieval
Runs Vector search (ChromaDB MMR) + BM25 in parallel across all rewritten
query variants, merges results, and triggers Tavily web fallback when KB
confidence is below threshold. Also searches per-user document namespace.
"""

import time
from typing import Optional

from langchain_core.documents import Document

from app.config import get_settings
from app.pipeline.state import PipelineState
from app.observability.events import publish
from app.observability.metrics import metrics

NODE_LABEL = "Hybrid Retrieval"
NODE_NAME = "retriever"


def _dedup_docs(docs: list[Document]) -> list[Document]:
    """Remove duplicate documents by page_content hash."""
    seen: set[int] = set()
    out: list[Document] = []
    for doc in docs:
        h = hash(doc.page_content[:200])
        if h not in seen:
            seen.add(h)
            out.append(doc)
    return out


def _bm25_search(
    bm25_index, bm25_corpus: list[dict], query: str, top_k: int
) -> list[Document]:
    """Run BM25 retrieval on the pre-built index."""
    if bm25_index is None or not bm25_corpus:
        return []
    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    docs = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        entry = bm25_corpus[idx]
        docs.append(Document(
            page_content=entry["content"],
            metadata={**entry.get("metadata", {}), "bm25_score": float(scores[idx])},
        ))
    return docs


async def retriever_node(
    state: PipelineState,
    retriever,
    bm25_index,
    bm25_corpus: list[dict],
    web_search_tool,
    user_ns_manager,
) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")
    settings = get_settings()

    await publish(response_id, {"type": "node_start", "node": NODE_NAME, "label": NODE_LABEL})

    queries = state.get("rewritten_queries") or [state.get("cleaned_query") or state.get("raw_query", "")]
    primary_query = queries[0]

    intent = state.get("intent", "rag")
    if intent == "out_of_topic":
        await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                    "status": "success", "latency_ms": 0, "detail": "Skipped (out of topic)"})
        return {
            "vector_docs": [],
            "bm25_docs": [],
            "web_docs": [],
            "user_doc_results": [],
            "retrieval_score": 0.0,
            "pipeline_trace": state.get("pipeline_trace", []) + [{
                "node": NODE_NAME, "label": NODE_LABEL,
                "status": "success", "latency_ms": 0, "detail": "Skipped (out of topic)"
            }]
        }

    # ── Vector Search (all query variants) ───────────────────────────────
    vector_docs: list[Document] = []
    for q in queries:
        try:
            docs = retriever.invoke(q)
            vector_docs.extend(docs)
        except Exception:
            pass
    vector_docs = _dedup_docs(vector_docs)

    # Compute best vector relevance score (estimate from count)
    retrieval_score = min(1.0, len(vector_docs) / max(settings.TOP_K, 1))

    # ── BM25 Search ───────────────────────────────────────────────────────
    bm25_docs: list[Document] = []
    for q in queries:
        bm25_docs.extend(_bm25_search(bm25_index, bm25_corpus, q, settings.BM25_TOP_K))
    bm25_docs = _dedup_docs(bm25_docs)

    # ── User Document Search (Enterprise Feature) ──────────────────────────
    user_doc_results: list[Document] = []
    if state.get("include_user_docs") and user_ns_manager:
        user_id = state.get("user_id", "anonymous")
        try:
            user_doc_results = user_ns_manager.search(user_id, primary_query, k=4)
            if user_doc_results:
                metrics.record_user_doc_hit()
        except Exception:
            pass

    # ── Tavily Web Fallback ────────────────────────────────────────────────
    web_docs: list[Document] = []
    if retrieval_score < settings.RETRIEVAL_THRESHOLD and web_search_tool and web_search_tool.enabled:
        web_docs = web_search_tool.search(primary_query)
        if web_docs:
            metrics.record_tavily_fallback()

    latency = int((time.perf_counter() - t0) * 1000)
    total = len(vector_docs) + len(bm25_docs) + len(user_doc_results) + len(web_docs)
    detail = f"{len(vector_docs)} vector + {len(bm25_docs)} BM25"
    if user_doc_results:
        detail += f" + {len(user_doc_results)} user docs"
    if web_docs:
        detail += f" + {len(web_docs)} web (Tavily)"

    await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                "status": "success", "latency_ms": latency, "detail": detail})
    return {
        "vector_docs": vector_docs,
        "bm25_docs": bm25_docs,
        "web_docs": web_docs,
        "user_doc_results": user_doc_results,
        "retrieval_score": retrieval_score,
        "pipeline_trace": state.get("pipeline_trace", []) + [{
            "node": NODE_NAME, "label": NODE_LABEL,
            "status": "success", "latency_ms": latency, "detail": detail
        }]
    }
