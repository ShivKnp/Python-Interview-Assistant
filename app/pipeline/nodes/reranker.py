"""
Node ⑥: Re-ranking
Applies Reciprocal Rank Fusion (RRF) to merge vector + BM25 + web ranked
lists into a single diversity-filtered top-K list.
"""

import time
from collections import defaultdict

from langchain_core.documents import Document

from app.config import get_settings
from app.pipeline.state import PipelineState
from app.observability.events import publish

NODE_LABEL = "Re-ranking"
NODE_NAME = "reranker"

# RRF constant (controls the impact of rank position, typically 60)
RRF_K = 60
# Final top-K after re-ranking
FINAL_TOP_K = 5


def _doc_key(doc: Document) -> str:
    """Stable key for deduplication: first 150 chars of content."""
    return doc.page_content[:150]


def _reciprocal_rank_fusion(ranked_lists: list[list[Document]]) -> list[Document]:
    """
    Merge N ranked lists using RRF.
    Score formula: sum(1 / (k + rank)) for each list the doc appears in.
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = _doc_key(doc)
            scores[key] += 1.0 / (RRF_K + rank)
            if key not in doc_map:
                doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys]


async def reranker_node(state: PipelineState) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")

    intent = state.get("intent", "rag")
    if intent == "out_of_topic":
        await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                    "status": "success", "latency_ms": 0, "detail": "Skipped (out of topic)"})
        return {
            "reranked_docs": [],
            "pipeline_trace": state.get("pipeline_trace", []) + [{
                "node": NODE_NAME, "label": NODE_LABEL,
                "status": "success", "latency_ms": 0, "detail": "Skipped (out of topic)"
            }]
        }

    ranked_lists = []
    if state.get("vector_docs"):
        ranked_lists.append(state["vector_docs"])
    if state.get("bm25_docs"):
        ranked_lists.append(state["bm25_docs"])
    if state.get("user_doc_results"):
        ranked_lists.append(state["user_doc_results"])
    if state.get("web_docs"):
        ranked_lists.append(state["web_docs"])

    if not ranked_lists:
        reranked: list[Document] = []
    elif len(ranked_lists) == 1:
        reranked = ranked_lists[0][:FINAL_TOP_K]
    else:
        merged = _reciprocal_rank_fusion(ranked_lists)
        reranked = merged[:FINAL_TOP_K]

    latency = int((time.perf_counter() - t0) * 1000)
    detail = f"Top {len(reranked)} docs after RRF"

    await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                "status": "success", "latency_ms": latency, "detail": detail})
    return {
        "reranked_docs": reranked,
        "pipeline_trace": state.get("pipeline_trace", []) + [{
            "node": NODE_NAME, "label": NODE_LABEL,
            "status": "success", "latency_ms": latency, "detail": detail
        }]
    }
