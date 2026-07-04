"""
Node ④: Query Rewriting
Expands the user query into multiple variants for better retrieval recall.
Uses HyDE (Hypothetical Document Embedding) + sub-query decomposition.
"""

import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.pipeline.state import PipelineState
from app.observability.events import publish

NODE_LABEL = "Query Rewriting"
NODE_NAME = "rewriter"

REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query expansion specialist for a Python programming knowledge base.
Given a user query, generate 2 alternative search queries that would retrieve the same information
from different angles. Also generate a short hypothetical answer snippet (HyDE) that an ideal
document would contain.

Output format (exactly 3 lines, one item per line):
1. <alternative query 1>
2. <alternative query 2>
3. <one-sentence hypothetical answer>

Keep each line concise (under 100 chars). No numbering prefix, just the text."""),
    ("human", "Original query: {query}"),
])


async def rewriter_node(state: PipelineState, llm) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")
    query = state.get("cleaned_query") or state.get("raw_query", "")

    await publish(response_id, {"type": "node_start", "node": NODE_NAME, "label": NODE_LABEL})

    intent = state.get("intent", "rag")
    if intent == "out_of_topic":
        await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                    "status": "success", "latency_ms": 0, "detail": "Skipped (out of topic)"})
        return {
            "rewritten_queries": [query],
            "pipeline_trace": state.get("pipeline_trace", []) + [{
                "node": NODE_NAME, "label": NODE_LABEL,
                "status": "success", "latency_ms": 0, "detail": "Skipped (out of topic)"
            }]
        }
    rewritten = [query]  # always keep original

    try:
        import asyncio
        chain = REWRITER_PROMPT | llm | StrOutputParser()
        raw = await asyncio.wait_for(chain.ainvoke({"query": query}), timeout=5.0)
        lines = [ln.strip().lstrip("123. ").strip() for ln in raw.strip().splitlines() if ln.strip()]
        # Take first 3 non-empty lines
        extras = [ln for ln in lines if ln and ln != query][:3]
        rewritten.extend(extras)
    except Exception:
        pass  # fall back to original query only

    latency = int((time.perf_counter() - t0) * 1000)
    detail = f"{len(rewritten)} query variants"

    await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                "status": "success", "latency_ms": latency, "detail": detail})
    return {
        "rewritten_queries": rewritten,
        "pipeline_trace": state.get("pipeline_trace", []) + [{
            "node": NODE_NAME, "label": NODE_LABEL,
            "status": "success", "latency_ms": latency, "detail": detail
        }]
    }

