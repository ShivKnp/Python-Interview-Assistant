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
    ("system", """You are a query expansion and context-aware search specialist for a Python programming knowledge base.
Your job is to generate search queries and a hypothetical answer snippet (HyDE) based on the user's original query.

If the user's query is a short follow-up or contains ambiguous/context-dependent words (like "Give some examples", "why?", "how?", "more details", "what about X?"), use the provided Conversation History to resolve the ambiguity and make the search queries specific to the subject under discussion.

Generate 2 alternative search queries that would retrieve the relevant information from different angles. Also generate a short hypothetical answer snippet (HyDE) that an ideal document would contain.

Output format (exactly 3 lines, one item per line):
1. <alternative query 1>
2. <alternative query 2>
3. <one-sentence hypothetical answer>

Keep each line concise (under 100 chars). No numbering prefix, just the text."""),
    ("human", """Conversation History:
{history}

Original query: {query}"""),
])


def _format_history(history: list[dict], max_turns: int) -> str:
    recent = history[-max_turns:] if max_turns > 0 else []
    return "\n".join(
        f"{h['role'].capitalize()}: {h['content']}" for h in recent
    ) or "None"


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
        from app.config import get_settings
        settings = get_settings()
        history = state.get("conversation_history", [])
        history_turns = max(1, settings.GENERATION_HISTORY_TURNS)
        history_str = _format_history(history, history_turns)

        chain = REWRITER_PROMPT | llm | StrOutputParser()
        raw = await asyncio.wait_for(
            chain.ainvoke({"query": query, "history": history_str}),
            timeout=5.0
        )
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

