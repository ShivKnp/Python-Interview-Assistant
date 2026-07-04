"""
Node ⑧: Hallucination Pre-Check
Before generating an answer, verifies the retrieved context actually contains
enough information to support a grounded response. If not, marks the state
so the generator uses a "limited context" fallback path.
"""

import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.pipeline.state import PipelineState
from app.observability.events import publish
from app.observability.metrics import metrics

NODE_LABEL = "Hallucination Check"
NODE_NAME = "hallucination"

GROUNDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a grounding verifier for a Python programming assistant.
Given the retrieved context and a user question, assess whether the context contains
sufficient information to answer the question factually.

Respond with ONLY one of:
- "yes" — context clearly supports answering the question
- "partial" — context is tangentially related but may not fully answer
- "no" — context is unrelated or insufficient

No explanation, just one word."""),
    ("human", """Question: {question}

Retrieved context (first 1500 chars):
{context_preview}

Can this context support a grounded answer? (yes/partial/no)"""),
])


async def hallucination_node(state: PipelineState, llm) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")

    intent = state.get("intent", "rag")
    if intent == "out_of_topic":
        await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                    "status": "success", "latency_ms": 0, "detail": "Skipped (out of topic)"})
        return {
            "can_answer": False,
            "pipeline_trace": state.get("pipeline_trace", []) + [{
                "node": NODE_NAME, "label": NODE_LABEL,
                "status": "success", "latency_ms": 0, "detail": "Skipped (out of topic)"
            }]
        }

    context = state.get("compressed_context", "")
    query = state.get("cleaned_query") or state.get("raw_query", "")
    has_user_docs = bool(state.get("user_doc_results"))

    if not context:
        # No context at all → definitely cannot ground answer
        latency = int((time.perf_counter() - t0) * 1000)
        metrics.record_hallucination_check_failed()
        await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                    "status": "warning", "latency_ms": latency, "detail": "No context — fallback mode"})
        return _result(state, False, latency, "No context found — using general knowledge")

    if has_user_docs:
        latency = int((time.perf_counter() - t0) * 1000)
        detail = "User documents retrieved — bypassing grounding check to allow document analysis"
        await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                    "status": "success", "latency_ms": latency, "detail": detail})
        return _result(state, True, latency, detail)

    try:
        import asyncio
        chain = GROUNDING_PROMPT | llm | StrOutputParser()
        verdict = await asyncio.wait_for(chain.ainvoke({
            "question": query,
            "context_preview": context[:1500],
        }), timeout=5.0)
        verdict = verdict.strip().lower().split()[0] if verdict.strip() else "no"
    except Exception:
        verdict = "partial"  # fail-open: attempt generation

    can_answer = verdict in ("yes", "partial")

    if not can_answer:
        metrics.record_hallucination_check_failed()

    latency = int((time.perf_counter() - t0) * 1000)
    status = "success" if can_answer else "warning"
    detail = {
        "yes": "Context fully supports answer",
        "partial": "Partial context — proceeding with caveat",
        "no": "Insufficient context — fallback mode",
    }.get(verdict, "Unknown")

    await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                "status": status, "latency_ms": latency, "detail": detail})
    return _result(state, can_answer, latency, detail)


def _result(state: PipelineState, can_answer: bool, latency: int, detail: str) -> dict:
    return {
        "can_answer": can_answer,
        "pipeline_trace": state.get("pipeline_trace", []) + [{
            "node": NODE_NAME, "label": NODE_LABEL,
            "status": "success" if can_answer else "warning",
            "latency_ms": latency, "detail": detail
        }]
    }
