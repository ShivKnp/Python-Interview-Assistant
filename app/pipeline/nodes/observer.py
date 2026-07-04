"""
Node ⑫: Logging & Observability
Final pipeline node. Writes structured log entry, updates metrics counters,
persists the message pair to SQLite, and signals SSE stream completion.
"""

import time

from app.pipeline.state import PipelineState
from app.observability.events import publish, close_bus
from app.observability.logger import logger
from app.observability.metrics import metrics
from app.memory import session_store

NODE_LABEL = "Logging & Observability"
NODE_NAME = "observer"


async def observer_node(state: PipelineState) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")
    session_id = state.get("session_id", "")

    await publish(response_id, {"type": "node_start", "node": NODE_NAME, "label": NODE_LABEL})

    # ── Compute total pipeline latency ────────────────────────────────────
    pipeline_start = state.get("pipeline_start_time", t0)
    total_latency_ms = int((t0 - pipeline_start) * 1000)

    # ── Determine final answer and error state ────────────────────────────
    error_type = state.get("error_type")
    generation = state.get("generation", "")
    intent = state.get("intent", "unknown")
    confidence = state.get("confidence", "low")
    is_safe = state.get("is_safe", True)
    query = state.get("cleaned_query") or state.get("raw_query", "")

    success = not bool(error_type)

    # ── Update global metrics ─────────────────────────────────────────────
    metrics.record_query(intent=intent, success=success, latency_ms=total_latency_ms)
    for trace_entry in state.get("pipeline_trace", []):
        metrics.record_node_latency(trace_entry["node"], trace_entry.get("latency_ms", 0))

    # ── Structured log entry ──────────────────────────────────────────────
    logger.info_data(
        "Pipeline completed",
        response_id=response_id,
        session_id=session_id,
        intent=intent,
        confidence=confidence,
        is_safe=is_safe,
        success=success,
        error_type=error_type,
        total_latency_ms=total_latency_ms,
        query_len=len(query),
        answer_len=len(generation),
        citation_count=len(state.get("citations", [])),
        tavily_used=bool(state.get("web_docs")),
        user_docs_used=bool(state.get("user_doc_results")),
    )

    # ── Persist messages to SQLite ────────────────────────────────────────
    if session_id and query and generation and not error_type:
        try:
            await session_store.save_message(session_id, "user", query)
            await session_store.save_message(
                session_id, "assistant", generation,
                intent=intent, confidence=confidence, response_id=response_id,
                citations=state.get("citations", []),
                pipeline_trace=state.get("pipeline_trace", []) + [trace_entry]
            )
        except Exception as exc:
            logger.warning(f"Failed to persist messages to SQLite: {exc}")

    latency = int((time.perf_counter() - t0) * 1000)
    detail = f"Total pipeline: {total_latency_ms}ms"

    trace_entry = {
        "node": NODE_NAME, "label": NODE_LABEL,
        "status": "success", "latency_ms": latency, "detail": detail
    }

    await publish(response_id, {
        "type": "node_done",
        "node": NODE_NAME,
        "label": NODE_LABEL,
        "status": "success",
        "latency_ms": latency,
        "detail": detail
    })

    # ── Signal SSE completion ─────────────────────────────────────────────
    final_event = {
        "type": "complete",
        "answer": state.get("generation", ""),
        "intent": intent,
        "confidence": confidence,
        "response_id": response_id,
        "citations": state.get("citations", []),
        "pipeline_trace": state.get("pipeline_trace", []) + [trace_entry],
        "total_latency_ms": total_latency_ms,
        "error_type": error_type,
        "error_message": state.get("error_message"),
        "web_fallback_used": bool(state.get("web_docs")),
    }
    await publish(response_id, final_event)
    await close_bus(response_id)

    return {
        "pipeline_trace": state.get("pipeline_trace", []) + [trace_entry]
    }
