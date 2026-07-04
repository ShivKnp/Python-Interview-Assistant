"""
Pipeline Node Decorator — eliminates boilerplate across all 13 pipeline nodes.

Each node currently duplicates:
  - Timing (perf_counter)
  - Event publishing (node_start / node_done)
  - Trace appending
  - Latency calculation

This module provides a decorator that handles all of that automatically.
"""

import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from app.observability.events import publish
from app.pipeline.state import PipelineState

T = TypeVar("T", bound=Callable[..., Awaitable[dict]])


def pipeline_node(node_name: str, node_label: str) -> Callable[[T], T]:
    """
    Decorator for pipeline nodes that handles:
    - Timing (start/end)
    - SSE event publishing (node_start, node_done)
    - Pipeline trace appending
    - Latency calculation in milliseconds
    
    Usage:
        @pipeline_node("retriever", "Hybrid Retrieval")
        async def retriever_node(state: PipelineState, retriever, bm25_index, ...) -> dict:
            # Your node logic here
            return {"vector_docs": docs, "bm25_docs": bm25_docs, ...}
    
    The decorated function should return a dict with the state updates.
    The decorator will automatically add 'pipeline_trace' to the returned dict.
    """
    def decorator(func: T) -> T:
        @wraps(func)
        async def wrapper(state: PipelineState, *args: Any, **kwargs: Any) -> dict:
            response_id = state.get("response_id", "")
            t0 = time.perf_counter()
            
            # Publish node_start event
            await publish(response_id, {
                "type": "node_start",
                "node": node_name,
                "label": node_label
            })
            
            try:
                # Call the actual node function
                result = await func(state, *args, **kwargs)
                
                # Calculate latency
                latency_ms = int((time.perf_counter() - t0) * 1000)
                
                # Build trace entry
                trace_entry = {
                    "node": node_name,
                    "label": node_label,
                    "status": result.get("trace_status", "success"),
                    "latency_ms": latency_ms,
                    "detail": result.get("trace_detail", "")
                }
                
                # Publish node_done event
                await publish(response_id, {
                    "type": "node_done",
                    "node": node_name,
                    "label": node_label,
                    "status": trace_entry["status"],
                    "latency_ms": latency_ms,
                    "detail": trace_entry["detail"]
                })
                
                # Append to pipeline_trace
                existing_trace = state.get("pipeline_trace", [])
                result["pipeline_trace"] = existing_trace + [trace_entry]
                
                return result
                
            except Exception as exc:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                trace_entry = {
                    "node": node_name,
                    "label": node_label,
                    "status": "error",
                    "latency_ms": latency_ms,
                    "detail": f"Node error: {exc}"
                }
                
                await publish(response_id, {
                    "type": "node_done",
                    "node": node_name,
                    "label": node_label,
                    "status": "error",
                    "latency_ms": latency_ms,
                    "detail": trace_entry["detail"]
                })
                
                existing_trace = state.get("pipeline_trace", [])
                return {
                    **state,
                    "pipeline_trace": existing_trace + [trace_entry],
                    "error_type": "pipeline_error",
                    "error_message": f"Node {node_name} failed: {exc}"
                }
        
        return wrapper  # type: ignore
    return decorator


def trace_entry(
    status: str = "success",
    detail: str = "",
    trace_status: str | None = None,
    trace_detail: str | None = None
) -> dict:
    """
    Helper to create standardized trace entry dicts for node returns.
    
    Usage in node:
        return {
            "my_field": value,
            **trace_entry(status="success", detail="Processed 5 docs")
        }
    """
    return {
        "trace_status": trace_status or status,
        "trace_detail": trace_detail or detail
    }


def error_trace(detail: str) -> dict:
    """Helper for error trace entries."""
    return trace_entry(status="error", detail=detail)


def warning_trace(detail: str) -> dict:
    """Helper for warning trace entries."""
    return trace_entry(status="warning", detail=detail)