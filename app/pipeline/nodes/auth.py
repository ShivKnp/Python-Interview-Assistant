"""
Node ①: Authentication
Validates the X-API-Key header. If API_KEY setting is empty, auth is disabled
(development mode) and all requests pass through.
"""

from app.config import get_settings
from app.pipeline.state import PipelineState
from app.observability.events import publish
from app.observability.metrics import metrics
from app.pipeline.decorators import pipeline_node, trace_entry, error_trace


NODE_LABEL = "Authentication"
NODE_NAME = "auth"


@pipeline_node(NODE_NAME, NODE_LABEL)
async def auth_node(state: PipelineState) -> dict:
    settings = get_settings()

    # Auth disabled in dev mode
    if not settings.API_KEY:
        return {
            "is_authenticated": True,
            **trace_entry(detail="Auth disabled (dev mode)")
        }

    provided_key = state.get("api_key", "").strip()
    is_valid = provided_key == settings.API_KEY

    if not is_valid:
        metrics.record_auth_failure()
        return {
            "is_authenticated": False,
            "error_type": "auth_error",
            "error_message": "Invalid or missing API key. Include X-API-Key header.",
            **error_trace("Invalid API key")
        }

    return {
        "is_authenticated": True,
        **trace_entry(detail="Authenticated")
    }