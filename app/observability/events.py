"""
SSE Event Bus — per-request asyncio.Queue for real-time pipeline streaming.

Each request creates a queue keyed by response_id. Pipeline nodes push events
to the queue; the SSE endpoint reads and forwards them to the browser.
"""

import asyncio
from typing import Any

# Global event bus: response_id → asyncio.Queue
_buses: dict[str, asyncio.Queue] = {}

_SENTINEL = None  # marks stream completion


def create_bus(response_id: str) -> asyncio.Queue:
    """Create and register an event queue for a new request."""
    q: asyncio.Queue = asyncio.Queue()
    _buses[response_id] = q
    return q


async def publish(response_id: str, event: dict[str, Any]) -> None:
    """Push an event onto the queue (no-op if bus doesn't exist)."""
    q = _buses.get(response_id)
    if q is not None:
        await q.put(event)


async def close_bus(response_id: str) -> None:
    """Signal stream completion with a sentinel value."""
    q = _buses.get(response_id)
    if q is not None:
        await q.put(_SENTINEL)


def destroy_bus(response_id: str) -> None:
    """Remove the queue from the global registry (call after SSE closes)."""
    _buses.pop(response_id, None)


async def read_events(response_id: str):
    """Async generator that yields events until the sentinel is received."""
    q = _buses.get(response_id)
    if q is None:
        return
    while True:
        event = await q.get()
        if event is _SENTINEL:
            break
        yield event
