"""
In-memory metrics store — tracks query counts, latency, intent distribution,
and pipeline-level KPIs. Exposed at GET /metrics as JSON.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class MetricsStore:
    """Thread-safe in-memory metrics accumulator."""

    _lock: Lock = field(default_factory=Lock, repr=False)

    # Counters
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    auth_failures: int = 0
    validation_failures: int = 0
    safety_blocks: int = 0
    hallucination_checks_failed: int = 0
    tavily_fallbacks: int = 0
    user_doc_hits: int = 0
    feedback_count: int = 0

    # Intent distribution
    intent_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Latency buckets (per node, in ms)
    node_latency_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    node_latency_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Overall pipeline latency (ms)
    total_latency_sum: float = 0.0
    total_latency_count: int = 0

    # Timestamps
    started_at: float = field(default_factory=time.time)

    # ─── Mutation helpers ────────────────────────────────────────────────

    def record_query(self, intent: str, success: bool, latency_ms: float) -> None:
        with self._lock:
            self.total_queries += 1
            if success:
                self.successful_queries += 1
            else:
                self.failed_queries += 1
            self.intent_counts[intent] += 1
            self.total_latency_sum += latency_ms
            self.total_latency_count += 1

    def record_node_latency(self, node: str, latency_ms: float) -> None:
        with self._lock:
            self.node_latency_sum[node] += latency_ms
            self.node_latency_count[node] += 1

    def record_auth_failure(self) -> None:
        with self._lock:
            self.auth_failures += 1

    def record_validation_failure(self) -> None:
        with self._lock:
            self.validation_failures += 1

    def record_safety_block(self) -> None:
        with self._lock:
            self.safety_blocks += 1

    def record_hallucination_check_failed(self) -> None:
        with self._lock:
            self.hallucination_checks_failed += 1

    def record_tavily_fallback(self) -> None:
        with self._lock:
            self.tavily_fallbacks += 1

    def record_user_doc_hit(self) -> None:
        with self._lock:
            self.user_doc_hits += 1

    def record_feedback(self) -> None:
        with self._lock:
            self.feedback_count += 1

    # ─── Snapshot ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all metrics."""
        with self._lock:
            uptime = time.time() - self.started_at
            avg_latency = (
                self.total_latency_sum / self.total_latency_count
                if self.total_latency_count
                else 0
            )
            node_avg = {
                node: self.node_latency_sum[node] / self.node_latency_count[node]
                for node in self.node_latency_count
            }
            return {
                "uptime_seconds": round(uptime, 1),
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "failed_queries": self.failed_queries,
                "auth_failures": self.auth_failures,
                "validation_failures": self.validation_failures,
                "safety_blocks": self.safety_blocks,
                "hallucination_checks_failed": self.hallucination_checks_failed,
                "tavily_fallbacks": self.tavily_fallbacks,
                "user_doc_hits": self.user_doc_hits,
                "feedback_count": self.feedback_count,
                "intent_distribution": dict(self.intent_counts),
                "avg_pipeline_latency_ms": round(avg_latency, 1),
                "node_avg_latency_ms": {k: round(v, 1) for k, v in node_avg.items()},
            }


# Singleton — imported and used by all pipeline nodes
metrics = MetricsStore()
