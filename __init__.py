"""RouteZero: Local-first algorithmic LLM cost-routing proxy and multi-file code auditor."""

from .models import MODEL_CATALOG, ModelSpec, RoutingTier, estimate_cost_usd
from .scrubber import SecretScrubber, ScrubResult
from .cache import SemanticCache, CacheHit
from .router import ComplexityRouter, RoutingDecision
from .breaker import CircuitBreaker, BreakerStatus
from .proxy import RouteZeroEngine, ProxyResponse
from .auditor import CodeAuditor, AuditFinding, render_ascii_mindmap

__all__ = [
    "MODEL_CATALOG",
    "ModelSpec",
    "RoutingTier",
    "estimate_cost_usd",
    "SecretScrubber",
    "ScrubResult",
    "SemanticCache",
    "CacheHit",
    "ComplexityRouter",
    "RoutingDecision",
    "CircuitBreaker",
    "BreakerStatus",
    "RouteZeroEngine",
    "ProxyResponse",
    "CodeAuditor",
    "AuditFinding",
    "render_ascii_mindmap",
]
