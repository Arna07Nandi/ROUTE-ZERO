"""RouteZero unified proxy engine and zero-dependency OpenAI-compatible HTTP server."""

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional
import urllib.request

from .breaker import BreakerStatus, CircuitBreaker
from .cache import SemanticCache
from .models import MODEL_CATALOG, RoutingTier, estimate_cost_usd, estimate_tokens
from .router import ComplexityRouter, RoutingDecision
from .scrubber import SecretScrubber


@dataclass
class ProxyResponse:
    status_code: int
    content: str
    model_used: str
    baseline_model: str
    cache_hit: bool
    cache_similarity: float
    complexity_score: float
    tier: int
    actual_cost_usd: float
    baseline_cost_usd: float
    saved_usd: float
    redacted_secrets: int
    breaker_tripped: bool
    breaker_reason: Optional[str]
    overhead_ms: float
    signals: Dict[str, float] = field(default_factory=dict)


class RouteZeroEngine:
    """Orchestrates scrubbing -> circuit breaking -> semantic caching -> complexity routing."""

    def __init__(
        self,
        baseline_model: str = "gpt-6.1-sol",
        scrubber: Optional[SecretScrubber] = None,
        cache: Optional[SemanticCache] = None,
        router: Optional[ComplexityRouter] = None,
        breaker: Optional[CircuitBreaker] = None,
        upstream_fn: Optional[Callable[[str, List[Dict[str, str]]], str]] = None,
    ) -> None:
        self.baseline_model = baseline_model
        self.scrubber = scrubber or SecretScrubber()
        self.cache = cache or SemanticCache()
        self.router = router or ComplexityRouter()
        self.breaker = breaker or CircuitBreaker()
        self.upstream_fn = upstream_fn
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "cache_hits": 0,
            "breaker_blocks": 0,
            "secrets_redacted": 0,
            "total_actual_usd": 0.0,
            "total_baseline_usd": 0.0,
            "tier_counts": {1: 0, 2: 0, 3: 0, 4: 0},
        }

    def _default_upstream(self, model_id: str, messages: List[Dict[str, str]]) -> str:
        """Calls live OpenAI-compatible upstream if configured, else returns deterministic dry-run reply."""
        api_key = os.environ.get("ROUTEZERO_UPSTREAM_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("ROUTEZERO_UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")
        if api_key and os.environ.get("ROUTEZERO_LIVE") == "1":
            payload = json.dumps({"model": model_id, "messages": messages}).encode("utf-8")
            req = urllib.request.Request(
                base_url,
                data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        last_prompt = messages[-1].get("content", "") if messages else ""
        return f"[RouteZero dry-run via {model_id}] Processed prompt ({len(last_prompt)} chars)."

    def handle_chat(
        self,
        messages: List[Dict[str, str]],
        client_id: str = "default",
        force_tier: Optional[int] = None,
        skip_cache: bool = False,
        expected_output_tokens: int = 350,
        now: Optional[float] = None,
    ) -> ProxyResponse:
        t0 = time.perf_counter()
        self.metrics["total_requests"] += 1

        # 1. Pre-cloud secret scrubbing
        scrubbed_messages: List[Dict[str, str]] = []
        total_redactions = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                res = self.scrubber.scrub(content)
                total_redactions += res.redacted_count
                scrubbed_messages.append({**msg, "content": res.scrubbed_text})
            else:
                scrubbed_messages.append(msg)
        self.metrics["secrets_redacted"] += total_redactions

        joined_prompt = "\n".join(m.get("content", "") for m in scrubbed_messages)
        tokens_in = estimate_tokens(joined_prompt)
        baseline_cost = estimate_cost_usd(self.baseline_model, tokens_in, expected_output_tokens)

        # 2. Deterministic complexity routing
        decision: RoutingDecision = self.router.score(scrubbed_messages)
        if force_tier in (1, 2, 3, 4):
            forced = RoutingTier(force_tier)
            decision = RoutingDecision(
                tier=forced,
                model_id=self.router.tier_primary[forced],
                fallback_models=decision.fallback_models,
                complexity_score=decision.complexity_score,
                signals=decision.signals,
                reason=f"Explicit tier override (Tier {force_tier})",
            )

        # 3. Runaway-loop & spend circuit breaker (checked before any upstream spend)
        projected_cost = estimate_cost_usd(decision.model_id, tokens_in, expected_output_tokens)
        breaker_status: BreakerStatus = self.breaker.check_and_record(
            client_id=client_id,
            prompt=joined_prompt,
            projected_cost_usd=projected_cost,
            now=now,
        )
        if not breaker_status.allowed:
            self.metrics["breaker_blocks"] += 1
            self.metrics["total_baseline_usd"] += baseline_cost
            overhead = (time.perf_counter() - t0) * 1000.0
            return ProxyResponse(
                status_code=429,
                content=breaker_status.reason,
                model_used="blocked",
                baseline_model=self.baseline_model,
                cache_hit=False,
                cache_similarity=0.0,
                complexity_score=decision.complexity_score,
                tier=int(decision.tier),
                actual_cost_usd=0.0,
                baseline_cost_usd=round(baseline_cost, 6),
                saved_usd=round(baseline_cost, 6),
                redacted_secrets=total_redactions,
                breaker_tripped=True,
                breaker_reason=breaker_status.reason,
                overhead_ms=round(overhead, 3),
                signals=decision.signals,
            )

        # 4. Semantic cache lookup
        if not skip_cache:
            hit = self.cache.lookup(joined_prompt, namespace=client_id, now=now)
            if hit is not None:
                self.metrics["cache_hits"] += 1
                self.metrics["total_baseline_usd"] += baseline_cost
                overhead = (time.perf_counter() - t0) * 1000.0
                return ProxyResponse(
                    status_code=200,
                    content=hit.response_text,
                    model_used=f"cache:{hit.original_model_id}",
                    baseline_model=self.baseline_model,
                    cache_hit=True,
                    cache_similarity=hit.similarity,
                    complexity_score=decision.complexity_score,
                    tier=int(decision.tier),
                    actual_cost_usd=0.0,
                    baseline_cost_usd=round(baseline_cost, 6),
                    saved_usd=round(baseline_cost, 6),
                    redacted_secrets=total_redactions,
                    breaker_tripped=False,
                    breaker_reason=None,
                    overhead_ms=round(overhead, 3),
                    signals=decision.signals,
                )

        # 5. Execute routed upstream model
        caller = self.upstream_fn or self._default_upstream
        reply = caller(decision.model_id, scrubbed_messages)
        tokens_out = max(expected_output_tokens, estimate_tokens(reply))
        actual_cost = estimate_cost_usd(decision.model_id, tokens_in, tokens_out)
        baseline_cost = estimate_cost_usd(self.baseline_model, tokens_in, tokens_out)

        self.cache.put(
            prompt=joined_prompt,
            response_text=reply,
            model_id=decision.model_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            namespace=client_id,
            now=now,
        )

        self.metrics["total_actual_usd"] += actual_cost
        self.metrics["total_baseline_usd"] += baseline_cost
        self.metrics["tier_counts"][int(decision.tier)] += 1
        overhead = (time.perf_counter() - t0) * 1000.0

        return ProxyResponse(
            status_code=200,
            content=reply,
            model_used=decision.model_id,
            baseline_model=self.baseline_model,
            cache_hit=False,
            cache_similarity=0.0,
            complexity_score=decision.complexity_score,
            tier=int(decision.tier),
            actual_cost_usd=round(actual_cost, 6),
            baseline_cost_usd=round(baseline_cost, 6),
            saved_usd=round(baseline_cost - actual_cost, 6),
            redacted_secrets=total_redactions,
            breaker_tripped=False,
            breaker_reason=None,
            overhead_ms=round(overhead, 3),
            signals=decision.signals,
        )

    def summary_metrics(self) -> Dict[str, Any]:
        baseline = self.metrics["total_baseline_usd"]
        actual = self.metrics["total_actual_usd"]
        savings_pct = ((baseline - actual) / baseline * 100.0) if baseline > 0 else 0.0
        return {
            **self.metrics,
            "total_actual_usd": round(actual, 6),
            "total_baseline_usd": round(baseline, 6),
            "total_saved_usd": round(baseline - actual, 6),
            "savings_pct": round(savings_pct, 2),
            "cache_stats": self.cache.stats(),
        }


def run_server(host: str = "127.0.0.1", port: int = 8088, baseline_model: str = "gpt-6.1-sol") -> None:
    engine = RouteZeroEngine(baseline_model=baseline_model)

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "service": "routezero"})
            elif self.path == "/metrics":
                self._send_json(200, engine.summary_metrics())
            elif self.path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": spec.model_id,
                                "provider": spec.provider,
                                "tier": int(spec.tier),
                                "input_per_1m_usd": spec.input_per_1m_usd,
                                "output_per_1m_usd": spec.output_per_1m_usd,
                            }
                            for spec in MODEL_CATALOG.values()
                        ],
                    },
                )
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                body = json.loads(raw_body)
                if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                    self._send_json(400, {"error": "Request body must be a JSON object with a 'messages' list."})
                    return
            except (ValueError, UnicodeDecodeError) as exc:
                self._send_json(400, {"error": f"Invalid JSON request body: {exc}"})
                return
            client_id = self.headers.get("X-RouteZero-Client", "default")
            force_tier_hdr = self.headers.get("X-RouteZero-Tier")
            force_tier = int(force_tier_hdr) if force_tier_hdr and force_tier_hdr.isdigit() else None
            skip_cache = self.headers.get("X-RouteZero-Skip-Cache", "0") == "1"

            res = engine.handle_chat(
                messages=body.get("messages", []),
                client_id=client_id,
                force_tier=force_tier,
                skip_cache=skip_cache,
            )
            self._send_json(
                res.status_code,
                {
                    "id": f"chatcmpl-rz-{int(time.time() * 1000)}",
                    "object": "chat.completion",
                    "model": res.model_used,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": res.content}, "finish_reason": "stop"}
                    ],
                    "routezero": {
                        "tier": res.tier,
                        "complexity_score": res.complexity_score,
                        "cache_hit": res.cache_hit,
                        "cache_similarity": res.cache_similarity,
                        "actual_cost_usd": res.actual_cost_usd,
                        "baseline_cost_usd": res.baseline_cost_usd,
                        "saved_usd": res.saved_usd,
                        "redacted_secrets": res.redacted_secrets,
                        "breaker_tripped": res.breaker_tripped,
                        "overhead_ms": res.overhead_ms,
                    },
                },
            )

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"RouteZero proxy listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
