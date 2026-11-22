"""Sliding-window runaway-loop detector and dollar burn-rate circuit breaker."""

from collections import deque
from dataclasses import dataclass
import hashlib
import time
from typing import Deque, Dict, Optional, Tuple


@dataclass
class BreakerStatus:
    allowed: bool
    reason: str
    trip_type: Optional[str] = None
    window_spend_usd: float = 0.0
    repeat_count: int = 0


class CircuitBreaker:
    """Protects LLM budgets from infinite agent retry loops and sudden spend spikes."""

    def __init__(
        self,
        max_identical_repeats: int = 4,
        loop_window_sec: float = 30.0,
        max_spend_per_min_usd: float = 2.00,
    ) -> None:
        self.max_identical_repeats = max_identical_repeats
        self.loop_window_sec = loop_window_sec
        self.max_spend_per_min_usd = max_spend_per_min_usd
        # client_id -> deque of (timestamp, prompt_hash)
        self._recent_calls: Dict[str, Deque[Tuple[float, str]]] = {}
        # client_id -> deque of (timestamp, cost_usd)
        self._spend_log: Dict[str, Deque[Tuple[float, float]]] = {}
        self.trips_loop = 0
        self.trips_budget = 0

    @staticmethod
    def _fingerprint(prompt: str) -> str:
        normalized = " ".join(prompt.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def check_and_record(
        self,
        client_id: str,
        prompt: str,
        projected_cost_usd: float,
        now: Optional[float] = None,
    ) -> BreakerStatus:
        ts = time.time() if now is None else now
        fp = self._fingerprint(prompt)

        calls = self._recent_calls.setdefault(client_id, deque())
        while calls and (ts - calls[0][0]) > self.loop_window_sec:
            calls.popleft()

        repeats = sum(1 for _, h in calls if h == fp)
        if repeats >= self.max_identical_repeats:
            self.trips_loop += 1
            return BreakerStatus(
                allowed=False,
                reason=f"Runaway loop detected: identical prompt fired {repeats + 1}x within {self.loop_window_sec:.0f}s",
                trip_type="RUNAWAY_LOOP",
                repeat_count=repeats + 1,
            )

        spends = self._spend_log.setdefault(client_id, deque())
        while spends and (ts - spends[0][0]) > 60.0:
            spends.popleft()

        window_spend = sum(c for _, c in spends)
        if (window_spend + projected_cost_usd) > self.max_spend_per_min_usd:
            self.trips_budget += 1
            return BreakerStatus(
                allowed=False,
                reason=(
                    f"Budget circuit open: 60s spend (${window_spend + projected_cost_usd:.4f}) "
                    f"exceeds cap (${self.max_spend_per_min_usd:.2f}/min)"
                ),
                trip_type="BUDGET_CAP",
                window_spend_usd=round(window_spend, 6),
                repeat_count=repeats,
            )

        calls.append((ts, fp))
        spends.append((ts, projected_cost_usd))
        return BreakerStatus(
            allowed=True,
            reason="OK",
            window_spend_usd=round(window_spend + projected_cost_usd, 6),
            repeat_count=repeats + 1,
        )
