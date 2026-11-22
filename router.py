"""Deterministic multi-signal prompt complexity scorer and cost-tier router."""

from dataclasses import dataclass, field
import math
import re
from typing import Dict, List, Optional

from .models import DEFAULT_TIER_FALLBACKS, DEFAULT_TIER_PRIMARY, RoutingTier, estimate_tokens


_REASONING_MARKERS = re.compile(
    r"\b(prove|proof|theorem|derive|derivation|invariant|complexity|asymptotic|big-o|"
    r"formal|verify|z3|smt|deadlock|race condition|distributed|consensus|raft|paxos|"
    r"architecture|trade-off|tradeoff|root cause|optimize|counterexample|rigorous)\b",
    re.I,
)
_CODE_MARKERS = re.compile(
    r"(```|\bdef\s+\w+\(|\bclass\s+\w+|\bfunction\s+\w+|\bSELECT\b.+\bFROM\b|"
    r"\basync\s+def\b|\bimport\s+\w+|#include|->|=>|::)",
    re.I | re.S,
)
_SIMPLE_INTENT_MARKERS = re.compile(
    r"\b(classify|sentiment|extract|json|translate|summarize|tldr|bullet points|"
    r"format|rewrite|fix grammar|capitalize|yes or no|label this|categorize)\b",
    re.I,
)
_CONSTRAINT_MARKERS = re.compile(
    r"\b(must|strictly|constraint|at least|at most|edge case|step by step|"
    r"without using|in o\(|time complexity|space complexity)\b",
    re.I,
)


@dataclass
class RoutingDecision:
    tier: RoutingTier
    model_id: str
    fallback_models: List[str]
    complexity_score: float
    signals: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


class ComplexityRouter:
    """Zero-LLM-call deterministic router mapping prompts to cost tiers in <0.5ms."""

    def __init__(
        self,
        t2_threshold: float = 0.28,
        t3_threshold: float = 0.56,
        t4_threshold: float = 0.80,
        tier_primary: Optional[Dict[RoutingTier, str]] = None,
    ) -> None:
        self.t2_threshold = t2_threshold
        self.t3_threshold = t3_threshold
        self.t4_threshold = t4_threshold
        self.tier_primary = dict(DEFAULT_TIER_PRIMARY)
        if tier_primary:
            self.tier_primary.update(tier_primary)

    @staticmethod
    def _bracket_depth(text: str) -> int:
        depth = 0
        max_depth = 0
        for ch in text:
            if ch in "{[(":
                depth += 1
                if depth > max_depth:
                    max_depth = depth
            elif ch in "}])":
                depth = max(0, depth - 1)
        return max_depth

    def score(self, messages: List[Dict[str, str]]) -> RoutingDecision:
        combined = "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        tokens = estimate_tokens(combined)
        turns = len(messages)

        # 1. Context length pressure (log-scaled, saturates near ~4k tokens)
        length_signal = min(1.0, math.log1p(tokens) / math.log1p(4000)) * 0.22

        # 2. Multi-turn conversational state depth
        turn_signal = min(1.0, max(0, turns - 1) / 8.0) * 0.12

        # 3. Structural / AST nesting depth (code, JSON, nested logic)
        depth = self._bracket_depth(combined)
        code_hits = len(_CODE_MARKERS.findall(combined))
        code_signal = min(1.0, (code_hits / 4.0) * 0.7 + (min(depth, 8) / 8.0) * 0.3) * 0.26

        # 4. Reasoning & formal-verification lexical density
        reasoning_hits = len(_REASONING_MARKERS.findall(combined))
        reasoning_signal = min(1.0, reasoning_hits / 3.0) * 0.34

        # 5. Multi-constraint density
        constraint_hits = len(_CONSTRAINT_MARKERS.findall(combined))
        constraint_signal = min(1.0, constraint_hits / 3.0) * 0.16

        # 6. Simple-intent discount (classification/formatting/extraction)
        simple_hits = len(_SIMPLE_INTENT_MARKERS.findall(combined))
        simple_discount = -min(0.18, simple_hits * 0.09) if reasoning_hits == 0 else 0.0

        raw_score = (
            length_signal
            + turn_signal
            + code_signal
            + reasoning_signal
            + constraint_signal
            + simple_discount
        )
        complexity = round(max(0.0, min(1.0, raw_score)), 4)

        if complexity >= self.t4_threshold:
            tier = RoutingTier.FRONTIER
            reason = "High formal/reasoning & structural complexity"
        elif complexity >= self.t3_threshold:
            tier = RoutingTier.STRONG
            reason = "Multi-constraint synthesis or complex code analysis"
        elif complexity >= self.t2_threshold:
            tier = RoutingTier.BALANCED
            reason = "Moderate structured generation or standard coding"
        else:
            tier = RoutingTier.ECONOMY
            reason = "Low-complexity extraction, classification, or short Q&A"

        model_id = self.tier_primary[tier]
        fallbacks = list(DEFAULT_TIER_FALLBACKS[tier])

        return RoutingDecision(
            tier=tier,
            model_id=model_id,
            fallback_models=fallbacks,
            complexity_score=complexity,
            signals={
                "length": round(length_signal, 4),
                "turns": round(turn_signal, 4),
                "code_structure": round(code_signal, 4),
                "reasoning": round(reasoning_signal, 4),
                "constraints": round(constraint_signal, 4),
                "simple_discount": round(simple_discount, 4),
            },
            reason=reason,
        )
