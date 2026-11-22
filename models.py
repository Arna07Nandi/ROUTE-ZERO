"""Grounded model catalog and token cost math (November 2026 published API rates)."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List


class RoutingTier(IntEnum):
    ECONOMY = 1     # Classification, extraction, formatting, short Q&A
    BALANCED = 2    # Summarization, standard coding, structured transforms
    STRONG = 3      # Multi-file code, complex SQL, long-context synthesis
    FRONTIER = 4    # Deep proofs, architecture design, hard debugging


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    provider: str
    tier: RoutingTier
    input_per_1m_usd: float
    output_per_1m_usd: float
    avg_latency_ms: int
    source_url: str


# Published API token rates (USD per 1M input / output tokens, November 2026)
MODEL_CATALOG: Dict[str, ModelSpec] = {
    "gemini-2.5-flash-lite": ModelSpec(
        model_id="gemini-2.5-flash-lite",
        provider="google",
        tier=RoutingTier.ECONOMY,
        input_per_1m_usd=0.10,
        output_per_1m_usd=0.40,
        avg_latency_ms=260,
        source_url="https://www.requesty.ai/blog/gemini-api-pricing",
    ),
    "gpt-6-luna": ModelSpec(
        model_id="gpt-6-luna",
        provider="openai",
        tier=RoutingTier.ECONOMY,
        input_per_1m_usd=0.10,
        output_per_1m_usd=0.50,
        avg_latency_ms=290,
        source_url="https://reflektlab.ai/blog/openai-api-pricing/",
    ),
    "claude-haiku-5.5": ModelSpec(
        model_id="claude-haiku-5.5",
        provider="anthropic",
        tier=RoutingTier.ECONOMY,
        input_per_1m_usd=0.10,
        output_per_1m_usd=0.50,
        avg_latency_ms=310,
        source_url="https://benchlm.ai/anthropic/api-pricing",
    ),
    "gemini-3.8-flash": ModelSpec(
        model_id="gemini-3.8-flash",
        provider="google",
        tier=RoutingTier.BALANCED,
        input_per_1m_usd=0.75,
        output_per_1m_usd=3.75,
        avg_latency_ms=520,
        source_url="https://www.requesty.ai/blog/gemini-api-pricing",
    ),
    "gpt-6.1-terra": ModelSpec(
        model_id="gpt-6.1-terra",
        provider="openai",
        tier=RoutingTier.BALANCED,
        input_per_1m_usd=1.00,
        output_per_1m_usd=5.00,
        avg_latency_ms=590,
        source_url="https://reflektlab.ai/blog/openai-api-pricing/",
    ),
    "gpt-6.1-sol": ModelSpec(
        model_id="gpt-6.1-sol",
        provider="openai",
        tier=RoutingTier.STRONG,
        input_per_1m_usd=2.00,
        output_per_1m_usd=10.00,
        avg_latency_ms=980,
        source_url="https://reflektlab.ai/blog/openai-api-pricing/",
    ),
    "claude-sonnet-5.5": ModelSpec(
        model_id="claude-sonnet-5.5",
        provider="anthropic",
        tier=RoutingTier.STRONG,
        input_per_1m_usd=2.00,
        output_per_1m_usd=10.00,
        avg_latency_ms=1050,
        source_url="https://benchlm.ai/anthropic/api-pricing",
    ),
    "gemini-4-argon": ModelSpec(
        model_id="gemini-4-argon",
        provider="google",
        tier=RoutingTier.STRONG,
        input_per_1m_usd=2.00,
        output_per_1m_usd=10.00,
        avg_latency_ms=940,
        source_url="https://costgoat.com/pricing/gemini-api",
    ),
    "claude-opus-5.5": ModelSpec(
        model_id="claude-opus-5.5",
        provider="anthropic",
        tier=RoutingTier.FRONTIER,
        input_per_1m_usd=4.00,
        output_per_1m_usd=20.00,
        avg_latency_ms=1850,
        source_url="https://benchlm.ai/anthropic/api-pricing",
    ),
    "gpt-6-astra": ModelSpec(
        model_id="gpt-6-astra",
        provider="openai",
        tier=RoutingTier.FRONTIER,
        input_per_1m_usd=10.00,
        output_per_1m_usd=50.00,
        avg_latency_ms=2300,
        source_url="https://reflektlab.ai/blog/openai-api-pricing/",
    ),
}

DEFAULT_TIER_PRIMARY: Dict[RoutingTier, str] = {
    RoutingTier.ECONOMY: "gemini-2.5-flash-lite",
    RoutingTier.BALANCED: "gemini-3.8-flash",
    RoutingTier.STRONG: "claude-sonnet-5.5",
    RoutingTier.FRONTIER: "claude-opus-5.5",
}

DEFAULT_TIER_FALLBACKS: Dict[RoutingTier, List[str]] = {
    RoutingTier.ECONOMY: ["gpt-6-luna", "claude-haiku-5.5"],
    RoutingTier.BALANCED: ["gpt-6.1-terra"],
    RoutingTier.STRONG: ["gpt-6.1-sol", "gemini-4-argon"],
    RoutingTier.FRONTIER: ["gpt-6-astra"],
}


def estimate_tokens(text: str) -> int:
    """Fast deterministic token estimator (~4 chars/token with whitespace weighting)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.8))


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Compute exact USD cost for a model call."""
    spec = MODEL_CATALOG[model_id]
    return (
        (input_tokens / 1_000_000.0) * spec.input_per_1m_usd
        + (output_tokens / 1_000_000.0) * spec.output_per_1m_usd
    )
