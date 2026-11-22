"""Local pre-cloud secret & PII scrubber using regex, Luhn validation, and Shannon entropy."""

from dataclasses import dataclass, field
import math
import re
from typing import Dict, List, Tuple


@dataclass
class ScrubResult:
    scrubbed_text: str
    redactions: List[Dict[str, str]] = field(default_factory=list)
    placeholder_map: Dict[str, str] = field(default_factory=dict)

    @property
    def redacted_count(self) -> int:
        return len(self.redactions)


def shannon_entropy(token: str) -> float:
    """Compute Shannon entropy in bits per character."""
    if not token:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in token:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(token)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def luhn_valid(digits: str) -> bool:
    """Validate 13-19 digit payment card candidates via the Luhn checksum."""
    if not (13 <= len(digits) <= 19) or not digits.isdigit():
        return False
    total = 0
    rev = digits[::-1]
    for idx, ch in enumerate(rev):
        val = int(ch)
        if idx % 2 == 1:
            val *= 2
            if val > 9:
                val -= 9
        total += val
    return total % 10 == 0


class SecretScrubber:
    """Deterministic pre-cloud redactor for API keys, JWTs, emails, cards, and high-entropy secrets."""

    PATTERNS: List[Tuple[str, re.Pattern]] = [
        ("OPENAI_KEY", re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{20,}\b")),
        ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
        ("GITHUB_TOKEN", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
        ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
        ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
        ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ]

    CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
    SECRET_ASSIGN = re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{18,})['\"]?"
    )

    def __init__(self, entropy_threshold: float = 4.1) -> None:
        self.entropy_threshold = entropy_threshold

    def scrub(self, text: str) -> ScrubResult:
        if not text:
            return ScrubResult(scrubbed_text=text)

        redactions: List[Dict[str, str]] = []
        placeholder_map: Dict[str, str] = {}
        counters: Dict[str, int] = {}

        def make_placeholder(kind: str, raw: str) -> str:
            counters[kind] = counters.get(kind, 0) + 1
            tag = f"<REDACTED_{kind}_{counters[kind]}>"
            placeholder_map[tag] = raw
            redactions.append({"type": kind, "placeholder": tag, "length": str(len(raw))})
            return tag

        out = text
        for kind, pattern in self.PATTERNS:
            out = pattern.sub(lambda m, k=kind: make_placeholder(k, m.group(0)), out)

        def replace_card(match: re.Match) -> str:
            raw = match.group(0)
            digits = re.sub(r"[^0-9]", "", raw)
            if luhn_valid(digits):
                return make_placeholder("CREDIT_CARD", raw)
            return raw

        out = self.CARD_CANDIDATE.sub(replace_card, out)

        def replace_high_entropy(match: re.Match) -> str:
            key_name, val = match.group(1), match.group(2)
            if val.startswith("<REDACTED_"):
                return match.group(0)
            if shannon_entropy(val) >= self.entropy_threshold:
                tag = make_placeholder("HIGH_ENTROPY_SECRET", val)
                return match.group(0).replace(val, tag, 1)
            return match.group(0)

        out = self.SECRET_ASSIGN.sub(replace_high_entropy, out)
        return ScrubResult(scrubbed_text=out, redactions=redactions, placeholder_map=placeholder_map)
