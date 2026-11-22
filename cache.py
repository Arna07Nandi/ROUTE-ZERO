"""Zero-external-dependency semantic cache using signed hashed n-gram projections and LRU/TTL."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import re
import time
from typing import Dict, List, Optional

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - pure stdlib fallback
    np = None


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _embed_text(text: str, dim: int = 256):
    """Deterministic sub-millisecond dense vector via signed feature hashing (word + char 3-grams)."""
    normalized = " ".join(_TOKEN_RE.findall(text.lower()))
    if np is not None:
        vec = np.zeros(dim, dtype=np.float32)
    else:
        vec = [0.0] * dim

    if not normalized:
        return vec

    tokens = normalized.split()
    features: List[str] = list(tokens)
    for i in range(len(tokens) - 1):
        features.append(f"{tokens[i]}_{tokens[i + 1]}")
    compact = normalized.replace(" ", "_")
    for i in range(max(0, len(compact) - 2)):
        features.append("c:" + compact[i : i + 3])

    for feat in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        idx = h % dim
        sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
        weight = 2.0 if not feat.startswith("c:") else 0.35
        vec[idx] += sign * weight

    if np is not None:
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
    else:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
    return vec


def _dot(a, b) -> float:
    if np is not None:
        return float(np.dot(a, b))
    return sum(x * y for x, y in zip(a, b))


def _is_nonzero(vec) -> bool:
    if np is not None:
        return bool(np.any(vec))
    return any(v != 0.0 for v in vec)


@dataclass
class CacheEntry:
    key: str
    prompt: str
    response_text: str
    model_id: str
    vector: object
    created_at: float
    saved_tokens_in: int
    saved_tokens_out: int
    hits: int = 0


@dataclass
class CacheHit:
    response_text: str
    original_model_id: str
    similarity: float
    exact: bool
    cache_key: str


class SemanticCache:
    """In-memory exact + cosine-similarity semantic cache with LRU eviction and TTL."""

    def __init__(
        self,
        similarity_threshold: float = 0.88,
        max_entries: int = 2048,
        ttl_seconds: float = 3600.0,
        dim: int = 256,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.dim = dim
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    def _evict_expired(self, now: float) -> None:
        expired = [k for k, v in self._entries.items() if (now - v.created_at) > self.ttl_seconds]
        for k in expired:
            del self._entries[k]

    @staticmethod
    def _exact_key(prompt: str, namespace: str = "default") -> str:
        canonical = " ".join(prompt.strip().lower().split())
        return hashlib.sha256(f"{namespace}::{canonical}".encode("utf-8")).hexdigest()[:24]

    def lookup(self, prompt: str, namespace: str = "default", now: Optional[float] = None) -> Optional[CacheHit]:
        ts = time.time() if now is None else now
        self._evict_expired(ts)

        exact_key = self._exact_key(prompt, namespace)
        if exact_key in self._entries:
            entry = self._entries[exact_key]
            entry.hits += 1
            self._entries.move_to_end(exact_key)
            return CacheHit(
                response_text=entry.response_text,
                original_model_id=entry.model_id,
                similarity=1.0,
                exact=True,
                cache_key=exact_key,
            )

        if not self._entries:
            return None

        query_vec = _embed_text(prompt, dim=self.dim)
        if not _is_nonzero(query_vec):
            return None

        best_entry: Optional[CacheEntry] = None
        best_sim = -1.0
        for entry in self._entries.values():
            sim = _dot(query_vec, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.similarity_threshold:
            best_entry.hits += 1
            self._entries.move_to_end(best_entry.key)
            return CacheHit(
                response_text=best_entry.response_text,
                original_model_id=best_entry.model_id,
                similarity=round(best_sim, 4),
                exact=False,
                cache_key=best_entry.key,
            )
        return None

    def put(
        self,
        prompt: str,
        response_text: str,
        model_id: str,
        tokens_in: int,
        tokens_out: int,
        namespace: str = "default",
        now: Optional[float] = None,
    ) -> str:
        ts = time.time() if now is None else now
        self._evict_expired(ts)
        key = self._exact_key(prompt, namespace)
        vec = _embed_text(prompt, dim=self.dim)
        self._entries[key] = CacheEntry(
            key=key,
            prompt=prompt,
            response_text=response_text,
            model_id=model_id,
            vector=vec,
            created_at=ts,
            saved_tokens_in=tokens_in,
            saved_tokens_out=tokens_out,
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return key

    def stats(self) -> Dict[str, float]:
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "threshold": self.similarity_threshold,
        }
