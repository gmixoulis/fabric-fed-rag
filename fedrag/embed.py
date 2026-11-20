"""Embedding and in-memory top-k retrieval (ported and extended from softone-rag).

Supports:
- ``OllamaEmbedder``: calls a local or host Ollama server (e.g. ``bge-m3`` or ``embeddinggemma``).
- ``HashEmbedder``: deterministic, zero-network lexical embedder for fast container/offline tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request

import numpy as np

DIM = 4096

STOPWORDS = frozenset(
    (
        "a an the is are was were of in on at to for and or with by from this that these "
        "those it its be do does did how many much what which who whom whose when where "
        "why we you they he she i me my your our their can could should would will than "
        "then so if not no yes one each every all any some show list find give tell get "
        "display please about over under again once"
    ).split()
)

_WORD = re.compile(r"[a-z0-9α-ωάέήίόύώ]+")


def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        return word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    return [
        _stem(word)
        for word in _WORD.findall(text.lower())
        if word not in STOPWORDS and len(word) > 1
    ]


class HashEmbedder:
    """Deterministic lexical embedder for fast container and offline execution."""

    model = "hash-embedder-v1"
    digest = hashlib.sha256(model.encode()).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), DIM), dtype=np.float64)
        for i, text in enumerate(texts):
            for word in _tokens(text):
                idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % DIM
                vecs[i, idx] = 1.0
        return vecs


class OllamaEmbedder:
    """Semantic embedder talking to a local or host Ollama server (bge-m3 / embeddinggemma)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model or os.environ.get("FEDRAG_EMBED_MODEL", "embeddinggemma")
        self.base_url = (
            base_url or os.environ.get("FEDRAG_OLLAMA_URL", "http://host.docker.internal:11434")
        ).rstrip("/")
        self.timeout = timeout

    @property
    def digest(self) -> str:
        with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=self.timeout) as resp:
            models = json.load(resp)["models"]
        name = self.model if ":" in self.model else f"{self.model}:latest"
        digest = next((m["digest"] for m in models if m["name"] == name), "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Missing or invalid Ollama model digest for {name}")
        return digest

    def embed(self, texts: list[str]) -> np.ndarray:
        payload = json.dumps(
            {"model": self.model, "input": texts, "keep_alive": "30m", "options": {"num_gpu": 0}}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return np.asarray(data["embeddings"], dtype=np.float64)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    qn = np.linalg.norm(query)
    if qn == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms == 0.0, 1.0, norms)
    return matrix @ query / (safe * qn)


def top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    if matrix.size == 0:
        return []
    scores = cosine_scores(query, matrix)
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]
