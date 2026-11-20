from __future__ import annotations
import hashlib
from collections.abc import Sequence
import httpx
import numpy as np

class DeterministicEmbeddingProvider:
    """Offline embedding for tests/demo. It is a real repeatable transform, not fake retrieval."""
    model_id = "deterministic-hash-v1"
    def __init__(self, dimension: int = 96): self.dimension = dimension
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            v = np.zeros(self.dimension, dtype=np.float32)
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                v[value % self.dimension] += 1.0 if value & 1 else -1.0
            norm = np.linalg.norm(v)
            rows.append(v / norm if norm else v)
        return np.vstack(rows) if rows else np.empty((0, self.dimension), dtype=np.float32)

class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_id, self._model = model_id, SentenceTransformer(model_id)
        self.dimension = self._model.get_sentence_embedding_dimension()
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


class OllamaEmbeddingProvider:
    """Adapter for Ollama's embedding API; it fails loudly when misconfigured."""
    def __init__(self, model_id: str, base_url: str, timeout: float = 45.0):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dimension = int(self.embed(["TreeMind embedding health probe"]).shape[1])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, getattr(self, "dimension", 0)), dtype=np.float32)
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model_id, "input": list(texts)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = response.json().get("embeddings")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ValueError("Ollama returned an invalid embedding response")
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("Ollama returned empty embeddings")
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
