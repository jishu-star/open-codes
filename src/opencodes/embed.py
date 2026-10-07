"""Text embedders. Cosine distance is the similarity used by the merger."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one vector per text, shape (n, d)."""


class HashingEmbedder:
    """Deterministic character-trigram embedder.

    Identical strings land on the same vector, so repeated labels merge. It is
    not a substitute for the paper's ``mxbai-embed-large`` model; use
    :class:`SentenceTransformerEmbedder` when the wording of codes matters.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])

    def _vector(self, text: str) -> np.ndarray:
        folded = " ".join(text.casefold().split())
        padded = f"  {folded}  "
        grams = [padded[i : i + 3] for i in range(max(len(padded) - 2, 1))]
        vector = np.zeros(self.dimensions, dtype=float)
        for gram in grams:
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return vector / norm


class TableEmbedder:
    """Lookup embedder for tests and fully controlled examples.

    Keys may be the raw label or the full ``Label: ...\\nDefinition: ...`` string.
    Unknown label-definition strings fall back to the label key, then to ``default``.
    """

    def __init__(
        self,
        table: dict[str, list[float] | np.ndarray],
        default: list[float] | None = None,
    ) -> None:
        if not table and default is None:
            raise ValueError("table embedder needs at least one vector")
        self.table = {key: np.asarray(value, dtype=float) for key, value in table.items()}
        width = len(next(iter(self.table.values()))) if self.table else len(default or [])
        self.default = (
            np.zeros(width, dtype=float) if default is None else np.asarray(default, dtype=float)
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = [self._lookup(text) for text in texts]
        return np.vstack(rows)

    def _lookup(self, text: str) -> np.ndarray:
        if text in self.table:
            return self.table[text]
        label = text
        if text.startswith("Label:"):
            label = text.split("\n", 1)[0].removeprefix("Label:").strip()
        return self.table.get(label, self.default)


class SentenceTransformerEmbedder:
    """Wrapper around a Sentence Transformers model.

    The paper embeds with ``mxbai-embed-large``. A smaller model is enough for
    exploration; pass its name as ``model_name``.
    """

    def __init__(self, model_name: str = "mixedbread-ai/mxbai-embed-large-v1") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install the embeddings extra: pip install 'opencodes[embeddings]'"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=float)


def cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance. Identical directions have distance 0."""
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("expected a 2-d embedding matrix")
    if matrix.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = matrix / norms
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return np.maximum(distance, 0.0)


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_TOKEN.findall(left.casefold()))
    right_tokens = set(_TOKEN.findall(right.casefold()))
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)
