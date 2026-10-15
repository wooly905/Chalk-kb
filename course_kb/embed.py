"""Local embeddings (sentence-transformers). Content never leaves the host."""
from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy, imported lazily

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=float)


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(a, float)) @ normalize(np.asarray(b, float)).T


def average_linkage_clusters(vecs: np.ndarray, similarity_threshold: float) -> list[list[int]]:
    """Agglomerative clustering on cosine distance with average linkage.
    Clusters merge only while their average pairwise similarity >= threshold."""
    n = len(vecs)
    if n == 0:
        return []
    sim = cosine_matrix(vecs, vecs)
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        best, best_pair = -2.0, None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = float(sim[np.ix_(clusters[i], clusters[j])].mean())
                if s > best + 1e-12:
                    best, best_pair = s, (i, j)
        if best < similarity_threshold:
            break
        i, j = best_pair
        clusters[i] = sorted(clusters[i] + clusters[j])
        del clusters[j]
    return sorted(clusters, key=lambda c: c[0])
