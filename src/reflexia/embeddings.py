"""Embedding helpers."""

from __future__ import annotations

import numpy as np
import ollama


def get_embedding(
    text: str,
    ollama_embedder: str = "embeddinggemma",
    embedder_keep_alive: str | int = "30m",
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Generate an embedding for text using an Ollama embedding model."""

    response = ollama.embed(
        model=ollama_embedder,
        input=text,
        keep_alive=embedder_keep_alive,
    )

    embedding = np.asarray(response["embeddings"][0], dtype=np.float32)

    if normalize:
        embedding /= np.linalg.norm(embedding) + epsilon

    return embedding
