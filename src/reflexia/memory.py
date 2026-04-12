"""Long-term memory primitives and persistence."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

MemoryKind = Literal[
    "pleasant",
    "painful",
    "realization",
    "identity",
    "biography",
    "journal",
]


def now_utc() -> datetime:
    """Return the current time in UTC."""

    return datetime.now(timezone.utc)


class MemoryItem(BaseModel):
    """A single persistent long-term memory item."""

    memory_id: int
    react_step: int
    text: str
    kind: MemoryKind
    created_at: datetime = Field(default_factory=now_utc)


class LongTermMemory:
    """In-memory vector store with JSON/NumPy checkpoint persistence."""

    def __init__(self) -> None:
        self._store: dict[int, MemoryItem] = {}
        self._vectors: dict[int, np.ndarray] = {}
        self._next_id = 0

    def remember(
        self,
        react_step: int,
        text: str,
        kind: MemoryKind,
        embedding: np.ndarray,
    ) -> int:
        """Store a new memory item together with its embedding."""

        memory_id = self._next_id
        self._next_id += 1

        memory_item = MemoryItem(
            memory_id=memory_id,
            react_step=react_step,
            text=text,
            kind=kind,
        )

        self._store[memory_id] = memory_item
        self._vectors[memory_id] = np.asarray(embedding, dtype=np.float32)
        return memory_id

    def recall(self, query_embedding: np.ndarray, top_k: int = 5) -> list[MemoryItem]:
        """Return the most relevant memories for the query embedding."""

        if not self._vectors:
            return []

        memory_ids = list(self._vectors.keys())
        memory_matrix = np.stack([self._vectors[memory_id] for memory_id in memory_ids])
        query_vector = np.asarray(query_embedding, dtype=np.float32)

        scores = memory_matrix @ query_vector
        top_indices = np.argsort(-scores)[: min(top_k, len(memory_ids))]
        return [self._store[memory_ids[index]] for index in top_indices]

    def dump(self, checkpoint_dir: str) -> None:
        """Persist all memory items and vectors to a checkpoint directory."""

        if not self._store:
            warnings.warn("Long-term memory is empty. Nothing was dumped.")
            return

        checkpoint_root = Path(checkpoint_dir)
        memory_items_dir = checkpoint_root / "memory_items"

        checkpoint_root.mkdir(parents=True, exist_ok=True)
        memory_items_dir.mkdir(parents=True, exist_ok=True)

        ordered_memory_ids = sorted(self._store.keys())

        for old_file in memory_items_dir.glob("*.json"):
            old_file.unlink()

        for memory_id in ordered_memory_ids:
            memory_item = self._store[memory_id]
            item_path = memory_items_dir / f"{memory_id}.json"
            with item_path.open("w", encoding="utf-8") as file:
                json.dump(
                    memory_item.model_dump(mode="json"),
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        vectors = np.stack(
            [self._vectors[memory_id] for memory_id in ordered_memory_ids]
        ).astype(np.float32)
        np.save(checkpoint_root / "vectors.npy", vectors)

        meta = {
            "next_id": self._next_id,
            "vector_memory_ids": ordered_memory_ids,
            "num_items": len(ordered_memory_ids),
        }
        with (checkpoint_root / "meta.json").open("w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)

    @staticmethod
    def load(checkpoint_dir: str) -> "LongTermMemory":
        """Load a memory checkpoint from disk."""

        checkpoint_root = Path(checkpoint_dir)
        memory_items_dir = checkpoint_root / "memory_items"
        meta_path = checkpoint_root / "meta.json"
        vectors_path = checkpoint_root / "vectors.npy"

        long_term_memory = LongTermMemory()

        with meta_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)

        long_term_memory._next_id = int(meta["next_id"])
        vector_memory_ids = [int(value) for value in meta["vector_memory_ids"]]
        vectors = np.load(vectors_path)

        for item_path in sorted(
            memory_items_dir.glob("*.json"),
            key=lambda path: int(path.stem),
        ):
            with item_path.open("r", encoding="utf-8") as file:
                item_data = json.load(file)
            memory_item = MemoryItem.model_validate(item_data)
            long_term_memory._store[memory_item.memory_id] = memory_item

        if len(vector_memory_ids) != len(vectors):
            raise ValueError(
                "Mismatch between meta.json and vectors.npy: "
                f"{len(vector_memory_ids)=}, {len(vectors)=}"
            )

        for index, memory_id in enumerate(vector_memory_ids):
            long_term_memory._vectors[memory_id] = np.asarray(
                vectors[index],
                dtype=np.float32,
            )

        return long_term_memory
