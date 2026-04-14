"""Long-term memory primitives and persistence."""

from __future__ import annotations

import json
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

MemoryKind = Literal[
    "pleasant",
    "painful",
    "world",
    "insight",
    "reflexia",
]


def now_utc() -> datetime:
    """Return the current time in UTC."""

    return datetime.now(timezone.utc)


def build_memory_id() -> str:
    """Generate a stable unique ID suitable for filesystem persistence."""

    timestamp = now_utc().strftime("%Y%m%dT%H%M%S%fZ")
    suffix = uuid.uuid4().hex[:12]
    return f"mem_{timestamp}_{suffix}"


class MemoryItem(BaseModel):
    """A single persistent long-term memory item."""

    memory_id: str
    react_step: int
    text: str
    kind: MemoryKind
    created_at: datetime = Field(default_factory=now_utc)


class LongTermMemory:
    """In-memory vector store with append-only filesystem persistence."""

    def __init__(self, storage_dir: str | None = None) -> None:
        self._store: dict[str, MemoryItem] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._storage_root = Path(storage_dir) if storage_dir else None
        if self._storage_root:
            self._load_from_storage()

    def _items_dir(self) -> Path:
        if not self._storage_root:
            raise ValueError("Storage directory is not configured.")
        return self._storage_root / "items"

    def _vectors_dir(self) -> Path:
        if not self._storage_root:
            raise ValueError("Storage directory is not configured.")
        return self._storage_root / "vectors"

    def _ensure_storage_dirs(self) -> None:
        if not self._storage_root:
            return
        self._items_dir().mkdir(parents=True, exist_ok=True)
        self._vectors_dir().mkdir(parents=True, exist_ok=True)

    def _load_from_storage(self) -> None:
        self._ensure_storage_dirs()

        for item_path in sorted(self._items_dir().glob("*.json")):
            with item_path.open("r", encoding="utf-8") as file:
                item_data = json.load(file)
            memory_item = MemoryItem.model_validate(item_data)
            self._store[memory_item.memory_id] = memory_item

            vector_path = self._vectors_dir() / f"{memory_item.memory_id}.npy"
            if vector_path.exists():
                self._vectors[memory_item.memory_id] = np.asarray(
                    np.load(vector_path),
                    dtype=np.float32,
                )

    def _persist_memory(self, item: MemoryItem, vector: np.ndarray) -> None:
        if not self._storage_root:
            return
        self._ensure_storage_dirs()

        item_path = self._items_dir() / f"{item.memory_id}.json"
        with item_path.open("w", encoding="utf-8") as file:
            json.dump(item.model_dump(mode="json"), file, ensure_ascii=False, indent=2)

        vector_path = self._vectors_dir() / f"{item.memory_id}.npy"
        np.save(vector_path, np.asarray(vector, dtype=np.float32))

    def remember(
        self,
        react_step: int,
        text: str,
        kind: MemoryKind,
        embedding: np.ndarray,
    ) -> str:
        """Store a new memory item together with its embedding."""

        memory_id = build_memory_id()
        memory_item = MemoryItem(
            memory_id=memory_id,
            react_step=react_step,
            text=text,
            kind=kind,
        )
        memory_vector = np.asarray(embedding, dtype=np.float32)

        self._store[memory_id] = memory_item
        self._vectors[memory_id] = memory_vector
        self._persist_memory(memory_item, memory_vector)
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
        """Persist in-memory items into append-only storage directory."""

        if not self._store:
            warnings.warn("Long-term memory is empty. Nothing was dumped.")
            return

        root = Path(checkpoint_dir)
        items_dir = root / "items"
        vectors_dir = root / "vectors"
        items_dir.mkdir(parents=True, exist_ok=True)
        vectors_dir.mkdir(parents=True, exist_ok=True)

        for memory_id, memory_item in self._store.items():
            item_path = items_dir / f"{memory_id}.json"
            if not item_path.exists():
                with item_path.open("w", encoding="utf-8") as file:
                    json.dump(
                        memory_item.model_dump(mode="json"),
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )

            vector_path = vectors_dir / f"{memory_id}.npy"
            if not vector_path.exists():
                np.save(vector_path, np.asarray(self._vectors[memory_id], dtype=np.float32))

    @staticmethod
    def load(checkpoint_dir: str) -> "LongTermMemory":
        """Load a memory store from filesystem."""

        return LongTermMemory(storage_dir=checkpoint_dir)
