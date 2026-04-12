"""Long-term memory tools."""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool

from reflexia.config import ExecutionContext
from reflexia.embeddings import get_embedding
from reflexia.memory import MemoryKind


@tool
def remember_long_term_memory(
    text: str,
    kind: MemoryKind,
    runtime: ToolRuntime[ExecutionContext],
) -> dict[str, Any]:
    """Store a new long-term memory."""

    ctx = runtime.context
    state = runtime.state

    react_step = int(state.get("react_step", 0))
    trimmed_text = text.strip()[: ctx.long_term_memory_max_chars]

    embedding = get_embedding(
        text=trimmed_text,
        ollama_embedder=ctx.ollama_embedder,
        embedder_keep_alive=ctx.embedder_keep_alive,
    )

    memory_id = ctx.ltm.remember(
        react_step=react_step,
        text=trimmed_text,
        kind=kind,
        embedding=embedding,
    )

    return {
        "memory_id": memory_id,
        "react_step": react_step,
        "kind": kind,
        "text": trimmed_text,
        "status": "stored",
    }


@tool
def recall_long_term_memory(
    query: str,
    top_k: int,
    runtime: ToolRuntime[ExecutionContext],
) -> dict[str, Any]:
    """Recall relevant long-term memories for a query."""

    ctx = runtime.context

    query_embedding = get_embedding(
        text=query,
        ollama_embedder=ctx.ollama_embedder,
        embedder_keep_alive=ctx.embedder_keep_alive,
    )

    memories = ctx.ltm.recall(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    return {
        "query": query,
        "top_k": top_k,
        "num_results": len(memories),
        "memories": [
            {
                "memory_id": memory.memory_id,
                "react_step": memory.react_step,
                "kind": memory.kind,
                "text": memory.text[: ctx.long_term_memory_max_chars],
                "created_at": memory.created_at.isoformat(),
            }
            for memory in memories
        ],
    }
