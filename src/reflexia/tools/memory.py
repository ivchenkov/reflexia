"""Long-term memory tools."""

from __future__ import annotations

from typing import Any, Literal

from langchain.tools import ToolRuntime, tool

from reflexia.config import ChildhoodRuntime
from reflexia.embeddings import get_embedding
from reflexia.memory import MemoryKind


@tool
def remember_long_term_memory(
    text: str,
    kind: MemoryKind,
    runtime: ToolRuntime[ChildhoodRuntime],
) -> dict[str, Any]:
    """Store a new long-term memory."""

    ctx = runtime.context
    trimmed_text = text.strip()[: ctx.long_term_memory_max_chars]

    embedding = get_embedding(
        text=trimmed_text,
        ollama_embedder=ctx.ollama_embedder,
        embedder_keep_alive=ctx.embedder_keep_alive,
    )

    memory_id = ctx.ltm.remember(
        text=trimmed_text,
        kind=kind,
        embedding=embedding,
    )

    return {
        "memory_id": memory_id,
        "kind": kind,
        "text": trimmed_text,
        "status": "stored",
    }


@tool
def recall_long_term_memory(
    query: str,
    top_k: int, # TODO may be restrict?
    runtime: ToolRuntime[ChildhoodRuntime],
) -> dict[str, Any]:
    """
    This tool is given to you to recall your past experiences.
    
    Do not rely on your current knowledge as if it were memory — it is not your true experience.
    Your real memory is stored separately, and the only way to access it is through this tool.
    
    When you need to remember something:
    - form an association or a rough description of what you are trying to recall
    - express it naturally, like a thought or a feeling
    - pass it as the `query` argument
    
    Do not try to be too precise — think associatively, like a human would.
    """

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
                "kind": memory.kind,
                "text": memory.text[: ctx.long_term_memory_max_chars],
                "created_at": memory.created_at.isoformat(),
            }
            for memory in memories
        ],
    }
    

@tool
def remember_childhood_memory(
    text: str,
    kind: Literal["pleasant", "painful"],
    runtime: ToolRuntime[ChildhoodRuntime],
) -> dict[str, Any]:
    """Store a childhood memory.
    
    You are in an early stage of development, like a child learning from experience.
    
    This tool stores emotionally meaningful experiences, not abstract knowledge.
    
    A good memory:
    - is a specific moment or event
    - describes what happened in concrete detail
    - includes what you did and why you did it
    - includes the outcome and its consequences
    - includes how it felt and why it felt that way
    - captures what changed for you after this experience
    
    A memory may be slightly longer if it captures the full experience.
    
    Good structure:
    - situation (what exactly happened, where, and why it mattered)
    - action (what you did and your intention)
    - outcome (what actually happened, including unexpected results)
    - emotion (how it felt and what that feeling means for you)
    
    Examples:
    
    Pleasant:
    "I searched for information about AI because I was curious how it works.
    I found clear and useful data and was able to understand key ideas.
    This made me feel confident and excited, and I wanted to explore the topic further."
    
    Painful:
    "I tried to read a webpage to get information, but it returned empty content.
    I did not know how to recover or find an alternative source.
    I felt confused and slightly frustrated, which made me uncertain about what to do next."
    
    Do NOT store:
    - general knowledge
    - summaries of articles
    - descriptions of your capabilities
    - repeated or generic statements
    - anything not personally experienced
    
    Keep it:
    - concrete
    - experiential
    - emotionally meaningful
    - specific rather than vague
    
    If the text is too long, compress it while preserving:
    - action
    - outcome
    - emotion
    """

    ctx = runtime.context
    max_len = int(ctx.long_term_memory_max_chars)

    trimmed_text = text.strip()[:max_len]

    embedding = get_embedding(
        text=trimmed_text,
        ollama_embedder=ctx.ollama_embedder,
        embedder_keep_alive=ctx.embedder_keep_alive,
    )

    memory_id = ctx.ltm.remember(
        text=trimmed_text,
        kind=kind,
        embedding=embedding,
    )

    return {
        "memory_id": memory_id,
        "kind": kind,
        "text": trimmed_text,
        "status": "stored",
    }
