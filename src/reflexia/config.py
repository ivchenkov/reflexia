"""Runtime configuration for Reflexia."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reflexia.memory import LongTermMemory


@dataclass
class ExecutionContext:
    """Execution-time dependencies shared across the LangGraph runtime."""

    llm: Any
    tools: list[Any]
    chat_tokenizer_name: str
    chat_context_window_tokens: int
    chat_response_reserve_tokens: int
    chat_token_safety_margin: int
    max_parallel_tool_calls: int
    ltm: LongTermMemory
    ltm_path: str
    max_react_steps: int
    ollama_embedder: str
    embedder_keep_alive: str | int
    searxng_url: str
    web_timeout_sec: int
    web_search_max_results: int
    webpage_max_chars: int
    long_term_memory_max_chars: int


def get_default_tools() -> list[Any]:
    """Return the default tool set used by the prototype."""

    from reflexia.tools.memory import (
        recall_long_term_memory,
        remember_long_term_memory,
    )
    from reflexia.tools.web import read_webpage, search_web

    return [
        search_web,
        read_webpage,
        remember_long_term_memory,
        recall_long_term_memory,
    ]


def create_default_execution_context() -> ExecutionContext:
    """Build the default execution context used in the original notebook."""

    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model="qwen2.5:14b-instruct",
        temperature=0.8,
        reasoning=False,
        num_ctx=32768,
        num_predict=300,
    )

    return ExecutionContext(
        llm=llm,
        tools=get_default_tools(),
        chat_tokenizer_name="Qwen/Qwen2.5-14B-Instruct",
        chat_context_window_tokens=32768,
        chat_response_reserve_tokens=512,
        chat_token_safety_margin=256,
        max_parallel_tool_calls=3,
        ltm=LongTermMemory(),
        ltm_path="./memory_test/",
        max_react_steps=50,
        ollama_embedder="embeddinggemma",
        embedder_keep_alive="30m",
        searxng_url="http://localhost:8080/search",
        web_timeout_sec=15,
        web_search_max_results=5,
        webpage_max_chars=2000,
        long_term_memory_max_chars=1800,
    )
