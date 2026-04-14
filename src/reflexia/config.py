"""Childhood runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexia.memory import LongTermMemory


@dataclass
class ChildhoodRuntime:
    """Execution-time dependencies for the childhood loop."""

    llm: Any
    tools: list[Any]
    ltm: LongTermMemory
    ltm_path: str
    chat_tokenizer_name: str
    chat_context_window_tokens: int
    chat_response_reserve_tokens: int
    chat_token_safety_margin: int
    max_react_steps: int
    ollama_embedder: str
    embedder_keep_alive: str | int
    searxng_url: str
    web_timeout_sec: int
    web_search_max_results: int
    webpage_max_chars: int
    long_term_memory_max_chars: int
    max_parallel_tool_calls: int


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """Load key/value pairs from a .env file into process environment."""

    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def setup_langsmith_from_env() -> None:
    """Configure LangSmith-related environment variables from .env if present."""

    _load_dotenv()

    mapping = {
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_ENDPOINT": "https://eu.api.smith.langchain.com",
        "LANGSMITH_API_KEY": "",
        "LANGSMITH_PROJECT": "reflexia",
    }
    for key, default_value in mapping.items():
        value = os.getenv(key, default_value)
        if value:
            os.environ[key] = value


def get_childhood_tools() -> list[Any]:
    """Return tools used in the childhood loop."""

    from reflexia.tools.web import read_webpage, search_web

    return [read_webpage, search_web]


def create_childhood_runtime() -> ChildhoodRuntime:
    """Build the default childhood runtime from environment variables."""

    _load_dotenv()
    setup_langsmith_from_env()

    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=os.getenv("REFLEXIA_MODEL", "qwen3.5:9b"),
        temperature=float(os.getenv("REFLEXIA_TEMPERATURE", "0.8")),
        num_predict=int(os.getenv("REFLEXIA_NUM_PREDICT", "400")),
        reasoning=os.getenv("REFLEXIA_REASONING", "true").lower() in {"1", "true", "yes"},
        num_ctx=int(os.getenv("REFLEXIA_NUM_CTX", "32768")),
    )

    memory_root = os.getenv("REFLEXIA_MEMORY_PATH", "./memory_test/childhood")
    ltm = LongTermMemory(storage_dir=memory_root)

    return ChildhoodRuntime(
        llm=llm,
        tools=get_childhood_tools(),
        ltm=ltm,
        ltm_path=memory_root,
        chat_tokenizer_name=os.getenv("REFLEXIA_TOKENIZER_NAME", "Qwen/Qwen2.5-14B-Instruct"),
        chat_context_window_tokens=int(os.getenv("REFLEXIA_NUM_CTX", "32768")),
        chat_response_reserve_tokens=int(os.getenv("REFLEXIA_RESPONSE_RESERVE", "512")),
        chat_token_safety_margin=int(os.getenv("REFLEXIA_TOKEN_SAFETY_MARGIN", "256")),
        max_react_steps=int(os.getenv("REFLEXIA_MAX_REACT_STEPS", "10")),
        ollama_embedder=os.getenv("REFLEXIA_EMBEDDER_MODEL", "embeddinggemma"),
        embedder_keep_alive=os.getenv("REFLEXIA_EMBEDDER_KEEP_ALIVE", "30m"),
        searxng_url=os.getenv("REFLEXIA_SEARXNG_URL", "http://localhost:8080/search"),
        web_timeout_sec=int(os.getenv("REFLEXIA_WEB_TIMEOUT_SEC", "15")),
        web_search_max_results=int(os.getenv("REFLEXIA_WEB_MAX_RESULTS", "5")),
        webpage_max_chars=int(os.getenv("REFLEXIA_WEBPAGE_MAX_CHARS", "2000")),
        long_term_memory_max_chars=int(os.getenv("REFLEXIA_MEMORY_MAX_CHARS", "800")),
        max_parallel_tool_calls=int(os.getenv("REFLEXIA_MAX_PARALLEL_TOOLS", "4")),
    )
