"""Message helpers for model invocation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import trim_messages
from transformers import AutoTokenizer

from reflexia.config import ExecutionContext


@lru_cache(maxsize=None)
def get_tokenizer(tokenizer_name: str) -> Any:
    """Load and cache the tokenizer used for chat-template token counting."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = 10**9
    return tokenizer


def _message_content_to_text(content: Any) -> str:
    """Convert LangChain message content into plain text for token counting."""

    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def to_qwen_chat_message(message: AnyMessage) -> dict[str, Any]:
    """Convert a LangChain message into the structure expected by Qwen."""

    if isinstance(message, SystemMessage):
        return {
            "role": "system",
            "content": _message_content_to_text(message.content),
        }

    if isinstance(message, HumanMessage):
        return {
            "role": "user",
            "content": _message_content_to_text(message.content),
        }

    if isinstance(message, AIMessage):
        qwen_message: dict[str, Any] = {
            "role": "assistant",
            "content": _message_content_to_text(message.content),
        }
        if getattr(message, "tool_calls", None):
            qwen_message["tool_calls"] = message.tool_calls
        return qwen_message

    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": _message_content_to_text(message.content),
            "tool_call_id": message.tool_call_id,
        }

    return {
        "role": "user",
        "content": _message_content_to_text(getattr(message, "content", message)),
    }


def qwen_token_counter(
    messages: Sequence[AnyMessage],
    *,
    tokenizer_name: str,
) -> int:
    """Count tokens for a sequence of chat messages using Qwen chat formatting."""

    tokenizer = get_tokenizer(tokenizer_name)
    chat_messages = [to_qwen_chat_message(message) for message in messages]
    encoded = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="np",
    )
    return int(encoded["input_ids"].shape[-1])


def trim_messages_for_model(
    messages: Sequence[AnyMessage],
    context: ExecutionContext,
) -> list[AnyMessage]:
    """Trim chat history to fit the configured model budget while preserving system messages."""

    max_input_tokens = (
        context.chat_context_window_tokens
        - context.chat_response_reserve_tokens
        - context.chat_token_safety_margin
    )

    trimmed = trim_messages(
        list(messages),
        strategy="last",
        token_counter=lambda batch: qwen_token_counter(
            batch,
            tokenizer_name=context.chat_tokenizer_name,
        ),
        max_tokens=max_input_tokens,
        include_system=True,
        allow_partial=False,
    )
    return list(trimmed)
