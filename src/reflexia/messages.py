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
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def _to_qwen_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Convert a LangChain tool call into the structure expected by Qwen templates."""

    if "function" in tool_call:
        return tool_call

    qwen_tool_call = {
        "type": "function",
        "function": {
            "name": tool_call["name"],
            "arguments": tool_call.get("args", {}),
        },
    }
    if tool_call.get("id") is not None:
        qwen_tool_call["id"] = tool_call["id"]
    return qwen_tool_call


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
            qwen_message["tool_calls"] = [
                _to_qwen_tool_call(tool_call) for tool_call in message.tool_calls
            ]
        return qwen_message

    if isinstance(message, ToolMessage):
        qwen_message = {
            "role": "tool",
            "content": _message_content_to_text(message.content),
        }
        if getattr(message, "tool_call_id", None):
            qwen_message["tool_call_id"] = message.tool_call_id
        if getattr(message, "name", None):
            qwen_message["tool_name"] = message.name
        return qwen_message

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
    input_ids = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(input_ids)


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
        start_on="human",
        allow_partial=False,
    )
    return list(trimmed)
