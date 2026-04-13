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
from transformers import AutoTokenizer

from reflexia.config import ExecutionContext

CYCLE_ID_KEY = "cycle_id"


def _tokenized_length(tokenized: Any) -> int:
    """Extract the token sequence length from common HF return shapes."""

    if hasattr(tokenized, "keys") and "input_ids" in tokenized:
        return _tokenized_length(tokenized["input_ids"])

    if isinstance(tokenized, dict):
        if "input_ids" not in tokenized:
            raise KeyError("Expected 'input_ids' in tokenized chat template output.")
        return _tokenized_length(tokenized["input_ids"])

    if hasattr(tokenized, "shape"):
        shape = tokenized.shape
        if len(shape) == 1:
            return int(shape[0])
        if len(shape) == 0:
            return 1
        return int(shape[-1])

    if isinstance(tokenized, list):
        if not tokenized:
            return 0
        first_item = tokenized[0]
        if isinstance(first_item, list):
            return len(first_item)
        return len(tokenized)

    raise TypeError(
        "Unsupported tokenized chat template output type: "
        f"{type(tokenized).__name__}"
    )


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
    tokenized = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="np",
    )
    return _tokenized_length(tokenized)


def get_cycle_id(message: AnyMessage) -> int | None:
    """Return the cycle id attached to a message, if present."""

    value = (message.additional_kwargs or {}).get(CYCLE_ID_KEY)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def annotate_messages_with_cycle_id(
    messages: Sequence[AnyMessage],
    cycle_id: int,
) -> list[AnyMessage]:
    """Return deep-copied messages annotated with a cycle id."""

    annotated: list[AnyMessage] = []
    for message in messages:
        additional_kwargs = dict(message.additional_kwargs or {})
        additional_kwargs[CYCLE_ID_KEY] = int(cycle_id)
        annotated.append(message.model_copy(update={"additional_kwargs": additional_kwargs}))
    return annotated


def remove_cycles_by_id(
    messages: Sequence[AnyMessage],
    cycle_ids_to_remove: set[int],
) -> list[AnyMessage]:
    """Remove all messages that belong to any of the provided cycle ids."""

    return [
        message
        for message in messages
        if (cycle_id := get_cycle_id(message)) is None
        or cycle_id not in cycle_ids_to_remove
    ]


def get_usable_input_token_budget(context: ExecutionContext) -> int:
    """Return the maximum number of input tokens available for chat history."""

    return (
        context.chat_context_window_tokens
        - context.chat_response_reserve_tokens
        - context.chat_token_safety_margin
    )


def trim_messages_for_model(
    messages: Sequence[AnyMessage],
    context: ExecutionContext,
) -> list[AnyMessage]:
    """Trim history by removing oldest cycle ids until the token budget is satisfied."""

    max_input_tokens = get_usable_input_token_budget(context)
    message_list = list(messages)

    token_count = qwen_token_counter(
        message_list,
        tokenizer_name=context.chat_tokenizer_name,
    )
    if token_count <= max_input_tokens:
        return message_list

    seen_cycle_ids: list[int] = []
    for message in message_list:
        cycle_id = get_cycle_id(message)
        if cycle_id is not None and cycle_id not in seen_cycle_ids:
            seen_cycle_ids.append(cycle_id)

    if not seen_cycle_ids:
        return message_list

    cycle_ids_to_remove: set[int] = set()
    for cycle_id in seen_cycle_ids:
        cycle_ids_to_remove.add(cycle_id)
        candidate_messages = remove_cycles_by_id(message_list, cycle_ids_to_remove)
        token_count = qwen_token_counter(
            candidate_messages,
            tokenizer_name=context.chat_tokenizer_name,
        )
        if token_count <= max_input_tokens:
            return candidate_messages

    # Keep at least the newest cycle if present.
    newest_cycle = seen_cycle_ids[-1]
    candidate_messages = remove_cycles_by_id(message_list, set(seen_cycle_ids[:-1]))
    return candidate_messages if candidate_messages else remove_cycles_by_id(
        message_list,
        {cycle_id for cycle_id in seen_cycle_ids if cycle_id != newest_cycle},
    )
