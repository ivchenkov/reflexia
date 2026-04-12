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


def _split_system_and_conversation(
    messages: Sequence[AnyMessage],
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """Split leading system messages from the rest of the conversation."""

    system_messages: list[AnyMessage] = []
    conversation_messages: list[AnyMessage] = []
    seen_non_system = False

    for message in messages:
        if not seen_non_system and isinstance(message, SystemMessage):
            system_messages.append(message)
            continue
        seen_non_system = True
        conversation_messages.append(message)

    return system_messages, conversation_messages


def _split_conversation_into_turns(
    messages: Sequence[AnyMessage],
) -> list[list[AnyMessage]]:
    """Group conversation messages into atomic turns for agent-cycle trimming."""

    turns: list[list[AnyMessage]] = []
    index = 0

    while index < len(messages):
        message = messages[index]

        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            turn = [message]
            index += 1
            while index < len(messages) and isinstance(messages[index], ToolMessage):
                turn.append(messages[index])
                index += 1
            turns.append(turn)
            continue

        if isinstance(message, ToolMessage):
            # Preserve odd/incomplete histories without crashing; trimming will
            # still operate on whole retained chunks.
            turn = [message]
            index += 1
            while index < len(messages) and isinstance(messages[index], ToolMessage):
                turn.append(messages[index])
                index += 1
            turns.append(turn)
            continue

        turns.append([message])
        index += 1

    return turns


def trim_messages_for_model(
    messages: Sequence[AnyMessage],
    context: ExecutionContext,
) -> list[AnyMessage]:
    """Trim history to the token budget while preserving system messages and whole turns."""

    max_input_tokens = (
        context.chat_context_window_tokens
        - context.chat_response_reserve_tokens
        - context.chat_token_safety_margin
    )

    system_messages, conversation_messages = _split_system_and_conversation(messages)
    turns = _split_conversation_into_turns(conversation_messages)

    kept_turns: list[list[AnyMessage]] = []

    for turn in reversed(turns):
        candidate_turns = [turn, *reversed(kept_turns)]
        candidate_messages = system_messages + [
            message for candidate_turn in candidate_turns for message in candidate_turn
        ]

        token_count = qwen_token_counter(
            candidate_messages,
            tokenizer_name=context.chat_tokenizer_name,
        )
        if token_count <= max_input_tokens:
            kept_turns.append(turn)

    # If the budget is too strict to fit even one full turn, keep the latest turn
    # instead of dropping all conversational context.
    if not kept_turns and turns:
        kept_turns = [turns[-1]]

    trimmed_messages = system_messages + [
        message for turn in reversed(kept_turns) for message in turn
    ]
    return trimmed_messages
