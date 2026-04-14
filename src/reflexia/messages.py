"""Message helpers for model invocation."""

from __future__ import annotations

import json
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

from reflexia.config import ChildhoodRuntime

CYCLE_ID_KEY = "cycle_id"


@lru_cache(maxsize=None)
def get_tokenizer(tokenizer_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def _to_qwen_message(message: AnyMessage) -> dict:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content or ""}

    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content or ""}

    if isinstance(message, AIMessage):
        msg = {
            "role": "assistant",
            "content": message.content or "",
        }

        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(
                            tc.get("args", {}),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    },
                    "id": tc.get("id"),
                }
                for tc in message.tool_calls
            ]

        return msg

    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content or "",
            "tool_call_id": message.tool_call_id,
        }

    raise TypeError(f"Unsupported message type: {type(message)}")


def count_tokens_qwen(
    messages: Sequence[AnyMessage],
    *,
    tokenizer_name: str,
) -> int:
    tokenizer = get_tokenizer(tokenizer_name)
    messages = [_to_qwen_message(message) for message in messages]
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )["input_ids"]
    return len(token_ids)


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


def get_usable_input_token_budget(context: ChildhoodRuntime) -> int:
    """Return the maximum number of input tokens available for chat history."""

    return (
        context.chat_context_window_tokens
        - context.chat_response_reserve_tokens
        - context.chat_token_safety_margin
    )


def trim_messages_for_model(
    messages: Sequence[AnyMessage],
    context: ChildhoodRuntime,
) -> list[AnyMessage]:
    """Trim history by removing oldest cycle ids until the token budget is satisfied."""

    max_input_tokens = get_usable_input_token_budget(context)
    message_list = list(messages)

    token_count = count_tokens_qwen(
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
        token_count = count_tokens_qwen(
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
