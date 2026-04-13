from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from reflexia.config import ExecutionContext
from reflexia.memory import LongTermMemory
from reflexia.messages import (
    annotate_messages_with_cycle_id,
    get_cycle_id,
    get_usable_input_token_budget,
    remove_cycles_by_id,
    to_qwen_chat_message,
    trim_messages_for_model,
)


def make_context() -> ExecutionContext:
    return ExecutionContext(
        llm=None,
        tools=[],
        chat_tokenizer_name="Qwen/Qwen2.5-14B-Instruct",
        chat_context_window_tokens=500,
        chat_response_reserve_tokens=100,
        chat_token_safety_margin=100,
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


def test_to_qwen_chat_message_supports_all_message_types() -> None:
    system = SystemMessage(content="sys")
    human = HumanMessage(content="hi")
    ai = AIMessage(
        content="answer",
        tool_calls=[
            {"name": "search_web", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}
        ],
    )
    tool = ToolMessage(content="tool-result", tool_call_id="c1")

    assert to_qwen_chat_message(system)["role"] == "system"
    assert to_qwen_chat_message(human)["role"] == "user"
    assert to_qwen_chat_message(ai)["role"] == "assistant"
    assert to_qwen_chat_message(tool)["role"] == "tool"


def test_annotate_and_remove_cycles() -> None:
    messages = [HumanMessage(content="a"), AIMessage(content="b")]
    tagged = annotate_messages_with_cycle_id(messages, cycle_id=7)

    assert all(get_cycle_id(message) == 7 for message in tagged)

    remaining = remove_cycles_by_id(tagged, {7})
    assert remaining == []


def test_trim_messages_for_model_removes_oldest_cycles(monkeypatch: Any) -> None:
    context = make_context()
    assert get_usable_input_token_budget(context) == 300

    def fake_counter(messages: list[Any], *, tokenizer_name: str) -> int:
        return len(messages) * 100

    monkeypatch.setattr("reflexia.messages.qwen_token_counter", fake_counter)

    system = SystemMessage(content="sys")
    c1 = annotate_messages_with_cycle_id([HumanMessage(content="c1")], 1)[0]
    c2 = annotate_messages_with_cycle_id([AIMessage(content="c2")], 2)[0]
    c3 = annotate_messages_with_cycle_id([ToolMessage(content="c3", tool_call_id="x")], 3)[0]
    c4 = annotate_messages_with_cycle_id([AIMessage(content="c4")], 4)[0]
    messages = [system, c1, c2, c3, c4]

    trimmed = trim_messages_for_model(messages, context)
    trimmed_cycles = [get_cycle_id(message) for message in trimmed if get_cycle_id(message) is not None]

    assert isinstance(trimmed[0], SystemMessage)
    assert trimmed_cycles == [3, 4]


def test_trim_messages_for_model_keeps_newest_cycle_if_budget_too_small(
    monkeypatch: Any,
) -> None:
    context = make_context()

    def fake_counter(messages: list[Any], *, tokenizer_name: str) -> int:
        return len(messages) * 1000

    monkeypatch.setattr("reflexia.messages.qwen_token_counter", fake_counter)

    system = SystemMessage(content="sys")
    c1 = annotate_messages_with_cycle_id([HumanMessage(content="c1")], 1)[0]
    c2 = annotate_messages_with_cycle_id([AIMessage(content="c2")], 2)[0]
    messages = [system, c1, c2]

    trimmed = trim_messages_for_model(messages, context)
    trimmed_cycles = [get_cycle_id(message) for message in trimmed if get_cycle_id(message) is not None]

    assert trimmed_cycles == [2]
