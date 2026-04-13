from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from reflexia.config import ExecutionContext
from reflexia.graph import AgentState, limit_tool_calls, tag_tool_messages_with_cycle_id
from reflexia.memory import LongTermMemory
from reflexia.messages import annotate_messages_with_cycle_id, get_cycle_id


class DummyRuntime:
    def __init__(self, context):
        self.context = context


def make_context(max_parallel_tool_calls: int = 3) -> ExecutionContext:
    return ExecutionContext(
        llm=None,
        tools=[],
        chat_tokenizer_name="Qwen/Qwen2.5-14B-Instruct",
        chat_context_window_tokens=32768,
        chat_response_reserve_tokens=512,
        chat_token_safety_margin=256,
        max_parallel_tool_calls=max_parallel_tool_calls,
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


def test_limit_tool_calls_clamps_to_context_limit() -> None:
    context = make_context(max_parallel_tool_calls=2)
    runtime = DummyRuntime(context)

    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "t1", "args": {}, "id": "c1", "type": "tool_call"},
            {"name": "t2", "args": {}, "id": "c2", "type": "tool_call"},
            {"name": "t3", "args": {}, "id": "c3", "type": "tool_call"},
        ],
    )
    state: AgentState = {"messages": [message], "react_step": 1}

    update = limit_tool_calls(state, runtime)
    assert update == {}
    assert len(state["messages"][-1].tool_calls) == 2


def test_limit_tool_calls_noop_for_non_ai_message() -> None:
    context = make_context(max_parallel_tool_calls=2)
    runtime = DummyRuntime(context)
    state: AgentState = {
        "messages": [ToolMessage(content="x", tool_call_id="c1")],
        "react_step": 1,
    }

    update = limit_tool_calls(state, runtime)
    assert update == {}


def test_tag_tool_messages_with_cycle_id_tags_only_trailing_untagged() -> None:
    context = make_context()
    runtime = DummyRuntime(context)

    tagged_ai = annotate_messages_with_cycle_id([AIMessage(content="agent")], cycle_id=1)[0]
    tool_1 = ToolMessage(content="r1", tool_call_id="c1")
    tool_2 = ToolMessage(content="r2", tool_call_id="c2")
    state: AgentState = {
        "messages": [SystemMessage(content="sys"), tagged_ai, tool_1, tool_2],
        "react_step": 2,
    }

    update = tag_tool_messages_with_cycle_id(state, runtime)
    assert update == {}
    assert all(get_cycle_id(message) == 2 for message in state["messages"][-2:])
