"""LangGraph graph definition for Reflexia."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from reflexia.config import ExecutionContext
from reflexia.embeddings import get_embedding
from reflexia.messages import (
    annotate_messages_with_cycle_id,
    get_cycle_id,
    trim_messages_for_model,
)
from reflexia.prompts import build_agent_system_prompt, build_reflexia_system_prompt


class AgentState(TypedDict):
    """State carried between graph steps."""

    messages: Annotated[list[AnyMessage], add_messages]
    react_step: int


def agent(
    state: AgentState,
    runtime: Runtime[ExecutionContext],
) -> dict[str, list[AnyMessage] | int]:
    """Agent node that calls the LLM and increments the ReAct step."""

    sys_msg = build_agent_system_prompt()

    next_step = state["react_step"] + 1
    execution_context = runtime.context
    llm_with_tools = execution_context.llm.bind_tools(execution_context.tools)
    llm_input_messages = trim_messages_for_model(
        [sys_msg] + state["messages"],
        execution_context,
    )

    execution_context.ltm.dump(execution_context.ltm_path)

    response = llm_with_tools.invoke(llm_input_messages)
    tagged_response = annotate_messages_with_cycle_id([response], cycle_id=next_step)
    return {
        "messages": tagged_response,
        "react_step": next_step,
    }


def limit_tool_calls(
    state: AgentState,
    runtime: Runtime[ExecutionContext],
) -> dict[str, object]:
    """Clamp tool calls on the latest AI message using execution-context limits."""

    if not state["messages"]:
        return {}

    max_parallel = max(1, int(runtime.context.max_parallel_tool_calls))
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return {}

    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls or len(tool_calls) <= max_parallel:
        return {}

    last_message.tool_calls = tool_calls[:max_parallel]
    return {}


def tag_tool_messages_with_cycle_id(
    state: AgentState,
    runtime: Runtime[ExecutionContext],
) -> dict[str, object]:
    """Annotate new tool messages from the latest step with the current cycle id."""

    cycle_id = int(state["react_step"])
    trailing_untagged: list[AnyMessage] = []

    for message in reversed(state["messages"]):
        if isinstance(message, SystemMessage):
            break
        if get_cycle_id(message) is not None:
            break
        trailing_untagged.append(message)

    if not trailing_untagged:
        return {}

    trailing_untagged.reverse()
    tagged = annotate_messages_with_cycle_id(trailing_untagged, cycle_id=cycle_id)
    for original, updated in zip(trailing_untagged, tagged, strict=False):
        original.additional_kwargs = dict(updated.additional_kwargs or {})
    return {}


def reflexia(state: AgentState, runtime: Runtime[ExecutionContext]):

    llm = runtime.context.llm
    ctx = runtime.context

    reflexia_prompt = build_reflexia_system_prompt()
    response = llm.invoke([reflexia_prompt] + state["messages"])

    if not isinstance(response.content, str):
        reflexia_text = str(response.content).strip()
    else:
        reflexia_text = response.content.strip()

    react_step = int(state.get("react_step", 0))
    trimmed_text = reflexia_text[: ctx.long_term_memory_max_chars]

    embedding = get_embedding(
        text=trimmed_text,
        ollama_embedder=ctx.ollama_embedder,
        embedder_keep_alive=ctx.embedder_keep_alive,
    )

    ctx.ltm.remember(
        react_step=react_step,
        text=trimmed_text,
        kind="reflexia",
        embedding=embedding,
    )

    reflexia_message = AIMessage(content=trimmed_text)
    tagged_reflexia = annotate_messages_with_cycle_id(
        [reflexia_message],
        cycle_id=react_step,
    )

    return {
        "messages": tagged_reflexia,
    }


def route_after_agent(
    state: AgentState,
    runtime: Runtime[ExecutionContext],
) -> str:
    """Route execution either to tools, the agent loop, or graph termination."""

    if state["react_step"] >= runtime.context.max_react_steps:
        return "__end__"

    return tools_condition(state).replace("__end__", "agent")


def build_graph() -> StateGraph:
    """Build and compile the Reflexia LangGraph workflow."""

    from reflexia.config import get_default_tools

    builder = StateGraph(
        AgentState,
        context_schema=ExecutionContext,
    )

    tool_node = ToolNode(get_default_tools())

    builder.add_node("agent", agent)
    builder.add_node("limit_tool_calls", limit_tool_calls)
    builder.add_node("tools", tool_node)
    builder.add_node("tag_tool_messages", tag_tool_messages_with_cycle_id)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "limit_tool_calls",
            "agent": "agent",
            "__end__": END,
        },
    )
    builder.add_edge("limit_tool_calls", "tools")
    builder.add_edge("tools", "tag_tool_messages")
    builder.add_edge("tag_tool_messages", "agent")
    return builder.compile()
