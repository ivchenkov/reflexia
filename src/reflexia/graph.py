"""LangGraph graph definition for Reflexia."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from reflexia.config import ExecutionContext


class AgentState(TypedDict):
    """State carried between graph steps."""

    messages: Annotated[list[AnyMessage], add_messages]
    react_step: int


def agent(
    state: AgentState,
    runtime: Runtime[ExecutionContext],
) -> dict[str, list[AnyMessage] | int]:
    """Agent node that calls the LLM and increments the ReAct step."""

    sys_msg = SystemMessage(
        content=(
            "You are a free-form autonomous research agent. "
            "You may browse the web, read pages, store long-term memories, and "
            "recall long-term memories. "
            "You are free to explore and use the internet when useful. "
            "Use tools whenever they help. "
            "You may store important information into long-term memory. "
            "Long-term memory is persistent and may be truncated by the system, "
            "so store only the most important information. "
            "You may recall long-term memories when they are relevant. "
            "Be concise when possible, but do what is needed to complete the task."
        )
    )

    next_step = state["react_step"] + 1
    execution_context = runtime.context
    llm_with_tools = execution_context.llm.bind_tools(execution_context.tools)

    execution_context.ltm.dump(execution_context.ltm_path)

    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {
        "messages": [response],
        "react_step": next_step,
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

    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(get_default_tools()))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "agent": "agent",
            "__end__": END,
        },
    )
    builder.add_edge("tools", "agent")
    return builder.compile()
