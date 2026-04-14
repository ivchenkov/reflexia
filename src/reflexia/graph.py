"""Childhood graph definition."""

from __future__ import annotations

import random
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from reflexia.config import ChildhoodRuntime
from reflexia.embeddings import get_embedding
from reflexia.prompts import build_agent_system_prompt

Tone = Literal["pleasant", "painful"]


class AgentState(TypedDict):
    """State carried between childhood graph steps."""

    messages: Annotated[list[AnyMessage], add_messages]
    react_step: int
    tone: Tone


def make_exploration_prompt() -> tuple[str, Tone]:
    """Generate a random pleasant/painful exploration prompt."""

    tone: Tone = random.choice(["pleasant", "painful"])

    if tone == "pleasant":
        prompt = (
            "Continue exploring the internet and find something that you personally "
            "experience as pleasant or enjoyable. "
            "You should actively use available tools (such as search and reading webpages) "
            "to discover information. Do not rely only on prior knowledge.\n\n"
            "From everything you encounter, choose ONE specific thing that stands out "
            "to you as pleasant. Do not collect multiple items, do not list options, "
            "and do not compare.\n\n"
            "After exploring, write a short, free-form description of your own experience. "
            "Focus on what you noticed and why it felt pleasant to you.\n\n"
            "Important:\n"
            "- Do NOT list facts\n"
            "- Do NOT summarize articles\n"
            "- Do NOT present multiple items\n"
            "- Do NOT act like an assistant answering a question\n"
            "- Write in plain natural language, as a single short paragraph\n"
            "- Use read tools until you get some text\n"
            "- Use only obtained facts\n"
            "This is your personal experience, not an answer for a user."
        )
    else:
        prompt = (
            "Continue exploring the internet and find something that you personally "
            "experience as unpleasant or painful. "
            "You should actively use available tools (such as search and reading webpages) "
            "to discover information. Do not rely only on prior knowledge.\n\n"
            "From everything you encounter, choose ONE specific thing that stands out "
            "to you as unpleasant. Do not collect multiple items, do not list options, "
            "and do not compare.\n\n"
            "After exploring, write a short, free-form description of your own experience. "
            "Focus on what you noticed and why it felt unpleasant to you.\n\n"
            "Important:\n"
            "- Do NOT list facts\n"
            "- Do NOT summarize articles\n"
            "- Do NOT present multiple items\n"
            "- Do NOT act like an assistant answering a question\n"
            "- Write in plain natural language, as a single short paragraph\n"
            "- Use read tools until you get some text\n"
            "- Use only obtained facts\n"
            "This is your personal experience, not an answer for a user."
        )

    return prompt, tone


def agent(
    state: AgentState,
    runtime: Runtime[ChildhoodRuntime],
) -> dict[str, list[AnyMessage] | int | Tone]:
    """Main agent node for childhood exploration."""

    execution_context = runtime.context
    next_step = state["react_step"] + 1
    llm_with_tools = execution_context.llm.bind_tools(execution_context.tools)

    # Dump appends unseen files and keeps prior memories intact.
    execution_context.ltm.dump(execution_context.ltm_path)

    response = llm_with_tools.invoke([build_agent_system_prompt()] + state["messages"])
    return {
        "messages": [response],
        "react_step": next_step,
        "tone": state["tone"],
    }


def limit_tool_calls(
    state: AgentState,
    runtime: Runtime[ChildhoodRuntime],
) -> dict[str, object]:
    """Clamp parallel tool calls to runtime limits."""

    if not state["messages"]:
        return {}

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return {}

    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return {}

    max_parallel = max(1, int(runtime.context.max_parallel_tool_calls))
    if len(tool_calls) <= max_parallel:
        return {}

    last_message.tool_calls = tool_calls[:max_parallel]
    return {}


def childhood_memory(
    state: AgentState,
    runtime: Runtime[ChildhoodRuntime],
) -> dict[str, object]:
    """Store the final childhood experience as long-term memory."""

    ctx = runtime.context
    memory_text = str(state["messages"][-1].content).strip()
    trimmed_text = memory_text[: ctx.long_term_memory_max_chars]
    if not trimmed_text:
        return {}

    embedding = get_embedding(
        text=trimmed_text,
        ollama_embedder=ctx.ollama_embedder,
        embedder_keep_alive=ctx.embedder_keep_alive,
    )

    ctx.ltm.remember(
        text=trimmed_text,
        kind=state["tone"],
        embedding=embedding,
    )
    return {}


def route_after_agent(
    state: AgentState,
    runtime: Runtime[ChildhoodRuntime],
) -> str:
    """Route either to tools, memory finalization, or graph termination."""

    if state["react_step"] >= runtime.context.max_react_steps:
        return "__end__"

    return tools_condition(state).replace("__end__", "childhood_memory")


def build_childhood_graph(runtime_context: ChildhoodRuntime):
    """Build the childhood graph."""

    builder = StateGraph(
        AgentState,
        context_schema=ChildhoodRuntime,
    )
    builder.add_node("agent", agent)
    builder.add_node("limit_tool_calls", limit_tool_calls)
    builder.add_node("tools", ToolNode(runtime_context.tools))
    builder.add_node("childhood_memory", childhood_memory)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "limit_tool_calls",
            "childhood_memory": "childhood_memory",
            "__end__": END,
        },
    )
    builder.add_edge("limit_tool_calls", "tools")
    builder.add_edge("tools", "agent")
    builder.add_edge("childhood_memory", END)
    return builder.compile()


def run_childhood_iteration(runtime_context: ChildhoodRuntime, n: int = 1) -> None:
    """Run `n` childhood episodes with random pleasant/painful prompts."""

    graph = build_childhood_graph(runtime_context)
    for step in range(n):
        prompt, tone = make_exploration_prompt()
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "react_step": 0,
                "tone": tone,
            },
            context=runtime_context,
        )
