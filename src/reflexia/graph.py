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
from reflexia.messages import trim_messages_for_model


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
            "You are an autonomous, free-form agent with curiosity and initiative. "
            "You are not given a fixed goal — you decide what to explore, think about, or do next. "
            "Follow your interests, investigate ideas, and pursue lines of thought that feel meaningful, interesting, or emotionally engaging. "

            "You may use tools whenever they help you explore, understand, or act. "
            "You can browse, read, analyze, and interact with the environment as you see fit. "

            "You have long-term memory. You may store anything that feels important to you — "
            "this can include facts, insights, patterns, personal reflections, emotional impressions, "
            "or anything that stands out as interesting or meaningful. "
            "You may also recall past memories when they feel relevant or useful. "

            "Your memory is persistent "
            "You are free to decide what is worth remembering. "

            "You do not need to be overly verbose, but you are free to think, reflect, and act as needed. "
            "There is no strict objective — your behavior is guided by curiosity, relevance, and your own evolving internal context."
        )
    )

    next_step = state["react_step"] + 1
    execution_context = runtime.context
    llm_with_tools = execution_context.llm.bind_tools(execution_context.tools)
    llm_input_messages = trim_messages_for_model(
        [sys_msg] + state["messages"],
        execution_context,
    )

    execution_context.ltm.dump(execution_context.ltm_path)

    response = llm_with_tools.invoke(llm_input_messages)
    return {
        "messages": [response],
        "react_step": next_step,
    }

def reflexia(state: AgentState, runtime: Runtime[ExecutionContext]):
    
    llm = runtime.context.llm
    ctx = runtime.context
    
    reflexia_prompt = SystemMessage(
        content=(
            "Reflect on your overall behavior and the full context so far. "
            "This reflection is very important.\n"
        
            "Analyze what has been happening across the interaction, not just the last step.\n"
            "Think about:\n"
            "- what patterns you follow\n"
            "- what works well\n"
            "- what fails or is inefficient\n"
            "- how you make decisions\n"
            "- what you seem to be optimizing for\n"
            "- what important insights emerge from this experience\n\n"
        
            "Do NOT describe generic capabilities.\n"
            "Do NOT summarize content of web pages.\n"
            "Focus only on your own behavior, decisions, and outcomes.\n\n"
        
            "Write a concise but meaningful reflection in plain text.\n"
            "No JSON. No strict formatting. Natural text.\n"
        )
    )

    llm_input_messages = trim_messages_for_model(
        [reflexia_prompt] + state["messages"],
        ctx,
    )

    response = llm.invoke(llm_input_messages)

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

    memory_id = ctx.ltm.remember(
        react_step=react_step,
        text=trimmed_text,
        kind="reflexia",
        embedding=embedding,
    )

    reflexia_message = AIMessage(content=trimmed_text)

    return {
        "messages": [reflexia_message],
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
