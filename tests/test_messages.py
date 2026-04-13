from __future__ import annotations

"""Heavy integration tests that run a local Ollama Qwen model.

These tests are intentionally expensive and require:
- Ollama daemon running locally
- model pulled: qwen2.5:14b-instruct
- env var RUN_HEAVY_OLLAMA_TESTS=1
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from reflexia.messages import count_tokens_qwen

if os.getenv("RUN_HEAVY_OLLAMA_TESTS") != "1":
    pytest.skip(
        "Set RUN_HEAVY_OLLAMA_TESTS=1 to run heavy Ollama integration tests.",
        allow_module_level=True,
    )

langchain_ollama = pytest.importorskip("langchain_ollama")
ChatOllama = langchain_ollama.ChatOllama

TOKENIZER_NAME = "Qwen/Qwen2.5-14B-Instruct"


def _make_llm() -> ChatOllama:
    return ChatOllama(
        model="qwen2.5:14b-instruct",
        temperature=0.0,
        reasoning=False,
        num_predict=50,
        num_ctx=32768,
    )


@pytest.mark.heavy
def test_token_counter_matches_native_input_tokens() -> None:
    llm = _make_llm()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi"),
        AIMessage(content="Hello. How can I help you?"),
        HumanMessage(content="Explain transformers in simple words."),
    ]

    local_tokens = count_tokens_qwen(messages, tokenizer_name=TOKENIZER_NAME)
    result = llm.invoke(messages)
    native_tokens = result.usage_metadata["input_tokens"]

    assert native_tokens == local_tokens


@pytest.mark.heavy
def test_stress_32k_context_counter_and_retention() -> None:
    llm = _make_llm()

    num_turns = 640
    target_turn = 37
    target_code = "CTXCODE_91357"

    messages = [
        SystemMessage(
            content="You must copy exact codes from prior assistant turns when asked."
        )
    ]

    for i in range(num_turns):
        messages.append(
            HumanMessage(
                content=(
                    f"[H{i}] Continue the synthetic dialogue. "
                    f"Acknowledge turn {i}."
                )
            )
        )
        if i == target_turn:
            assistant_text = (
                f"[A{i}] Persistent marker code: {target_code}. "
                "Remember this exact code."
            )
        else:
            assistant_text = (
                f"[A{i}] Synthetic assistant response for long-context "
                f"retention test on turn {i}."
            )
        messages.append(AIMessage(content=assistant_text))

    messages.append(
        HumanMessage(
            content=(
                f"What was the exact marker code from assistant turn {target_turn}? "
                "Return only the code token."
            )
        )
    )

    local_tokens = count_tokens_qwen(messages, tokenizer_name=TOKENIZER_NAME)
    result = llm.invoke(messages)

    native_tokens = result.usage_metadata["input_tokens"]
    predicted = str(result.content).strip()

    assert native_tokens == local_tokens
    assert predicted == target_code
