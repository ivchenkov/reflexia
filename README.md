# Reflexia

Reflexia is an experimental local-first agent built with LangGraph and Ollama. The current prototype models a simple autonomous loop with web access, persistent long-term memory, and a lightweight "self" that can accumulate biography, identity, realization, pleasant, painful, and journal-like memories.

The original project logic was developed in `poc.ipynb`. This repository now exposes the same core behavior as a regular Python package so it can be installed with `pip` and used outside the notebook.

## What is in the project

- LangGraph state machine with a single agent node and a tool node
- Local LLM runtime via `langchain-ollama`
- Long-term memory backed by embeddings from Ollama
- Web search through a locally deployed SearxNG instance
- Web page extraction through `trafilatura`

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
```

## Runtime requirements

The project expects two local services:

1. Ollama with:
   - a chat model, for example `qwen3.5:9b-q8_0`
   - an embedding model, for example `embeddinggemma`
2. SearxNG running locally and returning JSON responses

Pull the embedding model if needed:

```bash
ollama pull embeddinggemma
```

## SearxNG setup

The search server configuration is easy to miss. The original deployment note is preserved in `deployment.txt`; the important part is that SearxNG must explicitly enable JSON output.

Example `settings.yml`:

```yaml
use_default_settings: true

server:
  secret_key: "replace-this-with-your-own-secret"

search:
  formats:
    - html
    - json
```

Example launch command:

```bash
docker run -d \
  --name searxng \
  -p 8080:8080 \
  -e BASE_URL=http://localhost:8080/ \
  -v ~/searxng-config/settings.yml:/etc/searxng/settings.yml:ro \
  searxng/searxng
```

The default package configuration expects SearxNG at `http://localhost:8080/search`.

## Quick start

```python
from langchain_core.messages import HumanMessage

from reflexia import build_graph, create_default_execution_context

graph = build_graph()
context = create_default_execution_context()

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Do what you want! And please use your memory tools"
            )
        ],
        "react_step": 0,
    },
    context=context,
)
```

## Package layout

- `src/reflexia/config.py` - runtime context and default configuration
- `src/reflexia/embeddings.py` - Ollama embedding helper
- `src/reflexia/memory.py` - long-term memory model and persistence
- `src/reflexia/tools/` - web and memory tools
- `src/reflexia/graph.py` - LangGraph state, nodes, and graph builder

## Notes

- The repository restructuring intentionally does not change the core graph architecture.
- Memory dumps are written to the path configured in `ExecutionContext.ltm_path`.
- No command-line wrapper is included yet; the package currently exposes a Python API.
