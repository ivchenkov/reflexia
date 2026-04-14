# Reflexia (Childhood Prototype)

This repository now starts from the **childhood loop** prototype.

The model runs short autonomous exploration episodes, experiences something
pleasant or painful, and stores that experience into long-term memory.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
```

## Configure environment

Copy and edit env settings:

```bash
cp .env.example .env
```

Most important values:

- `REFLEXIA_MODEL` (default `qwen3.5:9b`)
- `REFLEXIA_MEMORY_PATH` (default `./memory_test/childhood`)
- `REFLEXIA_SEARXNG_URL` (default `http://localhost:8080/search`)

## Runtime requirements

You need:

1. Ollama running locally
2. A pulled chat model (`REFLEXIA_MODEL`)
3. A pulled embedder model (default `embeddinggemma`)
4. SearxNG with JSON enabled

```bash
ollama pull embeddinggemma
```

## Quick start (Python / notebook)

```python
from reflexia import create_childhood_runtime, run_childhood_iteration

runtime = create_childhood_runtime()
run_childhood_iteration(runtime, n=5)
```

## Build graph and invoke manually

```python
from langchain_core.messages import HumanMessage

from reflexia import build_childhood_graph, create_childhood_runtime, make_exploration_prompt

runtime = create_childhood_runtime()
graph = build_childhood_graph(runtime)

prompt, tone = make_exploration_prompt()
result = graph.invoke(
    {
        "messages": [HumanMessage(content=prompt)],
        "react_step": 0,
        "tone": tone,
    },
    context=runtime,
)
```

## Persistent memory behavior

Memory is append-only and file-based.

- New memories get unique IDs (`mem_...`) and are stored as new files.
- Previous memories are not overwritten.
- Files are written under `REFLEXIA_MEMORY_PATH` in:
  - `items/`
  - `vectors/`

This is intentional so long runs can accumulate many memory files.

## Current package layout

- `src/reflexia/config.py` - childhood runtime + `.env` loading
- `src/reflexia/graph.py` - childhood graph and run helpers
- `src/reflexia/prompts.py` - childhood system prompt
- `src/reflexia/memory.py` - append-only persistent long-term memory
- `src/reflexia/tools/web.py` - web tools
- `childhood.ipynb` - original notebook prototype used as logic source
