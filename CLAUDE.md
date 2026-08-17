# CLAUDE.md

## Project

Enterprise GraphRAG & Multi-Agent Swarm Engine — a modular GraphRAG engine combining FastAPI, LangChain, and knowledge graphs for autonomous multi-agent reasoning.

**Current state: scaffolding only.** Directory structure, dependencies, and the architectural contract exist; there is no application code, no tests, and no entrypoint yet. When asked to add a feature, expect to also establish the structure it lives in.

Read **`CONTEXT.md`** first — it defines the four pipeline stages, what each one owns, and the payload contracts between them. Code belongs in the stage that owns that responsibility:

```
stages/extraction/        parse · chunk · LLM entity/relation extraction (no DB access)
stages/graph_indexing/    entity resolution · idempotent Neo4j upserts · vector index (only writer)
stages/reasoning_agent/   query router · graph+vector retrieval · LangGraph agent swarm (read-only)
stages/fastapi_service/   HTTP surface · async jobs · streaming (no reasoning, no Cypher)
_config/                     graph schema · model selection · prompts · retrieval params
```

Stage directories are importable Python packages (`stages.extraction`, `stages.graph_indexing`, …). They were originally numbered (`01_extraction`) and renamed once the first real module landed, since a leading digit is not a legal identifier — keep them that way, and put ordering in docs rather than in directory names.

## Environment

- Python **3.11.15**, in the in-repo `venv/` (created with `uv`-managed CPython; `pyvenv.cfg` records its parent as an unrelated `hermes-agent` venv — harmless, but don't treat that path as a dependency of this project).
- Activate: `venv\Scripts\Activate.ps1` (PowerShell) or `source venv/Scripts/activate` (Bash). Windows layout — executables live in `venv/Scripts/`, not `bin/`.
- Install deps: `pip install -r requirements.txt`
- Secrets come from `.env` (gitignored) via `python-dotenv`; `.env.example` lists the expected keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `LANGCHAIN_API_KEY`, and the `NEO4J_*` connection settings.
- Neo4j runs via `docker-compose.yml` (`docker compose up -d`): Neo4j 5 community with APOC, Bolt on 7687, Browser on 7474, default creds `neo4j` / `graphrag_dev_password`. Note the **Docker CLI is not currently on PATH in this environment**, so the compose file has been YAML-validated but never actually started — `docker compose config` could not be run.

## Installed stack

`requirements.txt` is unpinned and lists only the top level (`langchain`, `langchain-community`, `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`). What is actually resolved in `venv/` is more current and matters for API choices:

- **LangChain 1.3.x** with `langchain-core` 1.5.x, `langgraph` 1.2.x, `langgraph-checkpoint`, `langgraph-prebuilt`, `langsmith`. This is the v1 API — chains/agents differ substantially from pre-1.0 LangChain, and `langchain_classic` is present for legacy imports. Verify against the installed version rather than recalling older LangChain idioms. LangGraph is the natural home for the "multi-agent swarm" side of the project.
- **FastAPI 0.141** / Starlette 1.6 / **uvicorn 0.52**, **Pydantic v2** (2.13) plus `pydantic-settings`.
- **Providers**: `langchain-anthropic` 1.5.6 (`anthropic` 0.122) and `langchain-openai` 1.5.1 (`openai` 3.1, `tiktoken`). When adding a Claude model call, consult the `claude-api` skill for current model IDs instead of guessing.
- **Graph store**: `neo4j` Python driver **6.2.0** against a Neo4j **5** server. Major versions intentionally differ; if a driver call behaves unexpectedly, check the 6.x API rather than assuming 5.x driver semantics.
- `docker` SDK 7.2 is installed for programmatic container control (bringing Neo4j up in tests/fixtures).
- SQLAlchemy 2.0, numpy 2.4, httpx, websockets are available transitively.

## Conventions

Nothing is established yet, so prefer the defaults the stack implies: Pydantic v2 models for API and config schemas, `pydantic-settings` for env-backed config, async FastAPI handlers, and a `uvicorn` entrypoint. Keep new dependencies added to `requirements.txt` in the same unpinned style unless asked to pin.

## Model Switching & Cost Optimization

This project uses cost-optimized model switching to balance capability and expense across different task phases.

**Default Model: Haiku**
- Use Haiku for all standard work: file creation, routine editing, bug fixes, tests, Git commits, and documentation.
- Haiku is fast and cost-effective for structural tasks and mechanical code generation.

**Opus Escalation: When to Request**
- Run `/model opus` **only** when starting:
  - Complex multi-file architectural planning and design decisions.
  - Heavy Cypher query optimization or graph traversal algorithms.
  - Major structural redesigns affecting multiple stages or payloads.
- Opus brings stronger reasoning for algorithmic depth and cross-system tradeoffs.

**Transition Back: After Planning**
- As soon as an architectural plan is finalized and ready for implementation, run `/model haiku` to return to cost-effective mode.
- Write all implementation files (schemas, modules, tests) under Haiku; it excels at code generation once the design is locked.

This discipline keeps token cost linear in project scope rather than quadratic in iteration.

## Git

Default branch is `master`. Never commit `.env`.
