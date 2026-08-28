# Enterprise GraphRAG

A self-hosted, provider-agnostic knowledge graph pipeline that extracts entities and relationships from documents, validates every claim against source text before writing it to the graph, and answers questions with cited, grounded responses.

This is a portfolio project built to explore self-grounding RAG architecture end-to-end: document ingestion, LLM-based extraction, graph indexing, multi-provider LLM failover, and a chat interface with verifiable citations. It's a working local prototype — see [Scope & Limitations](#scope--limitations) below for what's in and out.

## What Makes This Different

Most RAG systems blindly trust LLM output. This one checks its work: every extracted entity and relationship is validated against the source document before being written to the graph. If a claim can't be verified word-for-word against the source, it's skipped — not silently, but with a friendly explanation shown to the user.

## Key Features

- 📄 Document ingestion with automatic semantic chunking
- 🕸️ Real-time knowledge graph construction (Neo4j)
- ✅ Self-grounding extraction — every claim verified against source text before writing
- 🔄 Provider-agnostic LLM layer with automatic failover (Anthropic, Groq, Cerebras, OpenAI-compatible)
- 💬 Chat interface with cited, grounded answers and expandable source references
- 🔍 Interactive graph visualization with hover tooltips
- 🗑️ One-click graph reset for testing multiple documents
- 🐳 One-click local deployment via Docker Compose

## Scope & Limitations

This is a working local prototype, not a production deployment:
- No authentication layer
- Tested single-user, not under concurrent load
- No staging/cloud deployment configured

These are natural next steps if this project were taken further.

