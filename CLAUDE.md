# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

This is an educational collection of **17 self-contained Jupyter notebooks**, one per agentic architecture (`01_reflection.ipynb` … `17_reflexive_metacognitive.ipynb`). There is no application code, no build, and no test suite — each notebook is the deliverable. `requirements.txt` lists only the *baseline* libraries; most notebooks `!pip install` additional packages (e.g. `neo4j`, `faiss-cpu`, `tiktoken`) inline in their first cell.

The project is meant to be read top-to-bottom: notebook N often assumes concepts introduced in notebooks 1…N-1.

## Environment

- Python venv lives at `.venv/` and is already provisioned. Use `.venv/bin/python` directly rather than activating.
- `jupyter` itself is not installed in the venv — notebooks are typically opened from an external Jupyter/VS Code/Cursor process that points at this venv as its kernel.
- Secrets go in `.env` (git-ignored). See `.env.example` for the expected keys. `python-dotenv` is loaded inside each notebook.

## LLM provider — important

Most notebooks use **Nebius** via `langchain_nebius.ChatNebius` (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`, `mistralai/Mixtral-8x22B-Instruct-v0.1`) and require `NEBIUS_API_KEY`.

`03_ReAct.ipynb` has been migrated in this fork to **SiliconFlow** via `langchain_openai.ChatOpenAI` pointed at `https://api.siliconflow.cn/v1`, requiring `SILICONFLOW_API_KEY`. The migration template is documented in `docs/03_ReAct_analysis.md` §1 — `bind_tools` / `with_structured_output` / LangGraph orchestration carry over unchanged because SiliconFlow exposes the OpenAI-compatible protocol. Apply the same template if asked to migrate other notebooks.

Other per-notebook external dependencies:

- `02`, `03`, `05`, `11` (and others using web search): `TAVILY_API_KEY`
- `08_episodic_with_semantic`, `12_graph`: a running Neo4j instance plus `NEO4J_URI`/`NEO4J_USERNAME`/`NEO4J_PASSWORD`
- `08_episodic_with_semantic`: also FAISS (installed inline) and `NebiusEmbeddings`

## Architectural pattern shared by every notebook

Each notebook builds an agent as a `langgraph.StateGraph`:

1. State is typically a `TypedDict` whose central field is a `messages: Annotated[list, add_messages]` reducer.
2. Nodes are LLM calls or tool invocations (`ToolNode`).
3. Edges (and especially **conditional** edges) encode the architecture's distinguishing idea — the difference between "Tool Use" and "ReAct" is literally one edge (`tools → END` vs `tools → agent`); see `docs/03_ReAct_analysis.md` §3 for a worked example.
4. Most notebooks finish with an **LLM-as-a-Judge** evaluation cell that scores the agent's output against a rubric — preserve this pattern when adding/modifying notebooks.

When asked to explain or modify behavior, locate the conditional-edge router function first; it almost always encodes the architecture's core decision logic.

## Working with the notebooks

- Read/modify cells using JSON tooling on the `.ipynb` files (e.g. `json.load` in `.venv/bin/python`); there is no helper script.
- Keep the existing `!pip install -q -U …` cell at the top of each notebook intact — it is how end users provision missing packages without touching `requirements.txt`.
- LangSmith tracing is enabled by default via `LANGCHAIN_TRACING_V2=true` and a per-notebook `LANGCHAIN_PROJECT` name; leave these in unless explicitly asked to remove them.

## `docs/` analysis notes

`docs/NN_<name>_analysis.md` files are hand-written deep-dives in **Chinese** that explain what each notebook is really doing (architecture diagrams, message-state walkthroughs, real-world analogues). When adding a new analysis doc:

- Match the existing files' structure (numbered sections, ASCII state diagrams, comparison tables).
- Write in Chinese to stay consistent with the existing notes.
- Focus on architectural insight, not API reference — the notebooks themselves already explain the API surface.
