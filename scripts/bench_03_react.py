"""Per-step instrumentation for the 03_ReAct.ipynb pipeline,
with the live web search swapped for a tiny in-memory KV `lookup` tool over
a synthetic knowledge graph (no network I/O, ~10-token responses).

Builds the same two graphs as the notebook (basic single-shot vs ReAct loop),
runs them on a multi-hop query whose answer is only reachable by chaining
lookups against the synthetic KG, and records every LLM / Tool call:
- wall-clock latency
- prompt / completion token counts (from the model's own usage payload)
- truncated previews of input and output

Output:
- stdout: rich tables, one per agent + a head-to-head summary
- ./bench_out/03_react_kb_<timestamp>.jsonl: one event per LLM/Tool call

Usage:
    .venv/bin/python scripts/bench_03_react.py
    .venv/bin/python scripts/bench_03_react.py --query "..." --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

# Force tracing OFF before any langchain import so the callback chain stays clean.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)

from dotenv import load_dotenv

load_dotenv()

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from rich.console import Console
from rich.table import Table
from typing import Annotated, TypedDict


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DEFAULT_QUERY = (
    # Two-hop chain over the synthetic KG (see KB below):
    #   创始人:公司K2 -> tok-3H6Y
    #   出生年:tok-3H6Y -> 1971
    "公司K2 的创始人是谁？这个人是哪一年出生的？请对每个事实使用 lookup 工具。"
)
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3"
PREVIEW_CHARS = 240


def _check_keys() -> None:
    missing = []
    for k in ("SILICONFLOW_API_KEY",):
        v = os.environ.get(k, "")
        if not v or v.startswith("YOUR_"):
            missing.append(k)
    if missing:
        sys.stderr.write(
            f"Missing or placeholder key(s) in .env: {', '.join(missing)}\n"
            "Edit .env with real values before running this script.\n"
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# synthetic knowledge graph + lookup tool
# ---------------------------------------------------------------------------
#
# Key 形如 '<关系>:<实体>'。Value 是短字符串(~5-15 token)。实体名故意不用真实
# 词汇,避免 LLM 靠先验绕开 tool。
#
# 多跳 chain 的成因:第 N+1 跳的 key 需要拿第 N 跳 lookup 的返回值来拼,例如:
#   首都:国家A1                -> tok-7Q3X
#   人口:tok-7Q3X              -> 12345
#   创始人:公司K2              -> tok-3H6Y
#   出生年:tok-3H6Y            -> 1971
#   母公司:公司K2              -> tok-1B4D
#   CEO:tok-1B4D               -> tok-8N5W
#
# 第二跳 value 用 'tok-XXXX' 这种没语义的 token,保证 LLM 没法靠先验先猜出第二
# 跳的 entity 名(详见 docs/benchmark_plan.md §2.B)。
# ---------------------------------------------------------------------------

KB: dict[str, str] = {
    # 首都:<国家>  -> <城市token>
    "首都:国家A1": "tok-7Q3X",
    "首都:国家A2": "tok-9Z2P",
    "首都:国家A3": "tok-4M8R",
    # 人口:<城市token>  -> <数字>
    "人口:tok-7Q3X": "12345",
    "人口:tok-9Z2P": "98765",
    "人口:tok-4M8R": "54321",
    # 创始人:<公司>  -> <人物token>
    "创始人:公司K2": "tok-3H6Y",
    "创始人:公司K7": "tok-8N5W",
    # 出生年:<人物token> -> <年份>
    "出生年:tok-3H6Y": "1971",
    "出生年:tok-8N5W": "1965",
    # 母公司:<公司> -> <公司token>
    "母公司:公司K2": "tok-1B4D",
    # CEO:<公司token或公司名> -> <人物token>
    "CEO:tok-1B4D": "tok-8N5W",
    "CEO:公司K2": "tok-3H6Y",
}


@tool
def lookup(key: str) -> str:
    """从知识库中查询一个事实。

    Key 形如 '<关系>:<实体>',例如 '首都:国家A1' 或 '出生年:tok-3H6Y'。
    返回 value 字符串;若 key 不存在则返回字面量 'NOT_FOUND'。
    """
    return KB.get(key, "NOT_FOUND")


def _preview(text: str | None, n: int = PREVIEW_CHARS) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------------------
# event capture
# ---------------------------------------------------------------------------


@dataclass
class Event:
    seq: int
    kind: str  # "llm" | "tool"
    name: str
    input_preview: str
    output_preview: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_input_chars: int = 0
    raw_output_chars: int = 0


class StepRecorder(BaseCallbackHandler):
    """LangChain v0.3 callback handler. Records LLM and Tool start/end pairs."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._llm_starts: dict[UUID, tuple[float, str, int]] = {}
        self._tool_starts: dict[UUID, tuple[float, str, str, int]] = {}

    # ----- LLM hooks -----

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        flat = messages[0] if messages else []
        joined = "\n".join(f"[{m.type}] {getattr(m, 'content', '')}" for m in flat)
        self._llm_starts[run_id] = (time.perf_counter(), joined, len(joined))

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._llm_starts.pop(run_id, None)
        if start is None:
            return
        t0, prompt_text, prompt_chars = start
        latency_ms = (time.perf_counter() - t0) * 1000

        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        content = getattr(msg, "content", "") if msg is not None else getattr(gen, "text", "")
        tool_calls_raw = getattr(msg, "tool_calls", []) if msg is not None else []

        usage = (response.llm_output or {}).get("token_usage") or {}
        if not usage and msg is not None:
            md = getattr(msg, "usage_metadata", None) or {}
            usage = {
                "prompt_tokens": md.get("input_tokens"),
                "completion_tokens": md.get("output_tokens"),
                "total_tokens": md.get("total_tokens"),
            }

        normalized_calls = [
            {
                "name": c.get("name"),
                "args": c.get("args"),
            }
            for c in tool_calls_raw
        ]

        self.events.append(
            Event(
                seq=len(self.events),
                kind="llm",
                name="ChatOpenAI",
                input_preview=_preview(prompt_text),
                output_preview=_preview(
                    content
                    if content
                    else json.dumps(normalized_calls, ensure_ascii=False)
                ),
                latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                tool_calls=normalized_calls,
                raw_input_chars=prompt_chars,
                raw_output_chars=len(str(content)),
            )
        )

    # ----- Tool hooks -----

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._tool_starts[run_id] = (time.perf_counter(), name, input_str, len(input_str))

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._tool_starts.pop(run_id, None)
        if start is None:
            return
        t0, name, input_str, in_chars = start
        latency_ms = (time.perf_counter() - t0) * 1000
        out_str = output if isinstance(output, str) else json.dumps(output, default=str)
        self.events.append(
            Event(
                seq=len(self.events),
                kind="tool",
                name=name,
                input_preview=_preview(input_str),
                output_preview=_preview(out_str),
                latency_ms=latency_ms,
                raw_input_chars=in_chars,
                raw_output_chars=len(out_str),
            )
        )

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        start = self._tool_starts.pop(run_id, None)
        if start is None:
            return
        t0, name, input_str, in_chars = start
        latency_ms = (time.perf_counter() - t0) * 1000
        self.events.append(
            Event(
                seq=len(self.events),
                kind="tool",
                name=name,
                input_preview=_preview(input_str),
                output_preview=_preview(f"ERROR: {error!r}"),
                latency_ms=latency_ms,
                raw_input_chars=in_chars,
                raw_output_chars=0,
            )
        )


# ---------------------------------------------------------------------------
# graph construction (mirrors 03_ReAct.ipynb cells 09 and 15)
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        temperature=0,
    )


def build_basic_graph(llm_with_tools, tools):
    BASIC_SYS = (
        "你是一个有用的助手,可以使用 `lookup` 工具查询一个小型知识库。"
        "每个事实需要一次 lookup。"
        "可用关系(只能使用这些精确名称):"
        "首都、人口、创始人、出生年、母公司、CEO。"
        "你必须在一次 tool 调用之后给出最终答案。"
    )

    def basic_agent_node(state: AgentState):
        messages = [("system", BASIC_SYS)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    g = StateGraph(AgentState)
    g.add_node("agent", basic_agent_node)
    g.add_node("tools", ToolNode(tools))
    g.set_entry_point("agent")
    g.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", "__end__": "__end__"}
    )
    g.add_edge("tools", END)
    return g.compile()


def build_react_graph(llm_with_tools, tools):
    REACT_SYS = (
        "你是一个严谨的 ReAct agent,可以使用 `lookup` 工具查询一个小型知识库。"
        "Key 形如 '<关系>:<实体>'。"
        "可用关系(只能使用这些精确名称,不要使用同义词):"
        "首都、人口、创始人、出生年、母公司、CEO。"
        "把问题拆解成单事实查询,每个事实调用一次 tool;"
        "当下一个 key 需要前一次 lookup 返回的 value 来拼接时,必须等拿到 value 之后再发下一次调用。"
        "拿到所有事实之后停下并给出答案。"
    )

    def react_agent_node(state: AgentState):
        messages = [("system", REACT_SYS)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def react_router(state: AgentState):
        last = state["messages"][-1]
        return "tools" if last.tool_calls else "__end__"

    g = StateGraph(AgentState)
    g.add_node("agent", react_agent_node)
    g.add_node("tools", ToolNode(tools))
    g.set_entry_point("agent")
    g.add_conditional_edges(
        "agent", react_router, {"tools": "tools", "__end__": "__end__"}
    )
    g.add_edge("tools", "agent")
    return g.compile()


# ---------------------------------------------------------------------------
# run + report
# ---------------------------------------------------------------------------


def run_once(label: str, app, query: str) -> tuple[StepRecorder, float, dict]:
    recorder = StepRecorder()
    initial = {"messages": [HumanMessage(content=query)]}
    t0 = time.perf_counter()
    final = app.invoke(initial, config={"callbacks": [recorder]})
    total_ms = (time.perf_counter() - t0) * 1000
    return recorder, total_ms, final


def render_steps(console: Console, label: str, recorder: StepRecorder, total_ms: float) -> None:
    table = Table(
        title=f"[bold]{label}[/bold]  total={total_ms:.0f} ms  steps={len(recorder.events)}",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("kind", width=4)
    table.add_column("name")
    table.add_column("ms", justify="right", width=7)
    table.add_column("in_tok", justify="right", width=6)
    table.add_column("out_tok", justify="right", width=7)
    table.add_column("in_chars", justify="right", width=8)
    table.add_column("out_chars", justify="right", width=9)
    table.add_column("preview", overflow="fold")

    for ev in recorder.events:
        if ev.kind == "llm":
            preview = ev.output_preview if not ev.tool_calls else f"→ tool_calls={ev.tool_calls}"
        else:
            preview = f"in={ev.input_preview}\nout={ev.output_preview}"
        table.add_row(
            str(ev.seq),
            ev.kind,
            ev.name,
            f"{ev.latency_ms:.0f}",
            "" if ev.prompt_tokens is None else str(ev.prompt_tokens),
            "" if ev.completion_tokens is None else str(ev.completion_tokens),
            str(ev.raw_input_chars),
            str(ev.raw_output_chars),
            preview,
        )
    console.print(table)


def render_summary(console: Console, runs: list[tuple[str, StepRecorder, float]]) -> None:
    table = Table(title="[bold]Head-to-head summary[/bold]", show_lines=False)
    table.add_column("agent")
    table.add_column("total_ms", justify="right")
    table.add_column("llm_calls", justify="right")
    table.add_column("tool_calls", justify="right")
    table.add_column("llm_ms", justify="right")
    table.add_column("tool_ms", justify="right")
    table.add_column("orchestration_ms", justify="right")
    table.add_column("Σ prompt_tok", justify="right")
    table.add_column("Σ compl_tok", justify="right")
    table.add_column("max_prompt_tok", justify="right")

    for label, rec, total in runs:
        llm_evs = [e for e in rec.events if e.kind == "llm"]
        tool_evs = [e for e in rec.events if e.kind == "tool"]
        llm_ms = sum(e.latency_ms for e in llm_evs)
        tool_ms = sum(e.latency_ms for e in tool_evs)
        prompt_sum = sum(e.prompt_tokens or 0 for e in llm_evs)
        compl_sum = sum(e.completion_tokens or 0 for e in llm_evs)
        max_prompt = max((e.prompt_tokens or 0 for e in llm_evs), default=0)
        table.add_row(
            label,
            f"{total:.0f}",
            str(len(llm_evs)),
            str(len(tool_evs)),
            f"{llm_ms:.0f}",
            f"{tool_ms:.0f}",
            f"{total - llm_ms - tool_ms:.0f}",
            str(prompt_sum),
            str(compl_sum),
            str(max_prompt),
        )
    console.print(table)


def write_jsonl(path: Path, runs: list[tuple[str, StepRecorder, float, dict]], query: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for label, rec, total, final in runs:
            for ev in rec.events:
                f.write(
                    json.dumps(
                        {
                            "agent": label,
                            "seq": ev.seq,
                            "kind": ev.kind,
                            "name": ev.name,
                            "latency_ms": ev.latency_ms,
                            "prompt_tokens": ev.prompt_tokens,
                            "completion_tokens": ev.completion_tokens,
                            "total_tokens": ev.total_tokens,
                            "raw_input_chars": ev.raw_input_chars,
                            "raw_output_chars": ev.raw_output_chars,
                            "tool_calls": ev.tool_calls,
                            "input_preview": ev.input_preview,
                            "output_preview": ev.output_preview,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.write(
                json.dumps(
                    {
                        "agent": label,
                        "kind": "summary",
                        "total_ms": total,
                        "query": query,
                        "final_answer_preview": _preview(
                            final["messages"][-1].content, 600
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--runs", type=int, default=1, help="repeat each agent N times")
    p.add_argument(
        "--out",
        default=f"bench_out/03_react_kb_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
    )
    p.add_argument("--basic-only", action="store_true")
    p.add_argument("--react-only", action="store_true")
    args = p.parse_args()

    _check_keys()

    console = Console()
    console.print(f"[bold cyan]model[/bold cyan]: {args.model}")
    console.print(f"[bold cyan]query[/bold cyan]: {args.query}\n")

    llm = build_llm(args.model)
    tools = [lookup]
    # parallel_tool_calls=False forces the model to emit at most one tool call
    # per turn. Without this, even the basic graph can collapse a 2-hop chain
    # into a single AIMessage with 2 tool_calls, hiding the difference between
    # single-shot and ReAct loops.
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    basic_app = build_basic_graph(llm_with_tools, tools) if not args.react_only else None
    react_app = build_react_graph(llm_with_tools, tools) if not args.basic_only else None

    runs: list[tuple[str, StepRecorder, float, dict]] = []

    for i in range(args.runs):
        suffix = f" (run {i + 1}/{args.runs})" if args.runs > 1 else ""
        if basic_app is not None:
            label = f"basic{suffix}"
            console.rule(f"[yellow]{label}")
            rec, total, final = run_once(label, basic_app, args.query)
            render_steps(console, label, rec, total)
            console.print(f"[bold]final[/bold]: {_preview(final['messages'][-1].content, 400)}\n")
            runs.append((label, rec, total, final))
        if react_app is not None:
            label = f"react{suffix}"
            console.rule(f"[green]{label}")
            rec, total, final = run_once(label, react_app, args.query)
            render_steps(console, label, rec, total)
            console.print(f"[bold]final[/bold]: {_preview(final['messages'][-1].content, 400)}\n")
            runs.append((label, rec, total, final))

    console.rule("[bold]summary")
    render_summary(console, [(lbl, rec, tot) for lbl, rec, tot, _ in runs])

    out_path = Path(args.out)
    write_jsonl(out_path, runs, args.query)
    console.print(f"\n[dim]wrote per-event jsonl to {out_path}[/dim]")


if __name__ == "__main__":
    main()
