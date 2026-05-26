# LangGraph 编排层 Benchmark 设计

本文档记录把仓库里 5 个 notebook（`02_tool_use` / `03_ReAct` / `04_planning` / `11_meta_controller` / `15_RLHF`）改造成可被 emon 分析的性能 benchmark case 的整体设计。**这一稿只锁设计、不动代码**，下一步才开始按本文档落地。

## 1. 背景与目标

随着 agent 应用越来越多，可以预见**编排层**（orchestrator）会从应用代码里独立出来，成为一个单独部署、对接多种 LLM 后端和工具集群的中间层。这一层的性能画像目前几乎没人系统测过。

本 benchmark 的目标很窄：

- **要测的**：LangGraph 自身的开销 —— `StateGraph` 调度、节点函数包装、`add_messages` reducer、`add_conditional_edges` 路由、`ToolNode` 包装、`with_structured_output` / `bind_tools` 在 invoke 路径上引入的 wrapper。
- **不测的**：LLM 推理本身（业务上 LLM 一般是另一个进程/服务）、Tavily 这类外部工具的网络往返（同样是外部依赖）。

所以最终方案是：**LLM 用本地真服务（毫秒级，OpenAI 兼容协议）**，**Tavily 替换成零 I/O 的 FakeTool**，整个编排层代码从 notebook 原样搬过来，只换这两处依赖。

## 2. 已确定的设计决策

下列条目已经和场景设计者对齐，不在落地阶段重新讨论：

1. **LLM**：本地 OpenAI 兼容服务。`.env` 由用户自己维护，benchmark 只通过环境变量 `BENCH_LLM_BASE_URL` / `BENCH_LLM_MODEL` / `BENCH_LLM_API_KEY` 读取，**不在代码里写死任何端点**。
2. **Tool**：`FakeTool` 替换 `TavilySearch` / `TavilySearchResults`，返回固定 dict/字符串，零网络 I/O。
3. **方差是信号**：不丢弃任何一轮、不写"轨迹守卫"，每轮迭代记录 `(hop_count, msg_count, tool_calls, latency_ns, ...)`，最终输出**分布**而不是均值。生产里的编排层就是要面对变长上下文和分支次数浮动，这部分方差恰恰是要测的。
4. **两种 workload**：
   - **A — micro**：一个简单 query 反复跑，给 emon 一段稳定 hot path。
   - **B — macro**：完整 query pool 顺序循环，让轨迹形态自然变化，测的是编排层在真实输入分布下的画像。
5. **每个 case 配一个 baseline**：用裸 Python 复刻同样的控制流（不引 `StateGraph`、不引 `ToolNode`、不引 `add_messages`）。`graph_latency − baseline_latency` 才是编排层的纯开销。
6. **Query pool**：每个 case 由 AI 按语义起草 20–50 条，写到 `query_pool.py` 里供用户审阅修改。
7. **LangSmith tracing 强制关闭**：所有 notebook 默认开 `LANGCHAIN_TRACING_V2=true`，会在每个节点回调里发 HTTP 到 LangSmith，对 benchmark 是巨大污染。`benchmarks/__init__.py` 在任何 LangChain 模块 import 之前把它强制关掉。

## 3. 目录布局

```
benchmarks/
  __init__.py                     # tracing kill-switch（必须最先 import）
  fakes.py                        # FakeTool；不 fake LLM
  llm.py                          # build_llm() 从 BENCH_LLM_* 环境变量构造 ChatOpenAI
  query_pool.py                   # POOLS: dict[case_name, list[str]]
  cases/
    __init__.py
    case_02_tool_use.py
    case_02_tool_use_baseline.py
    case_03_react.py
    case_03_react_baseline.py
    case_04_planning.py
    case_04_planning_baseline.py
    case_11_meta.py
    case_11_meta_baseline.py
    case_15_rlhf.py
    case_15_rlhf_baseline.py
  run.py                          # CLI 入口
  analyze.py                      # 后处理：分布、轨迹直方图、配对差分
  README.md                       # 环境变量契约、marker 格式、运行示例
```

每个 case 模块对外暴露统一 API：

```python
NAME: str                                    # 例如 "03_react"
def build() -> Runnable | Callable           # compile 完的 graph，或 baseline 的入口函数
def make_input(query: str) -> dict           # 构造该 case 的初始 state
def parse_trace(final_state) -> dict         # 从 final state 抽取 hop_count / msg_count 等
```

`run.py` 是唯一的驱动器，case 文件只负责"把图建出来、把状态读出来"。

## 4. 各 case 改造规则

每个 case 文件就是把 notebook 里的 LangGraph 代码搬过来，**只做三处替换**，其它一律保留（包括 pydantic schema、prompt 字符串、节点函数体、conditional edge 写法）：

| 替换对象 | 原版 | 新版 |
|---|---|---|
| LLM | `ChatNebius(model=...)` 或 notebook 里的 `ChatOpenAI(...)` | `build_llm()` 来自 `benchmarks/llm.py` |
| Tool | `TavilySearch(...)` / `TavilySearchResults(...)` | `FakeTool` 来自 `benchmarks/fakes.py` |
| 驱动方式 | `.stream(initial_input, stream_mode="values")` | `.invoke(initial_input)` |

把 stream 改成 invoke 的原因：stream 会触发额外的 callback / yield 路径，这部分开销和编排本身无关，且让计时窗口变模糊；invoke 路径更干净，emon 看到的也更接近"一次完整图执行"的开销。

`benchmarks/__init__.py` 顶部强制关掉 tracing：

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)
```

这必须在任何 `langchain` / `langgraph` 模块 import 之前执行，否则 callback manager 已经注册就没用了。`run.py` 启动时会再 assert 一次，并把"tracing 已关"打到 stderr。

### case 04 的 planner 正则脆弱性

`04_planning` notebook 的 `executor_node` 里有一段正则：

```python
re.search(r"(\w+)\((?:\"|\')(.*?)(?:\"|\')\)", next_step)
```

用来从 planner 给的 plan 字符串（形如 `web_search('population of Paris')`）里抠出 tool 名和参数。这是 LLM 输出契约最脆弱的一处 —— 一旦 planner 生成 `web_search("foo bar")` 之外的格式，正则就会 fallback 到 `tool_name = "web_search"`，把整段 step 当 query 直接传过去。

**这段不要修复**，它就是真实编排层会面对的脏边界。但 `parse_trace` 要把 fallback 次数计入 trace：

```python
{"regex_fallbacks": int}
```

后续分析里如果发现 `regex_fallbacks > 0` 的轮次延迟系统性偏低/偏高，就能直接定位到这个脆弱点。

## 5. Baseline 等价规则

Baseline 是**控制流等价**，不是**实现等价**。它必须满足：

- 用同一个 `build_llm()` 实例
- 用同一个 `FakeTool` 实例（同样的入参、同样的返回）
- 消息累积用最朴素的 `list.append` / `list.extend`，**不走 `add_messages` reducer**
- 路由用普通 `if/elif/else`，**不走 conditional edge**
- LLM 调用次数、tool 调用次数和 graph 版完全一致

满足这些，`graph_latency − baseline_latency` 才能干净地归因到"LangGraph 编排"这一层。

5 个 case 各自的 baseline 形态：

| Case | Baseline 控制流 |
|---|---|
| **02 tool_use** | `last = llm.invoke(msgs); while last.tool_calls: msgs += [tool_invoke(call)]; last = llm.invoke(msgs)` |
| **03 ReAct** | 同 02，多跳就多转几圈 |
| **04 planning** | `plan = planner.invoke(req); for step in plan: results.append(tool_invoke(step)); final = synth.invoke(results)` |
| **11 meta_controller** | `branch = controller.invoke(query); answer = {"Generalist": g, "Researcher": r, "Coder": c}[branch].invoke(query)` |
| **15 RLHF** | `draft = gen.invoke(req); for _ in range(3): c = crit.invoke(draft); if c.is_approved: break; draft = rev.invoke(draft, c)` |

注意 02 和 03 的 baseline 控制流是一样的 —— 这本来就是 `03_ReAct_analysis.md` 里强调过的核心：**ReAct 和 basic tool use 在代码层面就差一条 `tools → agent` 边**。所以 baseline 就用同一个循环写法，区别全在 query pool 设计的复杂度上。

## 6. `run.py` 契约

CLI：

```
python -m benchmarks.run \
    --case 03_react \
    --variant graph|baseline \
    --workload micro|macro \
    --iterations N \
    --warmup K \
    --json out.jsonl \
    --marker-file markers.txt
```

每轮迭代写一行 JSON 到 `--json` 指定的文件（jsonl 格式）：

```json
{
  "case": "03_react",
  "variant": "graph",
  "workload": "macro",
  "iter": 17,
  "query_idx": 17,
  "query_hash": "ab12cd34",
  "t_start_ns": 1234567890,
  "t_end_ns": 1234599999,
  "latency_ns": 32109,
  "trace": {
    "hop_count": 3,
    "msg_count": 8,
    "tool_calls": 3,
    "regex_fallbacks": 0
  },
  "exception": null
}
```

**异常处理**：每轮迭代外层 `try / except`，捕获后把 `exception: {"type": ..., "message": ...}` 写进当前行，**继续下一轮**。原因还是"方差是信号" —— planner 偶发崩溃、structured output 校验失败本身就是编排层在野外会遇到的情况，不是中断 benchmark 的理由。

`--marker-file` 写的是阶段标记，每行一条：

```
warmup_start  t_ns=...  monotonic_ns=...
warmup_end    t_ns=...  monotonic_ns=...
bench_start   t_ns=...  monotonic_ns=...
bench_end     t_ns=...  monotonic_ns=...
```

emon 在外面采样时拿这个文件去对齐采样窗口，避免把 warmup 阶段也算进性能数据。

stderr 输出（不污染 jsonl）：启动时打 tracing 状态、LLM 端点（`base_url` 和 `model`，**永远不打 api_key**）、case / variant / workload / 迭代数；结束时打总结：

```
iters=200  exceptions=2  p50=...  p99=...  hop_count_dist={1: 12, 2: 130, 3: 56, 4: 2}
```

## 7. Query pool 起草说明

`query_pool.py` 形如：

```python
POOLS: dict[str, list[str]] = {
    "02_tool_use": [...],
    "03_react": [...],
    "04_planning": [...],
    "11_meta": [...],
    "15_rlhf": [...],
}
```

每个 pool 由 AI 按 case 语义起草，用户审阅修改。规模和形态目标：

- **02 tool_use** （~30 条）：单点事实查询，且必须是训练数据 cutoff 之后才有答案的，例如 "What was announced at the latest Google I/O?"。预期轨迹：1 次 tool call。
- **03 ReAct**（~40 条）：真正的多跳事实链，例如 "Who directs the studio that made Spirited Away, and what was their last film's release year?"。预期轨迹：2–4 跳，方差自然存在。
- **04 planning**（~30 条）：可分解成 N 个独立查询 + 一次聚合的任务，例如 "Find the populations of the three largest cities in Spain and rank them."。预期 plan 长度：3–5 步。
- **11 meta_controller**（~30 条，三个分支大约各 ~10 条）：均衡覆盖 Generalist（闲聊/简单问答）、Researcher（时事查询）、Coder（"写一个 Python 函数 ..."）。重点不是某一分支跑得多好，而是 controller 路由本身的开销和准确性分布。
- **15 RLHF**（~30 条）：营销邮件 brief，描述详略不一，例如 "Write a launch email for our new B2B observability platform 'Sigil'"。预期大部分会过 ≥1 次 revision。

每条 query 在文件里附上简短中文注释，说明该条预期触发的轨迹形态，方便用户审阅时判断是否对齐。

## 8. `analyze.py` 范围

刻意做薄，emon 才是主力：

- **分布统计**：count / mean / p50 / p90 / p99 / max / stddev of `latency_ns`
- **轨迹直方图**：`hop_count`、`msg_count`、`tool_calls`、`regex_fallbacks`（适用时）
- **异常率**：按 exception type 分组
- **配对差分**：传入两个 jsonl 文件（典型用法是同 case 的 graph vs baseline），按 `query_idx` 配对算 `Δlatency`，输出 p50/p99 差和均值差

不做花哨可视化，输出是文本表格。需要画图时用户自己拿 jsonl 喂给别的工具。

## 9. 复用清单

下面这些 LangGraph / LangChain API 直接用现成的，**不重造轮子**：

- `langgraph.graph.StateGraph`：状态机本体
- `langgraph.graph.message.add_messages`：消息列表的 reducer
- `langgraph.prebuilt.ToolNode`：工具执行节点
- `langgraph.prebuilt.tools_condition`：判断 last message 有没有 `tool_calls` 的条件边
- `langchain_openai.ChatOpenAI`：连本地 OpenAI 兼容服务
- `langchain_core.tools.tool` 装饰器（04 里用到）
- `langchain_core.messages.{AIMessage, ToolMessage, SystemMessage, HumanMessage, BaseMessage, AnyMessage}`：baseline 构造消息
- 各 notebook 里的 pydantic 模型：`Plan`（04）、`Critique` / `MarketingEmail`（15）、`ControllerDecision`（11）、`ToolUseEvaluation`（02 评估部分，benchmark 不用）—— 直接复制原文，**不简化**。这些 schema 自身就是被测对象的一部分。

`langchain_nebius` 在 benchmark 里**不引入**：本地服务走 OpenAI 协议，`ChatOpenAI` 即可。

5 个 notebook 文件本身保持不动，benchmark 是独立产物。

## 10. 验证步骤

1. **烟雾测试**（不接 emon）：
   ```bash
   BENCH_LLM_BASE_URL=... BENCH_LLM_MODEL=... BENCH_LLM_API_KEY=... \
     /home/zeus/all-agentic-architectures/.venv/bin/python -m benchmarks.run \
       --case 03_react --variant graph --workload micro \
       --iterations 5 --warmup 1
   ```
   预期：5 行 JSON、`latency_ns > 0`、`trace.hop_count >= 1`，stderr 显示 tracing 已关。

2. **tracing 关闭断言**：`run.py` 启动时检查 `os.environ["LANGCHAIN_TRACING_V2"]` 必须是 `"false"` / `"0"` / `""` 之一，否则直接退出。手动 `export LANGCHAIN_TRACING_V2=true` 再跑一次，必须报错。

3. **graph / baseline 对齐**：每个 case 用 macro pool 的前 5 条同时跑 graph 和 baseline，比对 `tool_calls` 必须**完全相等**（FakeTool 是确定的，LLM 在 `temperature=0` 下对相同输入决策也应一致），final answer 容许文本差异。如有 case 对不上，说明 baseline 的控制流和 graph 偏离了，必须修。

4. **Pool 审阅**：用户读 `query_pool.py`，原地修改即可，不需要其它代码改动。

5. **emon marker 对齐 dry-run**：用 `--marker-file` 跑一次，emon 在外侧抓 `bench_start ↔ bench_end` 区间，确认 marker 时间戳和 emon 采样窗口对得上。

## 11. 实施顺序

1. 先搭骨架：`benchmarks/__init__.py`（tracing kill-switch）、`benchmarks/llm.py`、`benchmarks/fakes.py`、`benchmarks/query_pool.py`（先放一份草稿 pool）、`benchmarks/run.py`（CLI + per-iter 计时 + marker）
2. 落 **case 03 + baseline**：最有代表性（循环 + 回边），作为后续 4 个 case 的模板
3. 烟雾测试 03，根据真实输出微调 jsonl schema 和 marker 格式
4. 复制套路依次落 02 / 04 / 11 / 15 + 各自 baseline
5. `analyze.py`
6. `benchmarks/README.md`：环境变量契约、marker 格式、运行示例、第 10 节的 5 步验证

实施过程中如果发现 LangGraph 1.2 的 API 和 notebook 用法有出入，以**当前装的版本**（langgraph 1.2.1 / langchain 1.3.1 / langchain-core 1.4.0 / langchain-openai 1.2.2 / langchain-tavily 0.2.18 / pydantic 2.13.3）为准。


