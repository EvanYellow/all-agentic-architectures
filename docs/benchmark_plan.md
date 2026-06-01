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
2. **Tool**：`FakeTool` 替换 `TavilySearch` / `TavilySearchResults`，零网络 I/O。具体形态见 §2.A：单参数 KV `lookup` + 合成 KG，返回 5–15 token 字符串。**不**复刻 Tavily 的 dict 结构 / 长 snippet，目的就是把上下文压扁，让 LLM 端不被 prefill 主导。
3. **方差是信号**：不丢弃任何一轮、不写"轨迹守卫"，每轮迭代记录 `(hop_count, msg_count, tool_calls, latency_ns, ...)`，最终输出**分布**而不是均值。生产里的编排层就是要面对变长上下文和分支次数浮动，这部分方差恰恰是要测的。
4. **两种 workload**：
   - **A — micro**：一个简单 query 反复跑，给 emon 一段稳定 hot path。
   - **B — macro**：完整 query pool 顺序循环，让轨迹形态自然变化，测的是编排层在真实输入分布下的画像。
5. **每个 case 配一个 baseline**：用裸 Python 复刻同样的控制流（不引 `StateGraph`、不引 `ToolNode`、不引 `add_messages`）。`graph_latency − baseline_latency` 才是编排层的纯开销。
6. **Query pool**：每个 case 由 AI 按语义起草 20–50 条，写到 `query_pool.py` 里供用户审阅修改。
7. **LangSmith tracing 强制关闭**：所有 notebook 默认开 `LANGCHAIN_TRACING_V2=true`，会在每个节点回调里发 HTTP 到 LangSmith，对 benchmark 是巨大污染。`benchmarks/__init__.py` 在任何 LangChain 模块 import 之前把它强制关掉。

### 2.A FakeTool 形态：`lookup` + 合成 KG

替换 Tavily 的不是"另一个搜索 API mock"，而是**完全换一种 tool 接口**。具体到代码：

```python
# benchmarks/fakes.py
from langchain_core.tools import tool

KB: dict[str, str] = {
    # 首都:<国家>  -> <城市token>
    "首都:国家A1": "tok-7Q3X",
    # 人口:<城市token>  -> <数字>
    "人口:tok-7Q3X": "12345",
    # 创始人:<公司>  -> <人物token>
    "创始人:公司K2": "tok-3H6Y",
    # 出生年:<人物token> -> <年份>
    "出生年:tok-3H6Y": "1971",
    # ... 按 query pool 反推补,每个 case 共用同一份 KB
}

@tool
def lookup(key: str) -> str:
    """从知识库中查询一个事实。

    Key 形如 '<关系>:<实体>',例如 '首都:国家A1' 或 '出生年:tok-3H6Y'。
    返回 value 字符串;若 key 不存在则返回字面量 'NOT_FOUND'。
    """
    return KB.get(key, "NOT_FOUND")
```

工程上几条硬约束，**每一条都直接决定 hop 形态对不对得上**，落地时不要省：

1. **单参数 `key: str`**。不是 `lookup(关系, 实体)` 两个参数，也不暴露 batch 接口（`lookup_many(keys)`）。两个参数会让模型把 entity 默认 `None` 一次发并行，多跳塌成一跳；batch 接口同理。03 ReAct 之所以能串行，唯一根本约束就是单参数。
2. **`key` 形如 `"<关系>:<实体>"`**。docstring 说清 schema 就够，不做正则校验也不做切分。错的 key 走 `NOT_FOUND` 分支，让模型 in-band recover —— 这是 ReAct 在野外的真实形态，恰好也是要测的。
3. **返回值是裸 `str`**，不是 dict / pydantic 模型。`"tok-3H6Y"` 五个 token 包成 `ToolMessage` 后大概 ~15 token；返回 dict 序列化后翻倍。
4. **`NOT_FOUND` 字面量而非异常**。抛异常会被 `ToolNode` 包成错误 message 走另一条路径，引入跟"上下文"无关的额外编排开销。
5. **KB 是模块级常量**。跨 iteration 复用同一个 `lookup` 实例和 KB dict，不要每轮重建。
6. **Key 用中文关系名 + 中文实体名**。跟 query pool 语言一致（中文 query），避免模型在 query 中文 / KB 英文之间做翻译那一层与编排开销无关的 LLM 时间；token 端开销也更接近国内业务的真实分布。**`tok-XXXX` 这种"不可猜 token" 保留 ASCII** —— 它们的设计目标就是无语义，换中文反而更容易被先验猜中。

### 2.B Entity 命名：第二跳的值必须不可猜

**关键到不能省的一条**：合成 KG 里，"被 lookup 出来作为下一跳 key 一部分"的 value 必须是 LLM 没法靠先验拼出来的 token。

具体来说，2-hop chain 形如：
```
hop1: lookup("创始人:公司K2")  -> X
hop2: lookup("出生年:" + X)    -> 1971
```
`X` 必须是不可猜的（用 `tok-3H6Y` 这种合成 token），**不能**用语义连贯命名（例如 `人物-K2`、`Person-L9`）。否则模型会从 schema 命名规律 + 训练先验直接拼出第二跳的 entity，在第一跳还没返回之前就把第二跳一起塞到同一个 AIMessage 的 `tool_calls` 里。

实测过：当第二跳 entity 命名是 `Person-L9` 时，DeepSeek-V3 在 basic graph 下一发出 `[lookup(创始人:公司K2), lookup(出生年:Person-L9)]` 两个并行 tool_call，靠先验 one-shot 通过。把 value 改成 `tok-3H6Y` 后，模型在 basic 下仍然会瞎猜（实测在中文版本里它把第二跳猜成 `出生年:公司K2`），第二跳直接落到 `NOT_FOUND`，basic 失败、ReAct 老老实实分两步拿到正确答案 —— 这才是这两个 case 想测的形态。

实操规则：
- **作为下一跳 key 一部分的 value**：用 `tok-{4-字符随机十六进制大写}`（例：`tok-3H6Y`），ASCII 保留
- **只作为最终答案的 value**（数字、年份等）：保持普通字面量，不影响形态
- **作为 query 输入的 entity**：可以保留语义命名（`公司K2`、`国家A1`），便于人审 query pool

### 2.C ReAct system prompt 模板

03 case 的 system prompt 用下面这段，03 baseline 复用同一段：

```
你是一个严谨的 ReAct agent,可以使用 `lookup` 工具查询一个小型知识库。
Key 形如 '<关系>:<实体>'。
可用关系(只能使用这些精确名称,不要使用同义词):
首都、人口、创始人、出生年、母公司、CEO。
把问题拆解成单事实查询,每个事实调用一次 tool;
当下一个 key 需要前一次 lookup 返回的 value 来拼接时,必须等拿到 value 之后再发下一次调用。
拿到所有事实之后停下并给出答案。
```

四条都是必要的：

- **"可用关系"列表** —— 不列就会拼错（实测过：`创办人:` vs KB 里的 `创始人:`，第一跳就 `NOT_FOUND` 直接放弃）。这条不是为了让模型表现好看，而是把"key schema 拼写错误"这一类与编排开销无关的失败模式从 trace 里移除。
- **"每个事实调用一次 tool"** —— 显式禁止合并并行 call，prompt 层面的弱保证。
- **"必须等拿到 value 之后再发下一次调用"** —— 显式声明数据依赖，引导模型选串行路径。
- **"拿到所有事实之后停下并给出答案"** —— 防止某些 model 在拿到答案后还继续多发一次 lookup 探活。

02 case basic prompt 简化版（**保留**"一次 tool 调用之后给出最终答案"这句，让 02 的单跳形态成立）：

```
你是一个有用的助手,可以使用 `lookup` 工具查询一个小型知识库。
每个事实需要一次 lookup。
可用关系(只能使用这些精确名称):首都、人口、创始人、出生年、母公司、CEO。
你必须在一次 tool 调用之后给出最终答案。
```

### 2.D `parallel_tool_calls=False` 不可依赖

`bind_tools(..., parallel_tool_calls=False)` 这个参数 **DeepSeek-V3 / SiliconFlow 静默忽略**。只有 OpenAI / Anthropic 真接口尊重。所以"react 每步只发 1 个 tool_call"在 benchmark 里**不能**靠这个参数保证，得靠 §2.B 的不可猜 entity + §2.C 的 prompt 引导组合出来。

如果某 case 必须 deterministic 单 call（例如 `docs/benchmark_plan.md` §10.3 graph/baseline 对齐时 `tool_calls` 计数要严格相等），用节点内截断兜底：

```python
def react_agent_node(state):
    response = llm_with_tools.invoke([("system", REACT_SYS)] + state["messages"])
    if len(getattr(response, "tool_calls", []) or []) > 1:
        response.tool_calls = response.tool_calls[:1]
    return {"messages": [response]}
```

这改了 graph 的"教学版自由决策"语义，但作为 benchmark 形态固化是合理的 trade-off —— 跟 §2.A 把 Tavily 换成 lookup 是同一类决策。**默认不开**，落地时只在确实跑出方差 > 阈值的 case 加。

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

每个 pool 由 AI 按 case 语义起草，用户审阅修改。**所有涉及 lookup 的 pool（02 / 03 / 04）都基于 §2.A 的合成 KG**，query 措辞模板化、entity 名直接引用 KB 里的 token，避免任何"凭先验答得出来"的混入。**query 用中文，跟 KB 的关系名保持一语种**。

规模和形态目标：

- **02 tool_use** （~30 条）：**单跳**事实查询，模板形如 "{实体} 的 {关系} 是什么？"，例如 "国家A1 的首都是什么？"。预期轨迹：1 次 tool call。
- **03 ReAct**（~40 条）：**真正的多跳链**，第二跳的 entity 必须由第一跳 lookup 的返回 token 拼出来（见 §2.B），例如 "国家A1 的首都的人口是多少？"（KB: `首都:国家A1 → tok-7Q3X`，`人口:tok-7Q3X → 12345`）。预期轨迹：2–4 跳，方差自然存在。**审阅 query pool 时第一条要查的就是：第二跳的实体名是不是直接出现在了 query 里？如果是，hop 直方图会塌成单峰**。
- **04 planning**（~30 条）：可分解成 N 个**独立**子 lookup + 一次聚合，例如 "查询 国家A1、国家A2、国家A3 三个国家首都的人口并排序"。注意"独立"的语义在合成 KG 上等价于"N 个不相互依赖的 hop1 + 一个聚合"，每个 hop1 内部仍然可以是 1–2 跳，由 query 决定。预期 plan 长度：3–5 步。
- **11 meta_controller**（~30 条，三个分支大约各 ~10 条）：均衡覆盖 Generalist（闲聊/简单问答）、Researcher（基于合成 KG 的事实查询）、Coder（"写一个 Python 函数 ..."）。重点不是某一分支跑得多好，而是 controller 路由本身的开销和准确性分布。
- **15 RLHF**（~30 条）：营销邮件 brief，描述详略不一，例如 "为我们的新 B2B 可观测性平台 'Sigil' 写一封发布邮件"。预期大部分会过 ≥1 次 revision。**15 不接 lookup tool**，跟其它 case 共用 LLM 但不共用 KB。

每条 query 在文件里附上简短中文注释，说明该条**预期触发的 hop 数 / lookup key 序列**，方便用户审阅时判断是否对齐。例如：

```python
# 03_react: 期望 2 跳；hop1 创始人:公司K2 -> tok-3H6Y；hop2 出生年:tok-3H6Y -> 1971
"公司K2 的创始人是谁？这个人是哪一年出生的？",
```

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

## 12. 附录：FakeTool + KB 设计的早期实测

`scripts/bench_03_react.py` 是按 §2.A–§2.D 写的最小验证脚本，构造方式跟最终 `benchmarks/cases/case_03_react.py` 等价，只多了一层 `LangChain` callback handler 收集 per-step trace（latency / prompt_tokens / completion_tokens / tool_calls 预览），输出 jsonl。

**用法**：

```bash
.venv/bin/python scripts/bench_03_react.py
.venv/bin/python scripts/bench_03_react.py --runs 3
```

**已验证数字**（model: `deepseek-ai/DeepSeek-V3`，单轮，2-hop 中文 query "公司K2 的创始人是谁？这个人是哪一年出生的？"）：

| 项 | basic | react |
|---|---|---|
| LLM 调用 | 1 | 3 |
| Tool 调用 | 2 | 2 |
| **每次 Tool 延迟** | **0.3–0.5 ms** | **0.3 ms** |
| **编排开销 = total − LLM − Tool** | **~5–7 ms** | **~5–6 ms** |
| Σ prompt_tok | 226 | 945 |
| 第 1/2/3 跳 prompt_tok（react） | — | 279 → 315 → 351 |
| 单跳 prompt 增量（一条 ToolMessage） | — | ~36 token |

跟换上 Tavily 跑同样 query 时的 ~1000–3000 ms tool latency + 单跳 prompt 几千 token 比，**FakeTool 那一档 secs 的 tool 时间彻底没了**，prompt 累积也压扁了一个数量级。这就给 `graph_latency − baseline_latency` 留出可观测的窗口。

**已踩到的坑（落 `benchmarks/` 时不要重蹈）**：

1. **第二跳 entity 必须不可猜**：第一版 KB 用 `Person-L9` 做 founder value，DeepSeek-V3 的 basic 直接一发两个并行 tool_call 把 chain 猜对（`founder_of:Company-K2 + birthyear_of:Person-L9`），03 教学版 basic 单跳形态被破坏。换成 `tok-3H6Y` 后中文 basic 仍然会瞎猜（实测把第二跳猜成 `出生年:公司K2`），第二个并行 call `NOT_FOUND` 失败、ReAct 老老实实分两步走通。这条已沉淀到 §2.B。
2. **可用关系列表必须放进 prompt**：第一版没列，模型把 `创始人:` 拼成 `创办人:`、`founder_of:` 拼成 `founded_by:`，第一跳直接 `NOT_FOUND` 放弃。这条已沉淀到 §2.C。
3. **`parallel_tool_calls=False` 在 SiliconFlow / DeepSeek-V3 上是 noop**：basic 还是发了两个并行 call，证实这个参数靠不住。`benchmarks/` 落地时如果有 case 必须 deterministic 单 call，用 §2.D 的节点内截断。
4. **教学版 notebook 的 prompt 不要直接搬**：notebook 里没有"可用关系"列表也没有"单事实查询"约束，对真 Tavily 没问题（Tavily 的 query 是自由文本），但接 lookup tool 后会把 schema 描述失败模式跟编排开销混在一起。`benchmarks/cases/case_03_react.py` 的 prompt 用 §2.C 模板，不复用 notebook 文本。
5. **KB / prompt / query 同语言**：第一版 KB / prompt 是英文、用户 query 是中文时，模型会先把"创始人"翻译成"founder_of"再拼 key，多了一层与编排开销无关的 LLM 延迟。统一中文后这层消失。

脚本本身**不进 `benchmarks/`**，它是一次性诊断工具，完成 §2 设计验证后归档；`benchmarks/cases/case_03_react.py` 落地时直接对照本文档第 2 节写。


