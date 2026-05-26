# 03_ReAct.ipynb 分析笔记

本笔记是对原项目第 3 个 notebook（ReAct: Reason + Act）的拆解，记录了把模型接口从 Nebius 切换到 SiliconFlow（OpenAI 兼容协议）后的运行情况，以及对 agent 实际行为的分析。

## 1. 环境与接口改动

原 notebook 使用 `langchain-nebius` 调用 `meta-llama/Meta-Llama-3.1-8B-Instruct`。本 fork 改成走 SiliconFlow 的 OpenAI 兼容端点：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V4-Flash",
    api_key=os.environ.get("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    temperature=0,
)
```

`bind_tools` / `with_structured_output` / LangGraph 编排都不用改——这是 OpenAI 兼容协议的好处：业界绝大多数 LLM 服务商都暴露这种接口。

`.env` 模板见仓库根的 `.env.example`，需要：

- `SILICONFLOW_API_KEY` — LLM 调用，<https://siliconflow.cn>
- `TAVILY_API_KEY` — agent 工具，<https://www.tavily.com>（免费档每月 1000 次）
- `LANGCHAIN_API_KEY` — 可选，LangSmith tracing

## 2. notebook 实际跑的任务

整本 notebook 围绕一道多跳问答：

> Who is the current CEO of the company that created the sci-fi movie 'Dune', and what was the budget for that company's most recent film?

答案不在任何单一网页上，需要把以下事实串起来：

```
Dune → 制片公司 = Legendary Entertainment   （第 1 跳）
Legendary Entertainment → CEO = Joshua Grode  （第 2 跳）
Legendary → 最新电影 = ?                       （第 3 跳）
那部电影 → 制作预算 = ?                        （第 4 跳）
```

**Agent 的工具集只有一个：`web_search`（Tavily 包装的搜索引擎）**。所有"智能"都体现在它如何反复调这一个工具。

## 3. 两个 agent 的结构对比

### 3.1 Basic 单次工具 Agent（线性，无环）

```
            ┌─────────────┐
            │   START     │
            └──────┬──────┘
                   │  user query
                   ▼
            ┌─────────────┐
            │   agent     │   ChatOpenAI
            │  (LLM)      │   + bind_tools([web_search])
            └──────┬──────┘
                   │
       tools_condition (路由)
        ┌──────────┴──────────┐
        │ has tool_call?      │
        ▼ yes                 ▼ no
 ┌─────────────┐       ┌─────────────┐
 │   tools     │       │     END     │
 │  ToolNode   │       └─────────────┘
 │ (Tavily)    │
 └──────┬──────┘
        │ tool result
        ▼
 ┌─────────────┐
 │     END     │       ← 走完一次工具就结束，不回头
 └─────────────┘
```

**缺陷**：`tools → END` 是单向死路。模型搜完一次拿到结果，没机会再"读完结果再思考一次"，所以多跳问题答不出来。

### 3.2 ReAct Agent（带 Reason ↔ Act 闭环）

```
            ┌─────────────┐
            │   START     │
            └──────┬──────┘
                   │  user query
                   ▼
         ╔══════════════════╗
         ║                  ║
         ║   ┌──────────┐   ║
         ║   │  agent   │   ║   ← Reason
         ║   │  (LLM)   │   ║     看 messages 历史，决定下一步
         ║   └────┬─────┘   ║
         ║        │         ║
         ║   react_router   ║
         ║    ┌───┴───┐     ║
         ║ no │       │ yes ║
         ║    ▼       ▼     ║
         ║  END    ┌──────┐ ║
         ║         │tools │ ║   ← Act
         ║         │Tavily│ ║     执行 web_search
         ║         └──┬───┘ ║
         ║            │     ║
         ║            └─────╝   ← Observe
         ║              回 agent，把 tool 结果塞进 messages
         ╚══════════════════╝
                  │
                  ▼
                 END
```

**唯一但关键的差别**：Basic 图里 `tools → END` 这条边，被 ReAct 换成了 `tools → agent`。代码差别只有一行：

```python
react_graph_builder.add_edge("tools", "agent")   # ← 闭环的根源
```

### 3.3 一次完整执行（消息状态视角）

ReAct 的"记忆"全在 `state["messages"]` 这个 list 里，每跑一圈就追加新消息：

```
回合 1                回合 2                回合 3
─────────             ─────────             ─────────
[HumanMsg]            [HumanMsg]            [HumanMsg]
    │                 [AIMsg+toolcall1]     [AIMsg+toolcall1]
    ▼                 [ToolMsg result1]     [ToolMsg result1]
agent 思考             │                    [AIMsg+toolcall2]
    │                 ▼                     [ToolMsg result2]
    ▼                 agent 再思考           │
[AIMsg+toolcall1]     │                     ▼
    │                 ▼                     agent 综合
    ▼                 [AIMsg+toolcall2]    [AIMsg final answer]
tools 执行            │                     │
    │                 ▼                     ▼
    ▼                 tools 再执行          router → END
[ToolMsg result1]     │
loop ↩                ▼
                     [ToolMsg result2]
                     loop ↩
```

每次 agent 节点被调用时，prompt 里包含历史所有 tool 返回，所以模型能基于"我已经知道 Legendary 是 Dune 制片方"接着想"那 Legendary 的 CEO 是谁"。

### 3.4 决策路由的伪代码

```python
def react_router(state):
    last = state["messages"][-1]
    if last.tool_calls:        # 模型自己决定还要查 → 继续转
        return "tools"
    return "__end__"            # 模型不再发 tool_call → 收工
```

**何时停**完全由模型自己决定——它认为信息够了就直接输出 final answer（没有 tool_calls 字段），路由就把图引到 END。这也是 ReAct 的风险点：模型如果判断错（信息其实不够却以为够了），就提前结束；或者反过来反复搜同样的东西卡死循环——所以生产里通常还要给 graph 加 `recursion_limit`。

## 4. 这个 demo 模拟的现实场景

Notebook 名义上"查 Dune 的 CEO"，实际抽象的是这一类任务：**信息分散在多个来源、下一步搜什么取决于上一步搜到什么、任务者一开始并不知道完整路径**。

| 现实任务 | 多跳结构 |
|---|---|
| 尽职调查 / 投研 | "X 公司被谁收购了 → 收购方的 CEO 是谁 → 那个 CEO 之前在哪家公司" |
| 客服 / 技术支持 | "用户的报错码是什么 → 这个码对应哪个组件 → 这个组件最近的 release notes 写了什么" |
| 法律 / 合规检索 | "这个条款引用了哪个法规 → 那个法规去年是否被修订 → 修订后哪些条款受影响" |
| 学术文献综述 | "论文 A 的方法基于 B → B 的开源代码在哪 → 代码用的数据集许可是什么" |
| 新闻溯源 | "这条消息最早谁发的 → 那个账号过去有什么背景 → 是否被官方辟谣" |
| 运维排障 | "服务挂了的时间点 → 那时上线了哪个版本 → 那个版本改了什么文件" |

**Basic vs ReAct** 对应到的人类行为：

- **Basic agent** 像一个新手实习生：把整个长问题原封不动丢进 Google 搜索框，看到第一页结果就开始写答案——大概率只能回答其中一小部分。
- **ReAct agent** 像一个有经验的研究员：先拆解问题，搜"Dune 制片公司" → 看到 Legendary → 再搜"Legendary CEO" → ... → 最后把所有线索拼成完整答案。

## 5. ReAct 的核心价值不在 web_search

这个 demo 工具集只有 `web_search` 一个，是教学上的**降噪**：留一个工具、一个朴素任务，让读者只盯住"循环"这一个变化。如果再塞进来计算器、数据库、代码执行……就分不清"agent 变强"是因为工具多了还是因为有了 reason→act 闭环。

把 tools 列表从 `[web_search]` 换成下面任一组合，结构完全不变，但能力天差地别：

| Agent 类型 | tools 列表 |
|---|---|
| Claude Code / Cursor | `read_file`, `edit_file`, `bash`, `grep`, `glob` |
| 数据分析 agent | `run_sql`, `run_python`, `plot` |
| 客服 agent | `query_orders`, `refund`, `escalate_to_human` |
| 运维 agent | `kubectl`, `read_logs`, `restart_service` |
| 通用助理 | `web_search`, `send_email`, `calendar`, `read_doc` |

ReAct 的核心价值不在 web_search，而在 **"循环 + 工具调用"这个组合允许模型用环境反馈来修正自己的下一步**。这个 demo 只是用最小代价把这件事演给你看。

## 6. 已知小问题

- 原 notebook 用的是已弃用的 `langchain_community.tools.tavily_search.TavilySearchResults`，会有 deprecation warning，但仍能用。新代码建议迁到 `langchain_tavily.TavilySearch`。
- 用 8B 小模型（Llama-3.1-8B-Instruct）跑 ReAct，多跳推理经常一轮就停（注意原 notebook 里 ReAct 也只搜了一次就给了"找不到预算"的部分答案）。换成 DeepSeek/Qwen-72B 这类更强的 tool-calling 模型时通常能多跳几次。
- ReAct 没有内置防死循环。生产中应给 `compile()` 加 `recursion_limit`，或在 router 里加最大步数判断。
