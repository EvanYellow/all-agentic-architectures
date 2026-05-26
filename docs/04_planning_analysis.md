# 04_planning.ipynb 分析笔记

## 1. notebook 实际跑的是什么

把 ReAct（边走边想）和 Planning（先想完再走）放在**同一个任务**上对照跑：

> Find the population of the capital cities of France, Germany, and Italy. Then calculate their combined total. Finally, compare that combined total to the population of the United States, and say which is larger.

任务的特点：**步骤可以提前拆出来、互相之间没有依赖**——三个搜索 + 一个加法 + 一个比较。这种"流程预定"的题目正好暴露 ReAct 的低效（每步之间都要 LLM 思考一次）和 Planning 的优势（一次性把计划写出来，按部就班执行）。

## 2. 两种架构

### 2.1 ReAct（Phase 1，老朋友）

```
            ┌─────────┐
            │  agent  │  ← 每一步都要 LLM 想
            └────┬────┘
                 │
         tools_condition
            ┌────┴────┐
            ▼         ▼
        ┌──────┐    END
        │tools │
        └──┬───┘
           │
           └──── 回到 agent
```

代码上加了一行 system prompt 强制"一次只调一个工具"，让 ReAct 退化成"每次走一步"，方便和 Planning 公平对比：

```python
SystemMessage(content=
    "You are a helpful research assistant. You must call one and only one tool "
    "at a time. Do not call multiple tools in a single turn. After receiving "
    "the result from a tool, you will decide on the next step."
)
```

### 2.2 Planning（Phase 2，新模式）

```
   ┌─────────────────┐
   │     plan        │   Planner LLM：一次性输出 List[str] 步骤
   │   (Planner)     │
   └────────┬────────┘
            │ plan = ["web_search('pop Paris')", "web_search('pop Berlin')", ...]
            ▼
   ┌─────────────────┐         ┌────────────────────┐
   │    execute      │◀────────│  planning_router   │
   │  (Executor)     │         │  plan still has    │
   │  按 plan[0] 执行 │         │  steps?            │
   └────────┬────────┘         └────────────────────┘
            │ pop plan[0]，记结果
            ▼
   ┌─────────────────┐
   │  synthesize     │   Synthesizer LLM：基于全部 ToolMessage 合成答案
   └────────┬────────┘
            ▼
           END
```

三个角色分得很清楚：

- **Planner** = 思考层（**只在开头调一次 LLM**）
- **Executor** = 行动层（不调 LLM，纯执行 tool）
- **Synthesizer** = 合成层（**最后调一次 LLM**）

对比 ReAct：N 次工具调用 = N 次 LLM 思考；Planning：N 次工具调用 = 2 次 LLM 调用（1 次 planner + 1 次 synthesizer）。

## 3. 关键代码点

### 3.1 用 Pydantic 强制 Planner 输出结构化计划

```python
class Plan(BaseModel):
    steps: List[str] = Field(description="A list of tool calls...")

planner_llm = llm.with_structured_output(Plan)
```

不用 `with_structured_output`，模型可能输出散装文本，下游解析极易崩。这是 Planning 模式落地的**第一道护城河**——没有结构化输出，Plan 这个抽象就是空的。

### 3.2 Few-shot 提示让 Planner 知道"步骤长什么样"

```python
prompt = f"""
...
**Example:**
Request: "What is the capital of France and what is its population?"
Correct Plan Output:
[
    "web_search('capital of France')",
    "web_search('population of Paris')"
]
...
"""
```

注意一个细节：**Planner 只规划"调哪个工具、传什么参数"，不规划计算逻辑**。从实际跑出来的 plan 也能看到这个限制——它把"加法"和"比较"也写成了 web_search：

```
"web_search('add  + population of France + population of Germany + population of Italy')"
"web_search('compare + combined total + population of the United States')"
```

这是 demo 的瑕疵：Planner 只有一个工具能用，所以**计算/比较被强行塞进 web_search**。生产里如果要靠 plan 做算术，应当再给 agent 加个 `calculator` 或 `python_exec` 工具，让 Planner 可以选择。

### 3.3 Executor 用正则解析 Plan 字符串

```python
match = re.search(r"(\w+)\((?:\"|\')(.*?)(?:\"|\')\)", next_step)
```

把 `web_search('xxx')` 拆成 `("web_search", "xxx")`。这种字符串 → 调用的反序列化是 Plan 类架构常见的**脆弱点**：

- Planner 一旦返回不规则格式（带嵌套引号、带多个参数）就 break
- 多工具场景下 tool_name 还要用查表分发
- 更稳的方案：让 Planner 直接输出 `List[ToolCall]`（每个 ToolCall 是带 tool_name + args dict 的 Pydantic），跳过字符串解析这一层

### 3.4 双条件边构成的循环

```python
planning_graph_builder.add_conditional_edges("plan", planning_router,
    {"execute": "execute", "synthesize": "synthesize"})
planning_graph_builder.add_conditional_edges("execute", planning_router,
    {"execute": "execute", "synthesize": "synthesize"})
```

两个节点的出边都接同一个 `planning_router`，根据 `state["plan"]` 是否还有剩余步骤决定继续执行还是跳到合成。这就是 LangGraph 里实现"循环执行计划"的标准写法。

## 4. ReAct vs Planning：实际跑下来的差别

| 维度 | ReAct（Phase 1） | Planning（Phase 2） |
|---|---|---|
| LLM 调用次数 | 每步一次（N+1 次） | 2 次（plan + synthesize） |
| 计划是否可见 | 不可见，得边跑边看 | 一开始就打印出来 |
| 中途能否调整 | 能（每步都重新想） | 不能（plan 是固定的） |
| 失败模式 | 卡在某步上反复试 | 计划本身错了就一路错下去 |
| 适用任务 | 路径未知、依赖中间结果 | 步骤可枚举、相互独立 |

一句话：**ReAct 用 LLM 调用换灵活性，Planning 用预先思考换效率和透明度**。

## 5. Demo 暴露的 Planning 局限

跑下来的最终答案，Planning agent 给的人口数其实没那么准（"Paris 2,048,472 (according to Instagram)" 这种来源就有点离谱）。原因是 demo 的 Planner 只能把所有事情都包成 web_search，所以**搜回来的数据靠合成器去解析数字**——而 Mixtral/Llama 之类的模型在解析非结构化网页内容上并不可靠。

这反映了 Planning 模式的两个真实痛点：

1. **计划的颗粒度受限于工具集**：工具集越窄，Planner 越不得不"硬塞"。给 Planner 配一个 `calculator` 工具，加法那一步就不会变成 `web_search('add ...')`。
2. **环境变化时计划僵硬**：第二个搜索结果意外为空，ReAct 会立刻换 query；Planning 会把空结果带到下一步，最后让 Synthesizer 收拾烂摊子。

## 6. 这个 demo 模拟的现实场景

抽象出来：**任务流程能在开始时被一个有经验的人/模型大致写下来**。

| 现实任务 | Plan 大致长什么样 |
|---|---|
| 月度报表 | `[fetch_sales(month), fetch_costs(month), compute_margin(), generate_pdf()]` |
| 新员工入职 | `[create_account, assign_groups, send_welcome_email, schedule_orientation]` |
| 论文检索综述 | `[search('topic A'), search('topic B'), summarize_each(), cross_reference()]` |
| ETL 任务 | `[extract(table), transform(rules), load(warehouse), notify(slack)]` |
| 装修流程 | `[拆改, 水电, 防水, 瓦工, 木工, 油工, 安装]` ——经典的不可逆顺序 |
| 旅行规划 | `[订机票, 订酒店, 安排接送, 列必玩清单]` |

凡是能写"清单/SOP/流程图"的任务，Planning 都比 ReAct 合适。

## 7. 进阶：Plan-and-Execute / ReAct + Re-plan 混合

实际生产里很少有纯粹的 Planning 或纯粹的 ReAct。常见的混合形式：

- **Plan-and-Execute**（这本 notebook 的样子）：先 plan，再执行。
- **ReAct + Re-plan**（更鲁棒）：先 plan，每执行 K 步检查一次是否需要 re-plan（比如某步失败、新信息推翻了原假设）。LangChain 官方的 `plan_and_execute` agent 用的就是这个变体。
- **Hierarchical Planning**：Planner 输出抽象 plan（高层目标），下面再有 sub-planner 把每个高层步骤拆成具体动作。适合很长流程。

LangGraph 写这些只需要在两个节点之间加一条"回到 planner"的条件边，本 notebook 的图是最朴素的版本。

## 8. 已知问题与改进点

- Planner 把计算步骤塞进 web_search，是工具集贫瘠的副作用——加 calculator 工具能修。
- Executor 的正则解析对 Plan 字符串格式敏感——改成结构化 `List[ToolCall]` 更稳。
- 没有 re-plan 机制——execute 失败时 plan 不会重新生成。
- evaluator 给 ReAct 打了 8 分、Planning 打了 6 分（task completion: 10 vs 8），看似 ReAct 赢了。这部分原因是 demo 任务对 ReAct 太友好（步骤少、依赖弱），加上 Planning 的搜索质量被工具贫瘠拖累——评分本身**反映的是 demo 实现而不是模式优劣**。

## 9. 本 notebook 在系列中的定位

03 ReAct 教会"让模型循环"；04 Planning 教会"让模型先想清楚再循环"。后续：

- 05 Multi-Agent / 11 Meta-Controller：把 Planner-Executor 的角色分离推到极致，每个角色变成独立 agent
- 06 PEV (Plan-Execute-Verify)：在 Planning 之上加上验证回路
- 09 Tree of Thoughts：把"单条 plan"扩展成"多分支 plan"，搜索最优路径

所以 04 是从"单 agent 多回合"走向"多 agent 协作"的桥。
