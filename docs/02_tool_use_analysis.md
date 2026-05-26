# 02_tool_use.ipynb 分析笔记

## 1. notebook 实际跑的是什么

把一个 **LLM + 一个 web 搜索工具** 接到一起，演示"工具使用（Tool Use）"这个最基本的 agentic 模式：模型自己判断该不该调工具、调哪个、传什么参数，然后把工具返回值塞回上下文，输出最终答案。

测试用例：

> What were the main announcements from Apple's latest WWDC event?

这是一个 LLM 训练数据里没有的"现时信息"问题，必须借助外部工具才能回答——所以是一个**最干净的工具触发场景**。

## 2. 架构（单跳工具调用）

```
            ┌─────────────┐
            │   START     │
            └──────┬──────┘
                   │  user query
                   ▼
            ┌─────────────┐
            │   agent     │   LLM + bind_tools([web_search])
            │   (LLM)     │
            └──────┬──────┘
                   │
            router_function
                   │
        ┌──────────┴──────────┐
        │ has tool_calls?     │
        ▼ yes                 ▼ no
 ┌─────────────┐       ┌─────────────┐
 │  call_tool  │       │     END     │
 │  ToolNode   │       └─────────────┘
 │  (Tavily)   │
 └──────┬──────┘
        │ tool result
        ▼
 ┌─────────────┐
 │   agent     │   ← 关键：tool 输出回到 agent 做合成
 └─────────────┘
        │
        ▼
       END
```

代码里关键的两条边：

```python
graph_builder.add_conditional_edges("agent", router_function)
graph_builder.add_edge("call_tool", "agent")   # 让模型能"看"到工具结果再答题
```

## 3. 关键代码点

### 3.1 工具的"描述"决定 LLM 是否会调它

```python
search_tool.name = "web_search"
search_tool.description = (
    "A tool that can be used to search the internet for up-to-date information "
    "on any topic, including news, events, and current affairs."
)
```

LLM 选不选这个工具，**完全靠这段自然语言描述**。如果描述含糊（比如只写 "search"），模型可能在简单问候时也调用它；如果太狭窄（比如 "weather lookup"），又会错过其他场景。这一行实际上就是工具的"提示词"。

### 3.2 `bind_tools` 把工具元信息塞进 system prompt

```python
llm_with_tools = llm.bind_tools(tools)
```

LangChain 在底层把 `tools` 列表里每个工具的 name/description/参数 schema，按 OpenAI function calling 的格式注入到模型请求里。模型如果要调工具，会输出一个带 `tool_calls` 字段的 AIMessage。

### 3.3 Router：模型说调就调，没说就结束

```python
def router_function(state):
    last = state["messages"][-1]
    if last.tool_calls:
        return "call_tool"
    return "__end__"
```

这是 ReAct/Tool Use 这一类 agent 共用的最小判定器。判断依据**只看最后一条 AIMessage 上有没有 tool_calls**。

## 4. 一次完整执行的消息流

```
[HumanMsg]   user: "What were the main announcements from Apple's latest WWDC..."
     │
     ▼
[AIMsg + tool_calls]   web_search(query="Apple WWDC latest announcements")
     │
     ▼
[ToolMsg]   {results: [...MacRumors, Apple Developer...]}
     │
     ▼
[AIMsg]   "The main announcements from Apple's latest WWDC event include..."
     │
     ▼
   END
```

注意这里整个任务**只调了一次工具**——不是因为图里限制了一次，而是因为这个问题一次搜索就够了。如果换成多跳问题（见 `03_ReAct.ipynb`），同一张图会自然循环多次。

> 也就是说：**这张图既是 Tool Use agent，也是 ReAct agent**。02 和 03 两本 notebook 的图结构其实一样，差别只在"题目难度"——02 用单跳问题让你看懂工具调用机制，03 用多跳问题让你看出循环的必要性。

## 5. 这个 demo 模拟的现实场景

抽象出来就是一类任务：**模型本身知识不够，但给定合适的工具，它能自己决定何时去查**。

| 现实场景 | tools 列表 |
|---|---|
| 实时新闻问答 | `web_search` |
| 股价 / 汇率查询 | `get_quote(ticker)` |
| 客服查订单 | `query_order(order_id)`, `query_user(user_id)` |
| 电商助理 | `search_product`, `compare_price`, `add_to_cart` |
| 数据分析助理 | `run_sql`, `read_csv`, `plot` |
| 计算助理 | `calculator`, `wolfram_alpha` |

工具集变了，**这张图一行都不用改**。这才是 Tool Use 模式的核心价值。

## 6. 评估部分（LLM-as-Judge）

notebook 用 `with_structured_output(ToolUseEvaluation)` 把判分模型的输出限制成一个 Pydantic 三元组：

- `tool_selection_score`：工具选得对不对（这个 demo 里只有一个工具，分数意义不大）
- `tool_input_score`：传给工具的参数（query 字符串）质量
- `synthesis_quality_score`：拿到工具输出后，模型最终答案有没有把信息整合好

这是判断 agent 工具使用质量的常用切片：**选对工具 / 调对参数 / 用好结果**，三个步骤任何一步出问题都会拖累整体效果。

## 7. 已知问题与改进点

- 用的是已弃用的 `langchain_community.tools.tavily_search.TavilySearchResults`，会有 deprecation warning。建议迁到 `langchain_tavily.TavilySearch`（在 `04_planning.ipynb` 里已经用上了新版）。
- 没有错误处理：Tavily 限流、API 报错、返回空结果时 agent 会拿到一个错误字符串，可能直接幻觉。生产中工具应包一层 try/except，把错误分类后回灌给模型。
- 没有 token 上限保护：搜索结果是整段网页 content，如果 `max_results` 调高，很容易把上下文撑爆。
- pygraphviz 没装，所以 `draw_png()` 会失败——这是可视化的可选项，不影响功能。

## 8. 本 notebook 在系列中的定位

这是整个 agentic 系列的**奠基式**——后面所有引入"工具"的 notebook（03 ReAct、04 Planning、11 Meta-Controller、15 Self-Refine 等等）都默认你已经理解了：

1. 怎么定义工具
2. 怎么把工具绑给 LLM
3. 怎么用 router + ToolNode 在 LangGraph 里编排"think → act → observe"

后续 notebook 的复杂度全部体现在**节点结构**上（多 agent、多步骤、带反馈），但**调工具这件事的机制**从这里就定型了。
