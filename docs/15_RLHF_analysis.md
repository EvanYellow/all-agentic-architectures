# 15_RLHF.ipynb 分析笔记

## 1. notebook 实际跑的是什么

标题挂着 RLHF，**但其实没有任何强化学习**。Notebook 演示的是两层"自我改进"机制：

1. **Self-Refine 循环（同一任务内的迭代）**：Generator 写一稿 → Critic 评分 + 给反馈 → Generator 拿反馈改稿 → Critic 再评 → ... 直到通过或达到最大轮数。
2. **持久化记忆（跨任务的迭代）**：把每次跑出来的"高分稿"存进一个内存 list，下次新任务时把这些好稿子作为 few-shot 示例塞进 Generator 的 prompt。

作者把第二层称为"RLHF analogy"——意思是**用 in-context learning（提示里塞例子）来模拟 RLHF 用奖励信号微调模型**这件事。两者机制完全不同，但效果方向一致：**让 agent 用过去的成功经验改善未来的输出**。

具体任务：写营销邮件。

```
请求 1: "宣传 InsightSphere（AI 数据分析平台）"
  → 第一稿 4/10（被 Critic 喷得很惨）
  → 第二稿 9/10（通过）
  → 把这封 9/10 的邮件存进 GoldStandardMemory

请求 2: "宣传 Visionary（AI CRM）"
  → 第一稿 9/10 直接通过（因为 prompt 里塞了 InsightSphere 那封做参考）
```

## 2. 架构

### 2.1 Self-Refine 循环（Phase 2）

```
            ┌─────────────┐
            │   START     │
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │  generate   │  Junior Copywriter 写第 1 版
            │ (Generator) │
            └──────┬──────┘
                   ▼
        ┌────────────────────┐
        │     critique       │  Senior Editor 打分 + 给反馈
        │     (Critic)       │  返回 Critique{score, feedback_points, is_approved}
        └─────────┬──────────┘
                  │
          should_continue?
        ┌─────────┴─────────┐
   approved 或             needs revision
   revision >= 3            (revision < 3)
        ▼                          ▼
       END                  ┌─────────────┐
                            │   revise    │  Generator + 反馈 → 改稿
                            │  (Reviser)  │
                            └──────┬──────┘
                                   │
                                   └──── 回到 critique
```

关键点：

- Generator 和 Reviser 实际上是**同一个角色，不同 prompt**——一个是"写第 1 稿"，一个是"基于反馈改稿"。两个 prompt 通过 chain factory 区分开。
- Critic 是**完全独立的 LLM 调用**，有自己的 system prompt 和评分 rubric。
- 退出条件**两个**：评分通过（`is_approved`）**或**达到最大轮数（`revision_number >= 3`）。第二个是必须的——否则 Critic 永不满意时图会无限循环。

### 2.2 持久化记忆（Phase 4）

```
GoldStandardMemory  （进程内的 list[MarketingEmail]）
        ▲
        │ add_example(approved_email)
        │
   循环跑完 ─┘

下一次任务：
   ┌─────────────────────────┐
   │ generate_node_with_memory│
   │  - 读 memory             │
   │  - 把 approved 例子拼到   │
   │    prompt 末尾           │
   │  - 让 LLM 模仿            │
   └─────────────────────────┘
```

**机制本质：把"反馈"变成"示例"**。RLHF 是用反馈调权重；这里是用反馈筛出好样本，用 in-context learning 让模型下次模仿。

## 3. 关键代码点

### 3.1 三个 Pydantic 模型把信息流锁死

```python
class MarketingEmail(BaseModel):
    subject: str
    body: str

class Critique(BaseModel):
    score: int
    feedback_points: List[str]
    is_approved: bool   # 冗余但好用：路由器直接看这个字段

class AgentState(TypedDict):
    user_request: str
    draft_email: Optional[MarketingEmail]
    critique: Optional[Critique]
    revision_number: int
```

`Critique.is_approved` 看起来冗余（score >= 8 就能算出来），但**让 LLM 自己输出布尔标志**比"路由器读 score 再判断"更稳健——因为模型自己判断时会综合 feedback 内容，不只是看分数（比如反馈里提到 "ready to send"，is_approved 就该是 true 即使打了 7 分）。这是 LangGraph 里很常见的模式：**让结构化输出把"决策"和"理由"一起返回**。

### 3.2 退出条件的双保险

```python
def should_continue(state):
    if state['critique'].is_approved:
        return "end"
    if state['revision_number'] >= 3:
        return "end"
    return "continue"
```

很重要——**没有这个上限，Self-Refine 是真的会死循环**。Critic 如果选了过严的 rubric，或者 Generator 卡在某个局部最优出不来，分数永远到不了 8。生产里通常还会加：

- 上限达到时把"最后一稿"和"最高分稿"都返回，让人决定
- 当连续两稿分数没提升时提前退出（防止无意义打转）

### 3.3 Reviser prompt 的关键设计

```python
("system", "You are the junior marketing copywriter who wrote the original draft. "
           "You have just received feedback from your senior editor. Your task is "
           "to carefully revise your draft to address every single point of feedback.")
("human", "Original Request: {request}\n\n"
          "Here is your original draft:\n**Subject:** {original_subject}\n"
          "**Body:**\n{original_body}\n\n"
          "Here is the feedback from your editor:\n{feedback}\n\n"
          "Please provide the revised email.")
```

三个东西都要塞进去：**原始任务 + 上一稿 + 反馈**。少任何一个都会出问题：

- 没有原始任务：模型可能为了讨好 Critic 偏离用户本意
- 没有上一稿：模型从零写，反馈里指代的"那个 CTA"没法对照
- 没有反馈：那这就是 generate 不是 revise

### 3.4 用 stream() 抓最后一个非终止状态

```python
final_state = None
for step in self_refine_agent.stream(initial_state):
    if END not in step:
        final_state = list(step.values())[0]
return final_state
```

LangGraph 的 `stream()` 每个 yield 是一个 `{node_name: state_update}` 字典。这段代码在循环里**记录最后一个不是 END 的状态**——绕过了 graph 在 END 时不再产 state 的细节。比直接用 `invoke()` 多了流式可观察性，但实现上稍微绕。

## 4. 一次完整执行（第一次任务）

```
revision_number=0
       │
       ▼ generate
draft_email = MarketingEmail("New Product Announcement", "Hello, We are happy...")
revision_number=1
       │
       ▼ critique
critique = Critique(
    score=4,
    feedback_points=[
        "subject is generic",
        "body too simplistic, doesn't explain value",
        "weak CTA",
        "tone is flat"
    ],
    is_approved=False
)
       │
       ▼ should_continue → "continue" (not approved, revision < 3)
       ▼ revise
draft_email = MarketingEmail(
    "Unlock Your Data's True Potential with InsightSphere",
    "Are you struggling to turn massive datasets into actionable insights? ..."
)
revision_number=2
       │
       ▼ critique
critique = Critique(score=9, feedback_points=["Excellent work...", ...], is_approved=True)
       │
       ▼ should_continue → "end"
       ▼ END
```

可以看到一个具体细节：**Critic 的反馈非常具体可执行**（"weak CTA" + "be more specific and create urgency"），所以 Reviser 能精准修改。如果 Critic 只说"不够好，再写一遍"，Reviser 就会乱改。**Self-Refine 的成败取决于 Critic 给的反馈质量**。

## 5. 第二阶段：跨任务"学习"是怎么生效的

第一次跑完后：

```python
gold_standard_memory.add_example(final_result['draft_email'])
# memory 里现在有 InsightSphere 那封 9 分邮件
```

第二次跑（新任务："宣传 Visionary CRM"）时，generator 的 prompt 变成：

```
You are a junior marketing copywriter. ... You should learn from the style 
and quality of past successful examples.

Here are some examples of high-quality emails that were approved by your editor:

Example Subject: Unlock Your Data's True Potential with InsightSphere
Example Body: Are you struggling to turn massive datasets...

Now, write a marketing email about the following topic: 
Write a promotional email for our new AI-powered CRM called 'Visionary'.
```

模型直接模仿了 InsightSphere 的结构：

| InsightSphere（参考） | Visionary（第一稿，9/10） |
|---|---|
| "Unlock Your Data's True Potential with..." | "Go From Data to Decisions, Instantly, with..." |
| "Are you struggling to turn massive datasets...?" | "Is your team drowning in data but starving for wisdom?" |
| "Stop guessing and start knowing..." | "doesn't just store customer information—it understands it..." |
| "[Request a Personalized Demo Today]" | "[Schedule a 15-Minute Live Demo]" |

**直接照着模板套**——一稿过。这就是所谓的"学到"了。

## 6. 这是不是 RLHF？严格来说不是

| 维度 | 真正的 RLHF | 本 notebook |
|---|---|---|
| 学什么 | 模型权重 | prompt context |
| 怎么学 | 梯度下降 | in-context learning |
| 持久性 | 永久（换 prompt 仍有效） | 临时（清空 memory 就忘） |
| 谁打分 | 训练好的 reward model | 一个 LLM 即时打分 |
| 用什么信号 | scalar reward + KL constraint | 通过/不通过的样本 |

**真正一致的地方**：用"高质量样本"约束未来生成方向。所以 notebook 标题里加了 "Analogy"。这件事在工程上有个更准确的名字——**Few-shot Demonstration Memory** 或者 **Episodic Prompt Memory**。

至于真正用到 RL 思想的，看 16/17 这两本（cellular_automata, reflexive_metacognitive），那里才有"模型行为根据环境信号迭代演化"的概念。

## 7. 这个 demo 模拟的现实场景

抽象出来：**一个生成任务有客观或半客观的质量标准，第一稿不行可以反复改**。

| 现实场景 | Generator | Critic |
|---|---|---|
| 写营销邮件（本 demo） | Copywriter | 总监 / 风控 |
| 写代码 | Coder | Linter / Code Reviewer / 测试 |
| 写论文摘要 | LLM | 同行评审风格的 LLM |
| 写客服回复 | LLM | 合规检查 LLM + 同情心检查 LLM |
| 写 SQL | LLM | EXPLAIN + 单元测试 |
| 翻译 | LLM | 反向翻译比对 LLM |

注意 Critic 不一定要是 LLM——能把"质量"压成可读信号的东西都行：单元测试、lint 输出、API 校验、规则引擎。这种 **LLM Generator + 程序化 Critic** 的组合在生产里其实更稳，因为 critic 不会幻觉。

## 8. Demo 暴露的 Self-Refine 局限

### 8.1 Critic 也是 LLM，会犯一致的错

Critic 用的是和 Generator **同一个 LLM**（Mixtral-8x22B），同一种偏见会同时存在于两端——比如它俩都觉得 "Be one of the first to experience the future of business intelligence" 这种 hype 很好，但实际上对于 B2B 客户可能太浮夸。这就是 RLHF 论文里反复提到的 **"reward hacking"**：模型学会了取悦 reward model，而不是真的变好。

缓解办法：

- Critic 用比 Generator **更强**的模型（让"老师"压得住"学生"）
- Critic 用**不同 family** 的模型（避免共有偏见）
- 多个 Critic 投票（ensemble）
- 引入**外部信号**做为锚（用户点击率、A/B 测试结果）

### 8.2 内存机制没有筛选和淘汰

```python
def add_example(self, email):
    self.examples.append(email)
```

来一封存一封——意味着：

- 早期只勉强通过的邮件会一直留着，拖累后期质量
- 不同行业、不同语气的邮件会混在一起，模型可能照搬不合适的风格
- list 增长无上限，token 成本越用越贵

生产做法：

- 按分数排序，只保留 top-K
- 按行业/语气标签分桶，按当前任务匹配最相关的几个
- 加 retire 机制（超过 N 天的或被新例子超过的剔除）

这就开始接近向量数据库 + RAG 的设计了。

### 8.3 没有外部 ground truth

整个系统是封闭的：Generator 写、Critic 评、再保存高分稿——但**所谓的"高分"只是 Critic 觉得高分**。如果 Critic 自己 calibration 漂了，整个 memory 就一起漂。

工业里通常会插入"真人偶尔 spot check"或"A/B 流量"作为校准信号。

## 9. 已知问题与改进点

- 用的是字符串拼接做 example 注入，没有 token 上限保护——memory 涨大后会撞上下文窗口。
- `GoldStandardMemory` 是进程内对象，进程一退就清空。生产里要持久化（pickle / sqlite / redis）。
- 没有 task 分类——所有 example 都被当成"通用范本"。
- Critic 的 rubric 写死在 prompt 里，调整时要改代码。可以做成数据驱动（rubric 配置文件）。
- 评分用整数 1-10，**粒度太粗**——8/9/10 的差异 LLM 很难稳定区分，导致评分抖动。改成 5 个维度各 1-5 分会更稳。

## 10. 本 notebook 在系列中的定位

01 reflection 是这本的"近亲"：reflection 让 agent 反思自己的输出。15 把 reflection 升级成两件事：

- **结构化反思**（Critic 是独立 LLM，rubric 明确）
- **跨任务持久化**（反思不仅改这一稿，还塞进未来）

后面：

- 16 cellular_automata：把"群体反复演化"做成空间结构
- 17 reflexive_metacognitive：让 agent 自己监督自己（Critic 也是同一个 agent 的不同 mode）

可以把 15 看作**第一个明确具备"长期改进"机制的架构**，前面 01-14 的 agent 跑完一次就忘。从这本开始，"learning system" 才有了形式上的载体——尽管这个载体只是 in-context examples，远不到训练权重。
