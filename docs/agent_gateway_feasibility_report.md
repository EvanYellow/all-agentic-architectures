# Agent Gateway 与范式路由小模型可行性分析报告

## 0. 结论摘要

本项目具备推进价值。建议立项方向不是"再做一个 agent 框架"，而是构建一个 **Agent Gateway Policy Layer**：在请求入口用低开销模型和检索系统快速判断应该触发哪一种 agent 范式、哪组 skill、哪档执行模型和多少执行预算。

核心判断：

- **技术上可行**：现有仓库已经把 Tool Use、ReAct、Planning、Meta-Controller、Self-Refine 拆成了可 benchmark 的独立范式；这正好可以升级成 gateway 可选择的 execution modes。
- **研究上有新意**：已有工作主要集中在 tool/skill retrieval、tool selection、workflow generation 或 utility orchestration。本方向的差异是把 **agentic paradigm selection** 本身作为一等对象，让小模型做"范式路由 + skill 路由 + 预算路由"。
- **训练小模型有必要**：大模型只看 skill 名称和描述做选择并不可靠。最新 SkillRouter 结果显示，隐藏 skill body 会造成 29-44 个百分点的准确率下降，full-text skill body 是决定性信号；SkillRet 也显示 task-specific fine-tuning 对 skill retrieval 有显著增益。
- **经济上有机会**：如果 gateway 能避免把简单请求送入 ReAct/Planning/Multi-Agent 这类重流程，就能节省模型调用、tool 调用和上下文 token。但如果所有请求无条件先跑一个小模型，简单请求反而会被加延迟。因此必须设计成分层 fast path。
- **主要风险**：小模型范式选择准确率、训练标签质量、wrong-route 成本、gateway 自身延迟、full-text skill 的安全边界、论文 novelty 与现有 orchestration 工作的区分。

推荐立项目标：

> 构建并验证一个本地 Agent Gateway Router：它基于请求、skill full text、范式说明、历史执行轨迹和成本画像，选择 direct / RAG / single-tool / ReAct / plan-execute / multi-agent / self-refine / clarify 等执行模式，在保持端到端成功率的前提下降低平均延迟和成本。

Go/No-Go 建议：

- 第一阶段目标：不训练模型，先用规则、embedding、large LLM teacher 和现有 benchmark case 搭出离线评估集。
- 第二阶段目标：训练或微调 0.6B-1.5B 级别的 retriever/reranker/policy model，验证 full-text + paradigm routing 是否显著优于 name/description routing 和 large LLM meta-controller。
- 第三阶段目标：在 CPU AMX 机器上跑本地推理，测 p50/p99、吞吐、CPU 利用率、cost/request 和 wrong-route recovery。

## 1. 背景与现有基础

当前 `docs/benchmark_plan.md` 已经明确提出：随着 agent 应用增多，编排层可能从业务代码中独立出来，成为对接多种 LLM 后端和工具集群的中间层。现有 benchmark 目标很窄，只测 LangGraph 自身开销，例如 `StateGraph` 调度、节点包装、`add_messages` reducer、conditional edges、`ToolNode`、`with_structured_output` 和 `bind_tools` wrapper。

这个现有设计有两个重要特点：

1. 它将 LLM 推理和外部 tool I/O 排除在纯编排 benchmark 之外，因此能干净测量 orchestration overhead。
2. 它已经覆盖多个 agent 范式：
   - 02 Tool Use：单工具、单跳或少量工具调用。
   - 03 ReAct：多轮 reason-act-observe 闭环。
   - 04 Planning：先计划再执行，适合可提前拆分的任务。
   - 11 Meta-Controller：入口路由到不同专家 agent。
   - 15 Self-Refine/RLHF analogy：生成、批评、修订循环。

这些 case 目前是 benchmark 对象。新的研究方向是把它们升级成 gateway 可以选择的 execution modes。

现有文档中还有几个直接支持本方向的观察：

- ReAct 的价值在于"循环 + 工具反馈"，但它也会引入更多 LLM 调用和死循环风险。
- Planning 在可提前分解、依赖弱的任务上比 ReAct 更高效，但在环境变化时僵硬。
- Meta-Controller 是最常见的统一入口形态，但每次多一次 LLM 调用，且在模糊请求上容易错路由。
- 当前 03 case 早期实测显示，纯编排开销约 5-7 ms。若 gateway 增加本地小模型推理，它会成为新的显著延迟项。因此 gateway 必须带来更大的下游节省，才有经济意义。

## 2. 问题定义

生产中的 agent 系统常见问题不是"没有 ReAct"或"没有 planner"，而是 **所有请求都被过度 agent 化**。

典型浪费包括：

- 简单问答进入 ReAct，产生多次 LLM 调用。
- 只需一次 tool 的任务被 planner 拆成多步。
- 可直接 RAG 的任务被交给 multi-agent。
- skill catalog 过大时，把大量工具描述塞进上下文，增加 token 成本并干扰模型选择。
- 大模型 meta-controller 为每次请求做路由，增加固定延迟和成本。

因此真正需要优化的是入口策略：

> 给定一个用户请求、上下文摘要、skill registry、agent 范式库、成本画像和安全策略，gateway 应该选择最低成本且足够可靠的执行路径。

这个问题可以形式化为 utility routing：

```text
decision = argmax U(success, latency, cost, risk, recoverability)
```

其中 decision 至少包含：

- `paradigm`: direct / rag / single_tool / react / plan_execute / multi_agent / self_refine / clarify / refuse。
- `skills`: 候选 skill 或 tool 集合。
- `model_tier`: small / medium / large / code / vision 等模型档位。
- `budget`: max_steps、timeout、token budget、tool budget。
- `confidence`: 路由置信度。
- `fallback`: 低置信度或失败后的升级路径。

## 3. 提议方案

### 3.1 总体架构

```text
User Request
  -> deterministic fast gates
  -> query normalization / context summary
  -> full-text skill and paradigm retrieval
  -> local policy router / reranker
  -> budget and fallback decision
  -> selected executor
  -> trace collection and outcome feedback
```

执行器可以是：

- Direct answer executor。
- RAG executor。
- Single-tool executor。
- ReAct executor。
- Plan-and-execute executor。
- Multi-agent executor。
- Self-refine / verifier executor。
- Clarification or human handoff executor。

关键原则：

- Gateway 小模型只做 **policy decision**，不承担最终复杂任务。
- 复杂执行仍交给对应 agent 或大模型。
- 对明显请求先走 deterministic fast gates，避免无意义调用小模型。
- skill 内容不一次性全塞 prompt，而是通过检索取 top-k，再精排。
- 每个 paradigm 也有 full-text "paradigm card"，包括适用条件、禁用条件、典型失败、成本画像、示例轨迹。

### 3.2 Router 输出协议

建议使用严格 JSON schema 或 constrained decoding：

```json
{
  "paradigm": "react",
  "agent": "research_agent",
  "skill_candidates": ["lookup_company", "search_news"],
  "model_tier": "medium",
  "max_steps": 4,
  "timeout_ms": 8000,
  "confidence": 0.82,
  "needs_clarification": false,
  "fallback": "large_router"
}
```

不要让 router 输出长 reasoning。若需要审计，可以输出短 `decision_factors`，例如：

```json
{
  "decision_factors": ["requires_fresh_info", "dependent_steps"]
}
```

### 3.3 范式选择策略

建议把范式选择变成可学习分类 + 排序问题：

| 请求特征 | 推荐范式 |
|---|---|
| 简单知识、改写、翻译、闲聊 | direct |
| 需要私有知识或文档依据 | rag |
| 明确只需一个外部动作 | single_tool |
| 下一步依赖上一步 observation | react |
| 多个独立子任务，可提前列计划 | plan_execute |
| 跨多个专业角色或权限域 | multi_agent |
| 输出质量要求高，且可迭代改进 | self_refine |
| 槽位缺失、风险高、需求含混 | clarify / human_handoff |
| 违规或不可执行 | refuse |

### 3.4 Skill 与 Paradigm 的统一表示

每个 skill 保存：

- name
- short description
- full implementation body 或 SKILL.md 全文
- tool schema
- examples
- preconditions
- side effects
- permissions
- failure modes
- historical success/failure traces
- cost profile

每个 paradigm 保存：

- name
- when_to_use
- when_not_to_use
- required_executor
- typical_latency
- typical_model_calls
- typical_tool_calls
- max_risk_level
- examples
- counterexamples
- fallback_policy

这样 gateway 可以同时做：

- skill routing
- paradigm routing
- budget routing
- model-tier routing
- safety routing

## 4. 外部依据与相关工作

### 4.1 SkillRouter

SkillRouter: Retrieve-and-Rerank Skill Selection for LLM Agents at Scale 明确指出，当前很多 agent 系统只把 skill name 和 description 暴露给 agent，而隐藏完整实现体。这种 progressive disclosure 假设 metadata 足够做选择。论文的关键发现是：skill body 是决定性信号，移除 body 会造成 29-44 个百分点的准确率下降；他们提出的 0.6B encoder + 0.6B reranker 两阶段系统达到 74.0% top-1 routing accuracy，并可部署在消费级硬件上。

对本项目的含义：

- full-text skill body 应进入 gateway 检索和精排。
- 训练小模型是合理路线，不应只依赖大模型读短描述。
- 可以借鉴两阶段 retrieve-and-rerank：embedding recall + cross-encoder/reranker。

### 4.2 SkillRet

SkillRet 提供大规模 skill retrieval benchmark，包含 17,810 个公开 agent skills、两级 taxonomy、4,997 evaluation queries 和 63,259 training queries。其结果显示，task-specific fine-tuning 对 skill retrieval 有显著提升，NDCG@10 比强 prior retriever 提升 13.1 点，比强 off-the-shelf retriever 提升 16.9 点。

对本项目的含义：

- 不训练的小模型或通用 embedding 可能不够。
- 如果目标是论文和生产系统，需要构造本领域 task-specific 数据集。
- 数据生成和标签质量会成为核心资产。

### 4.3 AutoTool

AutoTool 针对 ReAct 等 agent 中重复调用 LLM 进行 tool selection 的高成本问题，使用历史 agent trajectory 构建 tool transition graph，利用工具选择惯性降低推理成本。

对本项目的含义：

- agent 的 step-level 决策存在可学习的统计结构。
- Gateway 不必只做入口路由，也可以维护 transition prior 和 recovery prior。
- 但本项目应避免变成纯 tool transition graph；更大的差异是选择 agentic paradigm。

### 4.4 Utility-Guided Agent Orchestration

Utility-Guided Agent Orchestration 将 orchestration 建模为决策问题，在 respond、retrieve、tool call、verify、stop 等动作间做收益成本权衡。

对本项目的含义：

- "成本收益驱动的 agent orchestration"是合理研究问题。
- 本项目可进一步把 action space 提升到 paradigm-level，并强调本地小模型和 full-text skill/paradigm context。

### 4.5 FlowSteer 与 workflow 生成

FlowSteer 一类工作关注自动构建或控制 workflow graph，通常更偏重生成完整 workflow 或通过强化学习优化流程。

对本项目的含义：

- 本项目不应主打"自动生成任意 workflow graph"，否则会和这类工作重叠且评估难度更高。
- 更好的定位是：在一组已定义、可审计、可执行的 agent 范式之间做低成本选择。

### 4.6 Agent Skills 安全与治理

近期 Agent Skills 综述和安全论文都强调：skill 不只是 prompt 片段，它涉及权限、来源、生命周期和安全边界。另有工作指出社区贡献 skill 存在漏洞和治理风险。

对本项目的含义：

- full-text skill routing 不能简单把 skill body 当普通文档。skill body 可能包含恶意指令、过宽权限或 prompt injection。
- Gateway 需要把 permission tier、provenance、security scan result 纳入路由特征。
- 安全不是附加模块，而是 gateway policy 的一部分。

### 4.7 CPU AMX 与本地推理

Intel AMX 是面向矩阵运算的 CPU 内置加速能力，官方资料说明其支持 BF16 和 INT8 数据类型，覆盖训练和推理场景。OpenVINO CPU 文档也说明，在 4th Gen 及更新 Xeon 上，bfloat16 和 float16 可启用 AMX，相比 AVX512/AVX2 加速多种深度学习算子。

对本项目的含义：

- CPU AMX 适合部署 0.6B-1.5B 级别 router/reranker，尤其是短输入、短输出、批量或高并发场景。
- INT4 权重量化可降低内存和带宽压力，但不等于 AMX 原生 INT4 计算；具体性能取决于 OpenVINO/IPEX/oneDNN/llama.cpp 等 runtime kernel。
- 必须做实测，不能只按模型参数量推断延迟。

## 5. 技术可行性

### 5.1 可训练任务拆解

建议不要训练一个"全能 gateway LLM"。应拆成多个可评估子任务：

1. **Paradigm classification**
   - 输入：用户请求、上下文摘要、可选 top-k skill/paradigm cards。
   - 输出：direct / rag / single_tool / react / plan_execute / multi_agent / self_refine / clarify / refuse。

2. **Skill retrieval**
   - 输入：用户请求、任务上下文。
   - 输出：top-k skills。
   - 训练方式：dual-encoder contrastive learning。

3. **Skill reranking**
   - 输入：query + candidate skill full text。
   - 输出：relevance score。
   - 训练方式：pairwise/listwise ranking 或 cross-entropy。

4. **Budget prediction**
   - 输入：query + selected paradigm。
   - 输出：max_steps、model_tier、timeout、tool_budget。
   - 训练方式：分类或回归。

5. **Fallback and confidence calibration**
   - 输入：router logits、retrieval margin、query features。
   - 输出：confidence、是否升级到 large router。
   - 训练方式：calibration + threshold tuning。

6. **Safety and permission routing**
   - 输入：query + skill permission/provenance。
   - 输出：allow / deny / ask_approval / human_handoff。
   - 训练方式：分类 + policy rules。

### 5.2 模型形态

建议比较三条路线：

| 路线 | 说明 | 优点 | 风险 |
|---|---|---|---|
| Encoder + Reranker | 0.3B-0.8B embedding model + 0.3B-0.8B reranker | 与 SkillRouter/SkillRet 方向一致，适合 full-text retrieval | 不擅长生成复杂 JSON，需要另一个 policy head |
| Small generative policy model | 0.5B-1.5B instruction model，输出 constrained JSON | 一套模型能做范式、预算、fallback | 推理延迟更高，校准难 |
| Hybrid | retrieval/rerank 做 skill，generative/classifier 做 paradigm/budget | 工程上最稳 | 系统复杂度更高 |

推荐 MVP 采用 Hybrid：

```text
BM25 / embedding recall
  -> reranker scores query-skill pairs
  -> small policy model/classifier chooses paradigm and budget
  -> confidence gate decides fallback
```

### 5.3 训练数据来源

建议数据分三类：

1. **Synthetic task generation**
   - 为每个 skill/paradigm 生成正例、难负例和边界例。
   - 用 large teacher 标注 candidate paradigm、skill、budget、fallback。
   - 人工抽检高风险和边界样本。

2. **Benchmark traces**
   - 复用现有 02/03/04/11/15 benchmark。
   - 每条 query 附带 expected paradigm、expected tool calls、expected hop count。
   - 运行不同 executor 得到真实 latency/cost/success，形成 utility label。

3. **Production-like traces**
   - 如果能获取真实请求，做匿名化后标注。
   - 最有价值的是 wrong-route、fallback、用户纠正、执行失败 trace。

初始规模建议：

- Paradigm routing：5k-20k labeled queries 可启动 MVP。
- Skill retrieval：每个 skill 20-100 条合成 query，至少覆盖 positive、hard negative、permission negative。
- End-to-end eval：至少 1k 条固定评估集，不能和训练 query 重合。
- 人工审阅：优先覆盖高频 skill、边界范式、权限敏感任务。

### 5.4 训练方法

推荐顺序：

1. **不训练 baseline**
   - rules
   - BM25
   - off-the-shelf embedding
   - large LLM router
   - name+description router

2. **Retriever fine-tuning**
   - contrastive learning
   - in-batch negatives
   - hard negative mining

3. **Reranker fine-tuning**
   - pairwise/listwise ranking
   - full-text skill/paradigm cards as input

4. **Policy model SFT**
   - constrained JSON output
   - labels from teacher + execution outcome

5. **Calibration**
   - temperature scaling
   - threshold tuning per paradigm
   - reject option for low confidence

6. **Optional preference optimization**
   - 用执行结果构造 preference：成功且低成本的路线优于成功但昂贵的路线，失败路线最低。
   - 先不建议上复杂 RL，避免项目早期不稳定。

LoRA/QLoRA 可作为训练效率方案。QLoRA 通过冻结 4-bit 量化基座并训练低秩 adapter 来降低显存压力，适合早期资源有限时进行实验。

### 5.5 推理部署

建议 gateway runtime：

```text
Process A: Gateway API and deterministic gates
Process B: Embedding/retrieval service
Process C: Reranker/policy model service
Process D: Executor runtime
```

部署原则：

- 模型常驻内存，不按请求加载。
- 对 router 输出使用 constrained decoding 或分类头，避免长生成。
- 用 top-k retrieval 控制 full-text 输入长度。
- gateway 与 executor trace 全链路记录。
- CPU 上固定线程池和 core affinity，避免与业务服务抢资源。
- 同时测 single-request latency 和 batch throughput。

### 5.6 CPU AMX 可行性边界

AMX 的价值主要在 BF16/INT8 矩阵乘法。对 1B 左右模型，性能瓶颈可能来自：

- batch=1 的内存带宽和权重读取。
- prompt prefill 长度。
- reranker 对多个 candidate skill 的重复打分。
- JSON constrained decoding 的 runtime 实现。

因此不能简单说"1B + AMX 一定很快"。需要测：

- p50/p99 gateway latency。
- QPS under fixed CPU core budget。
- model load memory。
- token prefill/decode breakdown。
- retrieval/rerank/policy 各阶段耗时。
- AMX kernel 是否实际被 runtime 使用。

## 6. 评估方案

### 6.1 Offline Routing Evaluation

指标：

- Paradigm top-1 accuracy。
- Paradigm top-2 accuracy。
- Skill Recall@K。
- Skill MRR / NDCG@K。
- Reranker top-1 accuracy。
- Confidence calibration: ECE、coverage-accuracy curve。
- Safety false negative rate。
- Fallback precision/recall。

必须包含 ablation：

- name only
- name + description
- name + description + examples
- full skill body
- full skill body + historical traces
- full skill body + paradigm cards

### 6.2 End-to-End Evaluation

只看 router 准确率不够。最终要比较：

- task success
- answer quality
- latency p50/p90/p99
- model calls/request
- tool calls/request
- tokens/request
- cost/request
- recovery/fallback rate
- wrong-route penalty
- user-visible failures

建议 baseline：

| Baseline | 用途 |
|---|---|
| Fixed Direct | 测最低成本但能力不足 |
| Fixed ReAct | 测过度 agent 化成本 |
| Fixed Planning | 测计划式流程优劣 |
| Large LLM Meta-Controller | 测强但贵的路由器 |
| Rule Router | 测低成本规则上限 |
| Embedding Skill Router | 测通用检索路线 |
| Name+Description Router | 对照 progressive disclosure |
| Full-Text SkillRouter-like | 对照 full-text skill routing |
| Proposed Gateway | 目标系统 |

### 6.3 Utility Evaluation

定义统一 utility：

```text
U = success_score
    - alpha * latency_ms
    - beta * model_cost
    - gamma * tool_cost
    - delta * risk_penalty
    - eta * wrong_route_recovery_cost
```

论文里可以报告多个 alpha/beta/gamma 配置，展示在不同业务偏好下的 Pareto frontier。

### 6.4 Go/No-Go 阈值

第一版建议阈值：

- Paradigm top-1 >= 80%，top-2 >= 95%。
- Skill Recall@5 >= 90%。
- Full-text reranker top-1 比 name+description 至少提升 10 个百分点。
- Proposed Gateway 端到端成功率与 large LLM meta-controller 相差不超过 3-5 个百分点。
- 平均 cost/request 比 Fixed ReAct 降低 >= 25%。
- p50 gateway decision latency 初始目标 < 100 ms；在目标硬件实测后再收紧。
- low-confidence fallback 覆盖主要 wrong-route case，且 fallback 后成功率显著高于不 fallback。

这些阈值不是最终论文指标，而是资源继续投入的工程判断线。

## 7. 经济可行性

### 7.1 成本模型

gateway 是否省钱取决于：

```text
E[cost] =
  C_gateway
  + P(direct) * C_direct
  + P(rag) * C_rag
  + P(single_tool) * C_single_tool
  + P(react) * C_react
  + P(plan) * C_plan
  + P(fallback) * C_large_router
  + P(wrong_route) * C_recovery
```

它省钱的条件：

```text
C_gateway + C_selected_executor + C_error
  < C_fixed_heavy_agent
```

因此：

- 如果系统当前所有请求都进 ReAct/Multi-Agent，gateway 大概率省钱。
- 如果大量请求本来就能规则路由，gateway 必须先用 deterministic gates，不应无条件跑小模型。
- 如果 wrong-route 导致二次执行或用户失败，省下的成本会被错误代价抵消。

### 7.2 成本收益来源

潜在收益：

- 少调用大模型 meta-controller。
- 少把简单请求送入多步 agent loop。
- 少暴露全量工具描述，降低 prompt token。
- 减少不必要 tool call。
- 更早识别需要澄清的请求，避免无效执行。
- 把路由负载从 GPU/API 转移到本地 CPU。

新增成本：

- 小模型训练和维护。
- skill/paradigm 数据集建设。
- CPU gateway 推理资源。
- 评估和观测系统。
- 安全扫描和治理成本。
- fallback 逻辑复杂度。

### 7.3 资源申请建议

最小可行资源：

- 1 名 research/tech lead。
- 1 名 ML engineer，负责数据、训练、评估。
- 1 名 systems engineer，负责 gateway、runtime、benchmark。
- 1 台 CPU AMX 测试机器。
- 1-2 张训练 GPU 或等价云资源，用于 LoRA/QLoRA 和 reranker 训练。

推荐资源：

- 1 名 research lead。
- 2 名 ML engineers。
- 1 名 infra/systems engineer。
- 1 名 security/eval engineer。
- CPU AMX 测试环境 + GPU 训练环境 + trace storage。
- 小规模人工标注预算，用于边界样本和高风险样本。

## 8. 论文可行性

### 8.1 建议题目

可选题目：

- ParadigmRouter: Low-Latency Routing over Agentic Execution Modes
- Agent Gateway: Full-Context Policy Routing for Efficient LLM Agents
- Beyond Tool Routing: Local Paradigm Selection for Cost-Efficient Agent Systems

### 8.2 核心贡献

建议论文贡献写成四点：

1. **Paradigm-as-a-Callable abstraction**
   - 将 direct、RAG、single-tool、ReAct、planning、multi-agent、self-refine 等 agentic patterns 统一为可调用 execution modes。

2. **Full-context policy routing**
   - 不只基于 name/description，而是利用 skill body、paradigm cards、historical traces、cost profile 和 safety metadata 做路由。

3. **Compact local router**
   - 训练本地小模型或 hybrid retriever/reranker/policy stack，在 CPU 上低成本推理，减少大模型 meta-controller 依赖。

4. **Cost-quality benchmark**
   - 构造任务集和评估协议，衡量 paradigm accuracy、task success、latency、cost、wrong-route penalty 和 fallback behavior。

### 8.3 与已有工作的区分

| 工作方向 | 已有重点 | 本项目差异 |
|---|---|---|
| SkillRouter / SkillRet | 选 skill | 同时选择 paradigm、skill、budget、model tier |
| AutoTool | ReAct 内部 tool selection | 在入口决定是否需要 ReAct 以及用哪种范式 |
| Utility orchestration | respond/retrieve/tool/verify/stop 动作选择 | 提升到范式级 execution policy，并结合 full-text skill/paradigm context |
| Workflow generation | 生成或优化完整 workflow | 选择预定义、可审计、可执行的范式，工程风险更低 |
| Meta-controller | 大模型路由到专家 | 本地小模型 + full context + 成本约束 + fallback |

### 8.4 论文风险

- 如果只实现框架，没有强 benchmark 和 ablation，论文价值不足。
- 如果只做 skill routing，会被 SkillRouter/SkillRet 覆盖。
- 如果只做 LLM meta-controller，会被现有 agent orchestration 覆盖。
- 如果训练小模型但没有端到端 cost-quality 结果，容易被认为只是工程优化。

论文必须证明：

```text
Paradigm routing changes the cost-quality frontier.
```

即：

- 比 fixed ReAct 更便宜。
- 比 fixed direct 更准。
- 比 large LLM router 更快。
- 比 name/description routing 更稳。
- 比 pure skill routing 更能避免错误范式。

## 9. 分阶段实施计划

### Phase 0: 立项准备，1 周

交付：

- 确认范式集合。
- 定义 router output schema。
- 定义 benchmark labels。
- 确认硬件和训练资源。

决策点：

- 先做 hybrid router，而不是单一生成模型。
- 先做离线评估，再做生产集成。

### Phase 1: Benchmark 与 Teacher Baseline，2-3 周

交付：

- 扩展现有 benchmark query pool，增加 paradigm label。
- 为每个 query 标注 expected paradigm、acceptable fallback、skill candidates、budget。
- 实现 baselines：
  - rule router
  - embedding router
  - large LLM router
  - name+description router
  - full-text router

验收：

- 形成 1k+ 条固定 eval set。
- 能跑出 offline routing metrics 和 end-to-end metrics。

### Phase 2: Data Generation 与 Retriever/Reranker，3-4 周

交付：

- skill/paradigm cards。
- synthetic query generation pipeline。
- hard negative mining。
- fine-tuned retriever。
- fine-tuned reranker。

验收：

- Skill Recall@5 >= 90% 或显著超过 off-the-shelf embedding。
- Full-text reranker 明显优于 name+description。

### Phase 3: Policy Model 与 Gateway MVP，3-4 周

交付：

- paradigm classifier 或 small generative policy model。
- budget/model-tier predictor。
- confidence/fallback module。
- gateway service MVP。
- 结构化 trace 输出。

验收：

- Paradigm top-1 >= 80%，top-2 >= 95%。
- End-to-end cost/request 比 fixed ReAct 降低 >= 25%。
- 与 large LLM meta-controller 成功率差距 <= 3-5 个百分点。

### Phase 4: CPU AMX 优化与系统测量，2-3 周

交付：

- OpenVINO/IPEX/oneDNN 或其它 runtime 对比。
- BF16/INT8/INT4 storage 方案对比。
- p50/p99 latency、QPS、CPU utilization、memory。
- emon/VTune 或等价 profile。

验收：

- 明确每个 gateway stage 延迟占比。
- 明确是否真正触发 AMX kernel。
- 给出 production sizing 建议。

### Phase 5: 论文实验与写作，3-5 周

交付：

- 完整 ablation。
- cost-quality Pareto frontier。
- failure analysis。
- 安全与 fallback 分析。
- 论文初稿。

验收：

- 至少一个清晰、可复现、可解释的主表。
- 至少一个端到端系统图。
- 至少一个 wrong-route/fallback 分析表。

## 10. 风险与缓解

### 10.1 小模型准确率不足

风险：

- 1B 模型可能无法处理复杂、模糊、多意图请求。

缓解：

- 使用 top-2 paradigm + fallback。
- 对低置信度请求升级到 large router。
- 使用 hard negatives 和真实失败 trace 训练。
- 不让小模型直接执行复杂任务。

### 10.2 数据标签质量不足

风险：

- Teacher LLM 标签可能偏向自己的推理风格。
- Synthetic query 可能过于模板化。

缓解：

- 用 execution outcome 反标 utility。
- 人工审阅边界样本。
- 引入真实请求和失败 trace。
- 保持固定 blind eval set。

### 10.3 Gateway 反而增加延迟

风险：

- 对简单请求，小模型推理成本可能超过收益。

缓解：

- deterministic fast gates 放在最前面。
- 缓存高频路由结果。
- 用 classifier head 或 constrained short output。
- 对 obvious direct 请求绕过 router。

### 10.4 Wrong-route 成本高

风险：

- 错误范式导致多次执行、用户失败或安全问题。

缓解：

- 设计 fallback 和 recovery。
- 记录 wrong-route penalty。
- 对高风险范式要求更高 confidence。
- 对权限敏感 skill 强制 human approval。

### 10.5 Full-text skill 安全风险

风险：

- skill body 可能包含恶意指令、过宽权限、prompt injection 或隐藏行为。

缓解：

- skill provenance、hash、permission tier。
- static scan + runtime policy。
- full-text routing 时把 skill body 作为数据，不作为可执行指令。
- 高权限 skill 不仅看语义相关，还看授权上下文。

### 10.6 与已有工作 novelty 重叠

风险：

- 被认为只是 skill routing 或 tool routing。

缓解：

- 主任务必须是 paradigm-level routing。
- 报告端到端 cost-quality frontier。
- 和 SkillRouter/SkillRet/AutoTool/large meta-controller 都做 baseline。

## 11. 建议资源投入

### 11.1 人员

最低配置：

- Tech lead / research lead：1 人。
- ML engineer：1 人。
- Systems engineer：1 人。

推荐配置：

- Research lead：1 人。
- ML engineer：2 人。
- Systems engineer：1 人。
- Eval/security engineer：1 人。

### 11.2 计算资源

最低配置：

- 1 台 CPU AMX 测试机。
- 1-2 张中高端 GPU 或云训练资源，用于 LoRA/QLoRA、embedding/reranker fine-tuning。
- 向量检索和 trace 存储环境。

推荐配置：

- 独立 CPU AMX benchmark 机器，避免业务噪声。
- 多 runtime 对比环境：OpenVINO、IPEX/oneDNN、llama.cpp 或其它本地推理栈。
- 固定评估环境，支持重复跑 p50/p99 和 power/perf 观测。

### 11.3 数据资源

需要：

- skill registry 全文。
- skill schema、权限、示例、失败模式。
- paradigm cards。
- synthetic query generation pipeline。
- teacher labels。
- execution traces。
- 人工审核样本。

## 12. 近期最小可行动作

建议马上做三件事：

1. **把 benchmark case 升级为范式标签集**
   - 给 02/03/04/11/15 的 query pool 标注 expected paradigm。
   - 增加 direct、RAG、clarify、refuse 等对照任务。

2. **实现不训练的 gateway baseline**
   - rule router
   - embedding router
   - large LLM router
   - name+description router
   - full-text router

3. **定义 skill/paradigm card 格式**
   - 统一 name、description、body、when_to_use、when_not_to_use、cost、permissions、examples。

完成这三件事后，就能判断训练小模型是否有足够提升空间，而不是先盲目训练。

## 13. 总体判断

本项目值得投入更多资源，但要以 **benchmark-first** 和 **utility-first** 的方式推进。

最强的资源申请理由不是"我们要做一个更厉害的 agent 框架"，而是：

> 当前 agent 系统普遍把请求送入过重的执行范式，造成模型调用、tool 调用和上下文 token 浪费。我们将构建一个本地 Agent Gateway，通过 full-text skill routing 和 agentic paradigm routing，在入口选择最低成本且足够可靠的执行路径，并用端到端 benchmark 证明其 cost-quality 优势。

如果能证明以下结果，就有论文和工程双重价值：

- 小模型 full-context router 明显优于 name/description router。
- Paradigm routing 明显降低 fixed ReAct/Planning 的平均成本。
- 本地 router 接近 large LLM meta-controller 的成功率，但显著降低延迟和推理成本。
- Fallback 能有效控制 wrong-route 风险。
- CPU AMX 推理能满足目标延迟和吞吐。

建议立项，先投入 6-8 周做 Phase 1-3 的 proof-of-concept。若 offline routing 和 end-to-end utility 指标过线，再扩大训练和系统优化资源。

## 参考资料

- SkillRouter: Retrieve-and-Rerank Skill Selection for LLM Agents at Scale: https://arxiv.org/abs/2603.22455
- SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents: https://arxiv.org/abs/2605.05726
- AutoTool: Efficient Tool Selection for Large Language Model Agents: https://ojs.aaai.org/index.php/AAAI/article/view/40389
- Utility-Guided Agent Orchestration: https://arxiv.org/abs/2603.19896
- FlowSteer: Steering Agentic Workflows with Reinforcement Learning: https://flowsteer.org/paper.pdf
- Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward: https://arxiv.org/abs/2602.12430
- Towards Secure Agent Skills: Architecture, Threat Taxonomy, and Security Analysis: https://arxiv.org/abs/2604.02837
- Intel Advanced Matrix Extensions overview: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/what-is-intel-amx.html
- OpenVINO CPU device documentation: https://docs.openvino.ai/2024/openvino_docs_OV_UG_supported_plugins_CPU.html
- QLoRA: Efficient Finetuning of Quantized LLMs: https://arxiv.org/abs/2305.14314

