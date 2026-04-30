# Engram

Every thought leaves a trace.

给 AI Agent 装一颗会遗忘的大脑。

Engram 是一个完全本地运行的 MCP 记忆服务。它不只是"存取"——它模拟人类记忆的遗忘、强化、联想机制，让 Agent 跨会话记住真正重要的事，自然忘掉不再需要的细节。

**零云端依赖，数据永远在你的机器上。**

---

## 解决什么痛点

**痛点 1：跨会话状态断裂**

AI Agent 的每次对话都是一张白纸。你昨天告诉它的偏好、上周达成的架构决策、上个月踩过的坑——下次对话全部归零。Context window 清空 = 大脑格式化。

**痛点 2：文件熵增**

把上下文塞进 CLAUDE.md / .cursorrules 看似解决了问题，实际制造了新问题：文件越写越长、过时信息和新信息混在一起、手动维护成本持续增加。你不是在管记忆，你是在维护一个越来越难读的文档。

**痛点 3：工程状态丢失**

Agent 做了什么、卡在哪里、下一步该做什么——这些结构化的工程状态没有地方存。每次新会话都要花 10 分钟"重新对齐"，然后做重复的事。

---

**Engram 的解法：** 不是"把什么都存下来"，而是模拟人类记忆的遗忘-强化-联想机制：

- 重要的偏好和决策，衰减极慢，几乎永久保留
- 临时的调试上下文，11 天自然消退
- 反复被回忆起的知识，越用越牢固
- 矛盾的信息自动覆盖，不会左右互搏
- **v0.2 新增**：结构化会话交接（session handoff），让下一个会话从断点继续而非从零开始
- **v0.4 新增**：工程状态中枢 — 结构化失败归因（track_failure）和进度追踪（track_progress），让 Agent 不只记信息，还记工程状态

---

## 核心机制

### 1. 艾宾浩斯遗忘曲线

每条记忆都有一个**强度值**（strength），随时间按指数曲线衰减：

```
effective_λ = base_λ × (1 - importance × 0.8)
strength = importance × e^(-λ × days) × (1 + recall_count × 0.2)
```

三个因子共同决定一条记忆能活多久：

| 因子 | 作用 | 机制 |
|------|------|------|
| **重要性**（importance） | 越重要衰减越慢 | 最高可将衰减率降低 80% |
| **类别**（category） | 不同类型不同半衰期 | 见下表 |
| **回忆次数**（recall_count） | 越常用越牢固 | 每次召回 +20% 强度 |

**四种记忆类别：**

| 类别 | 衰减率 λ | 半衰期 | 适用场景 |
|------|---------|--------|----------|
| `strategy` | 0.10 | ~38 天 | 被验证有效的方法论、架构模式 |
| `fact` | 0.16 | ~24 天 | 用户偏好、身份信息、技术选型 |
| `assumption` | 0.20 | ~19 天 | 推断的上下文、不确定的信息 |
| `failure` | 0.35 | ~11 天 | 踩过的坑、环境问题、临时 workaround |

**设计意图**：成功的策略要记最久（strategy ~38天），失败的教训记最短（failure ~11天）——因为环境会变，昨天的坑明天可能已经填上了。

### 2. 智能去重与矛盾消解

存入新记忆时，系统不是简单追加，而是先和已有记忆做语义比对：

```
相似度 ≥ 0.85 → REINFORCE  只增加回忆次数，不重复存储
相似度 0.65~0.84 → 检测矛盾
  ├── 语义矛盾 → REPLACE   用新内容覆盖旧内容
  └── 语义兼容 → MERGE     合并为一条更完整的记忆
相似度 < 0.65 → NEW        存为新记忆
```

**矛盾检测**通过极性分析实现：提取正面词（prefer/love/adopt）和负面词（avoid/hate/reject），加上否定词（not/don't/never），判断两条记忆是否表达相反立场。

例：已有"用户偏好 TypeScript"，再存入"用户决定放弃 TypeScript 改用 Go"，系统识别为矛盾，自动用新记忆替换旧的。

### 3. 混合检索（向量 + BM25 + 图谱）

召回记忆时使用三路混合评分：

```
最终得分 = 0.4 × BM25关键词得分 + 0.6 × (语义相似度 × 衰减强度) + 图谱加成
```

**为什么不只用向量搜索？**

| 检索方式 | 擅长 | 不擅长 |
|---------|------|--------|
| 向量搜索 | "他上次提到的那个部署方式" → 语义理解 | 精确术语匹配 |
| BM25 | "DuckDB" → 精确关键词 | 语义相近但措辞不同 |
| 图谱扩展 | A→B→C 关联发现 | 独立的、无关联的记忆 |

三路融合的效果：用"数据库性能"查询，不仅能找到直接提到性能的记忆，还能通过图谱关联找到相关的索引策略、缓存决策等记忆。

### 4. 语义图谱

每条记忆在存入时自动和已有记忆建立语义关联：

- 计算与所有已有记忆的余弦相似度
- 相似度 ≥ 0.40 的建立双向边，权重 = 相似度 × 0.5
- 每条记忆最多连接 5 个最相似的邻居

**图谱的两个关键作用：**

**联想发现**：检索时从命中的记忆出发做 BFS（最大深度 2），沿边找到关联记忆，即使它们和查询词没有直接的语义相似度。就像人类"由此及彼"的联想。

**链式保护**：当一条记忆自身衰减到阈值以下，如果它的邻居中仍有强记忆，则保留它——因为它可能是连接两个重要知识点的桥梁。

### 5. 自动整合与淘汰

后台每 12 小时自动运行一次维护任务：

**整合（Consolidation）**：
1. 找出相似度 ≥ 0.70 的记忆簇
2. 保留重要性最高的一条作为主记忆
3. 将其余记忆的独有信息合并进来
4. 重新计算向量和图谱关系
5. 删除被合并的冗余记忆

**淘汰（Pruning）**：
1. 计算每条记忆的当前强度
2. 强度 < 0.05 **且** 通过链式安全检查 → 删除
3. 强度 < 0.05 但邻居仍强 → 保留（链式保护）

这意味着记忆库会自动保持精简——不需要手动清理，也不会无限膨胀。

---

## 工程状态中枢（v0.4）

Engram 不只是"存信息"的记忆插件——它是**懂工程流程的状态层**。


### 失败归因（track_failure）

当 Agent 遇到 bug、测试失败或部署问题时，用结构化格式记录：

```python
# MCP 调用
track_failure(
    error="CSRF token missing on checkout",
    component="payment",
    severity="critical",        # → importance=0.9
    root_cause="middleware not loaded after refactor",
    fix="re-add CsrfMiddleware to pipeline",
    related_test_ids=["test_checkout_01", "test_payment_csrf"]
)
```

**设计决策**：
- `severity` 自动映射 `importance`（critical=0.9, major=0.7, minor=0.5）
- 固定使用 `failure` 类别（最快衰减 λ=0.35，~11天半衰期）——环境会变，旧的失败记录自然过期
- `component` 字段支持按模块聚合统计，快速定位高风险区域

### 进度追踪（track_progress）

跨会话追踪功能/任务状态：

```python
track_progress(
    feature="login-flow-refactor",
    status="in_progress",       # → importance=0.8
    completion=60,
    blockers=["waiting for API design review"],
    quality_score=0.85,
    notes="auth module done, UI pending"
)
```

**设计决策**：
- `status` 自动映射 `importance`（blocked=0.9 最高，done=0.5 最低）
- 固定使用 `strategy` 类别（最慢衰减 λ=0.10，~38天半衰期）——进度状态要记最久
- 完成的特性自然衰减消失，不需要手动清理

### 工程指标（memory_stats 增强）

`memory_stats` 现在会自动聚合工程数据：

```json
{
  "total": 42,
  "categories": {"fact": 20, "failure": 8, "strategy": 14},
  "engineering": {
    "failures": {
      "total": 8,
      "by_component": {"auth": 5, "payment": 3},
      "by_severity": {"critical": 2, "major": 6}
    },
    "features": {
      "total_tracked": 4,
      "active": {
        "login-refactor": {"status": "in_progress", "completion": 60},
        "payment-fix": {"status": "blocked", "completion": 30}
      }
    }
  }
}
```

---

## 技术架构

```
┌──────────────────────────────────────────────┐
│              MCP Client                      │
│      (Claude Code / Cursor / ...)            │
└──────────────────┬───────────────────────────┘
                   │ stdio (JSON-RPC)
┌──────────────────▼───────────────────────────┐
│              server.py                       │
│  8 MCP tools  ·  APScheduler (12h 维护)      │
├──────────────────────────────────────────────┤
│                                              │
│  ┌─ 写入路径 ──────┐  ┌─ 读取路径 ──────┐    │
│  │  resolve.py     │  │  retrieve.py    │    │
│  │  去重/矛盾消解   │  │  混合检索+评分   │    │
│  └─────────────────┘  └─────────────────┘    │
│                                              │
│  ┌─ 维护路径 ──────┐  ┌─ 统计路径 ──────┐    │
│  │  consolidator   │  │  decay.py       │    │
│  │  聚类合并+剪枝   │  │  遗忘曲线+强度   │    │
│  └─────────────────┘  └─────────────────┘    │
│                                              │
├──────────────────────────────────────────────┤
│  embedding.py          │  graph.py           │
│  768d / 1024d 向量编码  │  NetworkX 语义图谱  │
├──────────────────────────────────────────────┤
│              db.py — DuckDB                  │
│  向量存储  ·  BM25 全文索引  ·  CRUD          │
└──────────────────────────────────────────────┘

数据文件（~/.engram/）：
├── memories.duckdb     # 向量数据库（单文件，零运维）
├── graph.json          # 语义图谱（JSON 序列化）
└── model_cache/        # 嵌入模型缓存
```
---


## MCP 工具接口

| 工具 | 参数 | 用途 |
|------|------|------|
| `recall_memory` | `query`, `user_id?`, `top_k?` | 语义检索记忆，每次任务开始时调用。返回结果包含 metadata |
| `store_memory` | `content`, `importance`, `category?`, `metadata?`, `user_id?` | 存储新记忆（自动去重），返回 memory_id |
| `update_memory` | `memory_id`, `new_content`, `importance?` | 更新已有记忆 |
| `session_handoff` | `summary`, `completed?`, `in_progress?`, `blocked?`, `next_steps?`, `user_id?` | 结构化会话交接，记录当前进度供下次会话继续 |
| `track_failure` | `error`, `component`, `root_cause?`, `severity?`, `fix?`, `related_test_ids?`, `user_id?` | **v0.4** 结构化失败归因，自动关联组件/严重级/修复方案 |
| `track_progress` | `feature`, `status`, `completion?`, `blockers?`, `quality_score?`, `notes?`, `user_id?` | **v0.4** 功能进度快照，跨会话追踪特性状态 |
| `consolidate_memory` | `user_id?` | 手动触发记忆整合 |
| `memory_stats` | `user_id?` | 记忆统计 + **v0.4** 工程指标（失败趋势、组件健康度、活跃特性） |

### 重要性参考

| 值 | 使用场景 |
|----|---------|
| 0.9–1.0 | 核心身份、永久事实（"用户是后端工程师"） |
| 0.7–0.8 | 强偏好、架构决策（"项目用 Go + PostgreSQL"） |
| 0.5 | 普通项目事实（"最近在做登录模块重构"） |
| 0.2–0.3 | 临时会话上下文（"这次调试用的测试账号"） |

---

## 对用户的收益

### 1. Agent 真正"认识"你

不再每次对话都要重新介绍自己的技术栈、编码习惯和项目背景。Agent 记住你偏好 Go 而不是 Java，知道你们项目用 monorepo，了解你上周做的架构决策。

### 2. 知识自然进化

矛盾消解意味着 Agent 的认知永远是最新的。你从 React 切到 Vue？一次对话自动更新。不需要手动维护一个"Agent 应该知道什么"的列表。

### 3. 零运维

- 不需要手动清理旧记忆——遗忘曲线自动淘汰
- 不需要手动合并重复——整合器自动处理
- 不需要担心数据膨胀——12 小时一次自动维护
- 不需要外部服务——DuckDB 单文件，开箱即用

### 4. 完全隐私

所有数据存在 `~/.engram/`，不联网、不上传、不依赖任何云服务。嵌入模型也是本地运行。你的记忆就是你的。

### 5. 联想式发现

图谱扩展让 Agent 不只是"搜到什么返回什么"，而是能沿着语义关联找到相关但不直接匹配的知识。就像你问一个老同事某个问题，他不光回答问题本身，还会提一嘴"对了，这个和上次那个事有关"。

### 6. 越用越聪明

回忆强化机制：被反复召回的记忆强度越来越高，衰减越来越慢。Agent 自动学会什么知识对你最有价值。

---

## 快速开始

```bash
# 安装
pip install mcp-engram

# 初始化（下载模型、创建数据库）
engram-setup

# 按照输出提示将配置块添加到 Claude Code 配置中
```

### Claude Code 配置

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "env": {
        "HF_ENDPOINT": "https://hf-mirror.com"
      }
    }
  }
}
```

### CLAUDE.md 集成

在项目 `CLAUDE.md` 中添加：

```markdown
## Memory Rules

### Step 1 — 先回忆再行动
每次任务开始时，用请求中的关键词调用 `recall_memory`。

### Step 2 — 学到新东西就存
| 情况 | 操作 |
|------|------|
| 全新知识 | `store_memory(content, importance)` |
| 补充已有 | `update_memory(memory_id, merged_content)` |
| 推翻已有 | `update_memory(memory_id, new_content)` |
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 模型镜像 |
| `ENGRAM_MODEL` | `all-mpnet-base-v2` | 嵌入模型名称 |

---

## 关键阈值速查

| 参数 | 值 | 含义 |
|------|-----|------|
| 嵌入维度 | 768 | all-mpnet-base-v2 |
| 去重 REINFORCE | ≥ 0.85 | 几乎相同，只加回忆次数 |
| 去重 MERGE/REPLACE | 0.65~0.84 | 检测矛盾或合并 |
| 整合聚类 | ≥ 0.70 | 自动合并相似记忆 |
| 图谱建边 | ≥ 0.40 | 创建语义关联 |
| 淘汰阈值 | < 0.05 | 删除衰减殆尽的记忆 |
| 检索高阈值 | ≥ 0.50 | 主向量搜索 |
| 检索低阈值 | ≥ 0.20 | 降级搜索 |
| BM25 权重 | 40% | 关键词匹配贡献 |
| 向量权重 | 60% | 语义匹配贡献 |
| 图谱加成 | 30% | 关联记忆的额外加分 |

---

## LoCoMo Benchmark 评测

基于 [LoCoMo](https://github.com/snap-research/locomo)（Snap Research 长期对话记忆基准）的检索质量评测。LoCoMo 是 Mem0/Zep/Memobase/MemMachine 等产品统一使用的评测标准。

### 评测配置

- 数据集：locomo10.json（2/10 conversations, 233 QA, 排除 adversarial）
- 检索：recall() top-k=5
- LLM：DeepSeek-V3.2 / GLM-5.1（注：基线产品统一使用 GPT-4o-mini）
- 指标：Token-level F1（LoCoMo 官方指标）+ Hit@5（LLM 无关的检索命中率）

### Turn Mode — 最佳配置（bge-m3 + bge-reranker-v2-m3, DeepSeek-V3.2）

两阶段检索：recall top-50 → CrossEncoder rerank to top-5，importance=1.0 修正权重比

| Category | Count | F1 | Hit@5 |
|----------|------:|-----:|------:|
| Single-Hop | 114 | 0.5121 | 76.3% |
| Temporal | 63 | 0.4501 | **95.2%** |
| Multi-Hop | 43 | 0.3181 | 60.5% |
| Open-Domain | 13 | 0.1324 | 61.5% |
| **Overall** | **233** | **0.4383** | **77.7%** |

### Turn Mode — 优化路径（DeepSeek-V3.2）

| 配置 | Overall F1 | Overall Hit@5 |
|------|-----------|---------------|
| **bge-m3 + reranker + weight fix** | **0.4383** | **77.7%** |
| bge-m3 + reranker (r20) | 0.3913 | 69.1% |
| bge-m3 (API, 1024d) | 0.3514 | 61.8% |
| all-mpnet-base-v2 (local, 768d) | 0.2916 | 51.5% |

> 四轮优化累计 **F1 +50.3%**（0.29 → 0.44），**Hit@5 +26.2pp**（51.5% → 77.7%）。

### Turn Mode — LLM 对比（all-mpnet-base-v2）

| LLM | Overall F1 | Single-Hop | Temporal | Multi-Hop | Open-Domain | 耗时 |
|-----|-----------|-----------|----------|-----------|-------------|------|
| **DeepSeek-V3.2** | **0.2916** | 0.3470 | 0.3257 | 0.1772 | 0.0192 | 239s |
| GLM-5.1 | 0.2477 | 0.2672 | 0.3214 | 0.1430 | 0.0659 | 2011s |

### Observation Mode（抽象 assertive facts）

| Category | Count | F1 |
|----------|------:|-----:|
| Single-Hop | 114 | 0.3000 |
| Multi-Hop | 43 | 0.1837 |
| Open-Domain | 13 | 0.0659 |
| Temporal | 63 | 0.0590 |
| **Overall** | **233** | **0.2003** |

### 与业界基线对比

| System | Overall F1 | LLM | Embedding |
|--------|-----------|-----|-----------|
| MemMachine | 0.8487 | GPT-4o-mini | — |
| Memobase | 0.7578 | GPT-4o-mini | — |
| Zep | 0.7514 | GPT-4o-mini | — |
| Mem0 | 0.6688 | GPT-4o-mini | — |
| **Engram** | **0.4383** | DeepSeek-V3.2 | bge-m3 + reranker |

> **结论**：四轮优化 mpnet(0.29) → bge-m3(0.35) → +reranker(0.39) → +weight fix+r50(0.44)。Hit@5: 51.5% → 77.7%。与 Mem0(0.67) 的差距从 56% 缩小到 35%。

### Best Config 速查

> **推荐配置**：`bge-m3` (1024d) + `bge-reranker-v2-m3` 两阶段检索
>
> | 指标 | 值 | 说明 |
> |------|-----|------|
> | Overall F1 | **0.4383** | Token-level，DeepSeek-V3.2 |
> | Overall Hit@5 | **77.7%** | 纯检索命中率，LLM 无关 |
> | Temporal Hit@5 | **95.2%** | 时序类问题表现突出 |
> | 优化幅度 | F1 +50.3%, Hit +26.2pp | 四轮累计（相对初始 mpnet） |
>
> 关键参数：`recall top-50 → rerank to top-5`，`importance=1.0` 修正权重比。
> 本地部署零云端依赖，与使用 GPT-4o-mini 的 Mem0 差距缩小至 35%。

---

## 开发

```bash
git clone https://github.com/hugfeature/engram.git
cd engram
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
