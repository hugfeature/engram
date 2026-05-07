# Engram

**Every task leaves a trace.**

Engram 是一个面向 AI Agent 的**会话接力系统**：让任务可以跨会话、跨 Agent 持续推进，而不是每次从零开始。

它本地运行，提供上下文召回、会话交接、工程状态管理和自动整理能力。

## 🎯 产品目标

> **Agent 可替换，任务不中断。**  
> **按需召回上下文，不靠无限堆 token。**  
> **数据始终保留在本机。**

---

## ✅ 你会得到什么

1. **会话接力**：任务中途换 Agent，仍能继续推进。  
2. **上下文压缩**：按需召回关键状态，不再每次喂整段历史。  
3. **本地可控**：数据留在本机，不依赖云端记忆服务。  
4. **工程可追踪**：失败原因、进度、决策可结构化沉淀。

---

## ⚡ 5 分钟上手

```bash
# 安装
pip install mcp-engram

# 初始化（下载模型、创建数据库）
engram-setup
```

在 MCP 客户端中配置 `engram` 服务后即可使用。

---

## 🔁 推荐工作流（跨 Agent 接力）

1. **任务开始**：`recall_memory(query=任务关键词)`  
2. **执行中**：遇到问题用 `track_failure`，进度变化用 `track_progress`  
3. **结束前**：调用 `session_handoff` 写明完成项、阻塞项、下一步  
4. **下个 Agent**：先 recall + 读 handoff，直接接着做

---

## 🔥 核心能力

### 🧩 持久上下文层（记忆引擎）

- 存储事实、偏好、决策（作为可检索上下文）
- 支持语义检索 + BM25 关键词检索 + 图谱关联扩展
- 混合评分：`0.3 × BM25 + 0.7 × (语义相似度 × 衰减强度) + 图谱加成`

### 🔄 Session Handoff（核心能力）

结构化记录：

```json
{
  "summary": "任务目标与当前状态",
  "completed": ["已完成项"],
  "in_progress": ["进行中"],
  "blocked": ["阻塞项"],
  "next_steps": ["下一步行动"]
}
```

👉 新 Agent 接手时，不需要读历史对话，直接继续执行

### 🛠 Engineering State

不仅记“发生了什么”，还记：

- `track_failure`：错误原因 + 根因 + 组件 + 严重级 + 修复方式
- `track_progress`：模块进度 / 阻塞 / 质量评分
- `update_memory`：状态更新

👉 本质是把“测试经验”和“开发上下文”结构化

### 🧹 Auto Maintenance

系统自动：

- **去重**（Deduplication）— 相似度 ≥ 0.85 只增加回忆次数
- **矛盾消解**（Conflict Resolution）— 检测到矛盾自动替换
- **衰减**（Decay）— 艾宾浩斯遗忘曲线，自然淘汰过时信息
- **整合**（Consolidation）— 相似度 ≥ 0.70 的记忆簇自动合并
- **剪枝**（Pruning）— 强度 < 0.05 且无链式保护的记忆自动删除

避免上下文无限膨胀，零运维。

---

## ⚙️ Architecture

```
┌──────────────────────────────────────────────┐
│              MCP Client                      │
│      (Claude Code / Cursor / ...)            │
└──────────────────┬───────────────────────────┘
                   │ stdio / HTTP
┌──────────────────▼───────────────────────────┐
│         shared.py (单例 + 分发)               │
│    server.py (stdio)  ·  http_server.py      │
│              8 MCP tools · APScheduler       │
├──────────────────────────────────────────────┤
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
│  768d 向量编码+降级     │  NetworkX 语义图谱  │
├──────────────────────────────────────────────┤
│              db.py — DuckDB                  │
│  向量存储  ·  BM25 FTS  ·  CRUD             │
└──────────────────────────────────────────────┘

数据文件（~/.engram/）：
├── memories.duckdb     # 向量数据库（单文件，零运维）
├── graph.json          # 语义图谱（JSON 序列化）
└── model_cache/        # 嵌入模型缓存
```

技术特点：

- 单机本地运行（隐私 + 可控）
- DuckDB 单文件存储，BM25 FTS 索引自动维护
- 双传输层：MCP stdio + HTTP/REST
- NetworkX 语义关系图，BFS 扩展联想发现
- 定时自动维护（12h 周期：整合 + 剪枝 + FTS 重建）

---

## 🧠 记忆机制

### 艾宾浩斯遗忘曲线

每条记忆都有一个**强度值**（strength），随时间按指数曲线衰减：

```
effective_λ = base_λ × (1 - importance × 0.8)
strength = importance × e^(-λ × days) × (1 + recall_count × 0.2)
```

| 类别 | 衰减率 λ | 半衰期 | 适用场景 |
|------|---------|--------|----------|
| `strategy` | 0.10 | ~38 天 | 被验证有效的方法论、架构模式 |
| `fact` | 0.16 | ~24 天 | 用户偏好、身份信息、技术选型 |
| `assumption` | 0.20 | ~19 天 | 推断的上下文、不确定的信息 |
| `failure` | 0.35 | ~11 天 | 踩过的坑、环境问题、临时 workaround |

**设计意图**：成功的策略记最久（strategy ~38天），失败的教训记最短（failure ~11天）——环境会变，昨天的坑明天可能已经填上了。

### 智能去重与矛盾消解

```
相似度 ≥ 0.85 → REINFORCE  只增加回忆次数
相似度 0.65~0.84 → 检测矛盾
  ├── 语义矛盾 → REPLACE   用新内容覆盖
  └── 语义兼容 → MERGE     合并为更完整的记忆
相似度 < 0.65 → NEW        存为新记忆
```

### 语义图谱

每条记忆存入时自动与已有记忆建立语义关联（相似度 ≥ 0.40 建边），检索时从命中记忆出发做 BFS（默认最大深度 3，可配置）发现关联知识。链式保护机制确保作为"桥梁"的弱记忆不被误删。

---

## 🚀 Advanced Setup

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

---

## 🧰 MCP Tools

| 工具 | 参数 | 用途 |
|------|------|------|
| `recall_memory` | `query`, `user_id?`, `top_k?` | 语义检索记忆 |
| `store_memory` | `content`, `importance`, `category?`, `metadata?`, `user_id?` | 存储新记忆（自动去重） |
| `update_memory` | `memory_id`, `new_content`, `importance?` | 更新已有记忆 |
| `session_handoff` | `summary`, `completed?`, `in_progress?`, `blocked?`, `next_steps?`, `user_id?` | 结构化会话交接 |
| `track_failure` | `error`, `component`, `root_cause?`, `severity?`, `fix?`, `related_test_ids?`, `user_id?` | 结构化失败归因 |
| `track_progress` | `feature`, `status`, `completion?`, `blockers?`, `quality_score?`, `notes?`, `user_id?` | 功能进度快照 |
| `consolidate_memory` | `user_id?` | 手动触发记忆整合 |
| `memory_stats` | `user_id?` | 记忆统计 + 工程指标 |

### 重要性参考

| 值 | 使用场景 |
|----|---------|
| 0.9–1.0 | 核心身份、永久事实 |
| 0.7–0.8 | 强偏好、架构决策 |
| 0.5 | 普通项目事实 |
| 0.2–0.3 | 临时会话上下文 |

---

## 🔍 What Makes Engram Different

普通 memory 系统：

> 存文本 → 检索文本

Engram：

> 存状态 → 驱动下一步执行

区别在于：

- 有**会话交接**而不是仅检索
- 有**工程状态**而不是仅知识
- 有**自动整理**而不是无限增长
- **本地优先**，零云端依赖，数据永远在你的机器上

---

## 🧪 Benchmark

基于 [LoCoMo](https://github.com/snap-research/locomo)（Snap Research 长期对话记忆基准）的评测。

### Turn Mode — 最佳配置（bge-m3 + reranker, DeepSeek-V3.2）

| Category | Count | F1 | Hit@5 |
|----------|------:|-----:|------:|
| Single-Hop | 114 | 0.5121 | 76.3% |
| Temporal | 63 | 0.4501 | **95.2%** |
| Multi-Hop | 43 | 0.3181 | 60.5% |
| Open-Domain | 13 | 0.1324 | 61.5% |
| **Overall** | **233** | **0.4383** | **77.7%** |

### 优化路径

| 配置 | Overall F1 | Overall Hit@5 |
|------|-----------|---------------|
| **bge-m3 + reranker + weight fix** | **0.4383** | **77.7%** |
| bge-m3 + reranker (r20) | 0.3913 | 69.1% |
| bge-m3 (API, 1024d) | 0.3514 | 61.8% |
| all-mpnet-base-v2 (local, 768d) | 0.2916 | 51.5% |

> 四轮优化累计 **F1 +50.3%**，**Hit@5 +26.2pp**。

### 与业界基线对比

| System | Overall F1 | LLM |
|--------|-----------|-----|
| MemMachine | 0.8487 | GPT-4o-mini |
| Memobase | 0.7578 | GPT-4o-mini |
| Zep | 0.7514 | GPT-4o-mini |
| Mem0 | 0.6688 | GPT-4o-mini |
| **Engram** | **0.4383** | DeepSeek-V3.2 |

> 本地部署零云端依赖，与 Mem0 差距从 56% 缩小到 35%。

---

## 🎯 Use Cases

### AI Coding Agent

- 记住项目约定，避免重复 bug
- 追踪重构进度，跨会话持续

### 多 Agent 协作

- Agent A → Agent B 无缝交接
- 避免重复分析

### 长任务执行

- 防止 context 爆炸
- 保持任务连续性

---

## 🧭 Design Principles

- Memory 是"执行上下文"，不是日志
- 会话交接比历史记录更重要
- 错误应该被结构化，而不是遗忘
- 本地优先，保证可控性

---

## 🔧 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 模型镜像地址 |
| `ENGRAM_MODEL` | `all-mpnet-base-v2` | 嵌入模型名称 |
| `ENGRAM_MODEL_TIMEOUT` | `30` | 模型加载超时（秒），超时进入降级模式 |
| `ENGRAM_PRUNE_THRESHOLD` | `0.05` | 记忆淘汰强度阈值 |
| `ENGRAM_DEDUP_THRESHOLD` | `0.65` | 去重相似度下限 |
| `ENGRAM_REINFORCE_THRESHOLD` | `0.85` | 强化相似度阈值 |
| `ENGRAM_SIM_HIGH` | `0.50` | 检索高阈值 |
| `ENGRAM_SIM_LOW` | `0.20` | 检索低阈值 |
| `ENGRAM_REINFORCE_SIM` | `0.75` | 检索时强化阈值 |
| `ENGRAM_W_BM25` | `0.30` | BM25 关键词权重 |
| `ENGRAM_W_VECTOR` | `0.70` | 向量语义权重 |
| `ENGRAM_GRAPH_DEPTH` | `3` | 图谱 BFS 最大深度 |
| `ENGRAM_EDGE_THRESHOLD` | `0.40` | 图谱建边相似度阈值 |
| `ENGRAM_MAX_EDGES` | `5` | 每条记忆最大边数 |
| `ENGRAM_CONSOLIDATE_THRESHOLD` | `0.70` | 整合聚类相似度阈值 |

---

## 🛣 Roadmap

- [ ] Error-aware Memory（避免重复错误）
- [ ] Handoff Validation（交接一致性检测）
- [ ] Eval Pipeline（量化质量提升）
- [ ] Coding Agent 深度集成（Cursor / Claude）

---

## 📌 Final Thought

> AI 不缺能力，缺的是"记住并持续执行"的能力。

Engram 解决的不是记忆问题，而是：

> **让 Agent 的任务可以不中断地往前走**

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
