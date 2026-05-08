# Engram

**AI Agent Continuity ** — 让 Agent 的任务跨会话、跨实例持续推进，而不是每次从零开始。

## 🎯 产品目标

> **Agent 可替换，任务不中断。**  
> **按需召回上下文，不靠无限堆 token。**  
> **数据始终保留在本机。**

[![PyPI](https://img.shields.io/pypi/v/mcp-engram)](https://pypi.org/project/mcp-engram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 问题

AI Agent 的每次会话都是一座孤岛：

- **换个 Agent？** 从头来。
- **上下文窗口满了？** 丢掉历史，继续猜。
- **昨天踩过的坑？** 不知道，再踩一遍。
- **一个任务跨了三次会话？** 没人知道整体进度。

根本原因：**Agent 没有可持续的执行上下文，也没有以"任务"为中心的记忆组织。**

## 解法

Engram 是一个本地运行的 **MCP Server**，为 Agent 提供六项 continuity 能力：

| 能力                   | 做什么                                                  | 为什么需要               |
| ---------------------- | ------------------------------------------------------- | ------------------------ |
| **Task Context**       | Task 是一等实体，handoff/failure/progress 挂在 task 下  | 跨会话的任务全景视图     |
| **Session Handoff**    | 结构化记录完成项、阻塞项、下一步 + 自动验证执行情况     | Agent 可替换，任务不中断 |
| **Smart Recall**       | 语义 + 关键词 + 图谱混合检索 + type 过滤 + handoff 置顶 | 按需召回，不堆 token     |
| **Error-aware Memory** | 按 component 附带历史 failure 上下文 + 记忆质量评分     | 经验可追溯，错误不重犯   |
| **Engineering State**  | 结构化的失败归因、进度快照、session outcome 反馈闭环    | 量化执行质量             |
| **Auto Maintenance**   | 去重、矛盾消解、Ebbinghaus 衰减、整合、剪枝             | 上下文不膨胀，零运维     |

**数据始终在本机，零云端依赖。**

---

## 快速开始

```bash
pip install mcp-engram
engram-setup          # 下载嵌入模型、初始化数据库
```

### MCP 客户端配置（Claude Code / Cursor）

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "env": { "HF_ENDPOINT": "https://hf-mirror.com" }
    }
  }
}
```

### 推荐写入 CLAUDE.md 的规则

```markdown
## Memory Rules

- 接到多步任务：`create_task(name, goal)` — 创建跟踪任务
- 每次任务开始：`recall_memory(query=任务关键词)` — 自动置顶 handoff，附带 failure 上下文
- 全新知识：`store_memory(content, importance)`
- 补充已有：`update_memory(memory_id, merged_content)`
- 推翻已有：`update_memory(memory_id, new_content)`
- 阶段性进展：`track_progress(feature, status, completion, task_id=X)`
- 遇到错误：`track_failure(error, component, root_cause, task_id=X)`
- 任务结束：`session_handoff(summary, completed, in_progress, blocked, next_steps, task_id=X)`
- 查看任务全景：`get_task(task_id)` — 包含所有关联记忆
```

---

## 工作流：跨 Agent 接力

```
Agent A                             Agent B
  │                                   │
  ├─ create_task(任务名, 目标)          │
  ├─ recall_memory(关键词)              │
  │    └─ handoff 自动置顶              │
  │    └─ 附带 related_failures         │
  │    └─ 附带 quality_score            │
  ├─ ... 执行任务 ...                  │
  ├─ track_failure(错误, task_id)       │
  ├─ track_progress(进度, task_id)      │
  ├─ session_handoff(交接, task_id) ───▶ recall_memory(关键词)
  │                                   ├─ 读取 handoff + next_steps 验证
  │                                   ├─ get_task(task_id) 查看全景
  │                                   ├─ ... 接着做 ...
  │                                   ├─ update_task(task_id, status)
  │                                   └─ session_handoff(交接, task_id) ──▶ ...
```

**核心循环**：create_task → recall → execute → track → handoff → repeat

---

## MCP Tools

Engram 提供 **12 个 MCP 工具**：

### 核心

| 工具              | 用途                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------- |
| `recall_memory`   | 混合检索记忆，支持 `memory_type` 过滤、handoff 自动置顶、附带 failure 上下文和质量评分 |
| `store_memory`    | 存储新记忆，自动去重 / 合并 / 矛盾消解                                                 |
| `update_memory`   | 按 ID 更新已有记忆                                                                     |
| `session_handoff` | 结构化会话交接 + next_steps 执行验证（支持 `task_id` 关联）                            |

### 任务管理

| 工具          | 用途                                                        |
| ------------- | ----------------------------------------------------------- |
| `create_task` | 创建跟踪任务（name / goal），返回 task_id                   |
| `update_task` | 更新任务状态 / 目标 / metadata                              |
| `get_task`    | 获取任务详情 + 关联的全部 handoff / failure / progress 记忆 |

### 工程状态

| 工具              | 用途                                                       |
| ----------------- | ---------------------------------------------------------- |
| `track_failure`   | 结构化失败记录（支持 `task_id` 关联，按 component 可追溯） |
| `track_progress`  | 功能进度快照（支持 `task_id` 关联）                        |
| `session_outcome` | 标记会话成功/失败，多次失败记忆额外降权                    |

### 维护

| 工具                 | 用途                                 |
| -------------------- | ------------------------------------ |
| `consolidate_memory` | 手动触发记忆整合（相似记忆聚类合并） |
| `memory_stats`       | 记忆统计 + 工程指标概览              |

### 重要性参考

| 值          | 场景               |
| ----------- | ------------------ |
| **0.9–1.0** | 核心身份、永久事实 |
| **0.7–0.8** | 架构决策、强偏好   |
| **0.5**     | 普通项目事实       |
| **0.2–0.3** | 临时会话上下文     |

---

## 记忆机制

### 艾宾浩斯遗忘曲线

每条记忆有一个 **strength** 值，随时间按指数衰减：

```
effective_λ = base_λ × (1 - importance × 0.8)
strength = importance × e^(-λ × days) × (1 + recall_count × 0.2)
```

| 类别         | λ    | 半衰期 | 适用                      |
| ------------ | ---- | ------ | ------------------------- |
| `strategy`   | 0.10 | ~38 天 | 被验证有效的方法论        |
| `fact`       | 0.16 | ~24 天 | 用户偏好、技术选型        |
| `assumption` | 0.20 | ~19 天 | 推断的上下文              |
| `failure`    | 0.35 | ~11 天 | 踩过的坑、临时 workaround |

> 策略记最久，失败记最短——环境会变，昨天的坑明天可能已经填上了。

### 智能去重

```
相似度 ≥ 0.85 → REINFORCE   增加回忆次数
相似度 0.65~0.84 → 检测矛盾
  ├── 语义矛盾 → REPLACE    覆盖
  └── 语义兼容 → MERGE      合并
相似度 < 0.65 → NEW         新记忆
```

### 混合检索

```
score = 0.3 × BM25 + 0.7 × (语义相似度 × 衰减强度) + 图谱加成
```

- **BM25**：精确关键词匹配（DuckDB FTS）
- **向量检索**：语义相似度（HNSW 索引加速）
- **图谱扩展**：从命中记忆出发 BFS（深度 ≤ 3），发现关联知识

### Recall 增强

每次 `recall_memory` 返回的结果自动附带三个维度的增强信息：

| 增强                   | 说明                                                                                |
| ---------------------- | ----------------------------------------------------------------------------------- |
| **Handoff Validation** | 检测 handoff 的 `next_steps` 是否被后续 session 执行，返回每步的执行状态            |
| **Related Failures**   | 非 failure 类记忆如果涉及特定 component，自动附带该 component 的历史 failure 上下文 |
| **Quality Score**      | 动态计算 `importance × recall频率 × session outcome` 综合质量分                     |

```
quality_score = clamp(importance × (1 + recall_bonus) × outcome_factor, 0, 1)
  - recall_bonus = min(recall_count × 0.1, 0.5)
  - outcome_factor = (success - failure × 0.5) 调整
```

> 新 Agent 接手时不仅知道"记住了什么"，还知道"哪些做了哪些没做"、"这块踩过什么坑"、"哪条记忆更可靠"。

### 自动维护

每 12 小时自动执行：

- **整合**：相似度 ≥ 0.70 的记忆簇合并
- **剪枝**：strength < 0.05 且无链式保护的记忆删除
- **FTS 重建**：保持全文索引新鲜

---

## 架构

```
┌──────────────────────────────────────────────┐
│              MCP Client                      │
│      (Claude Code / Cursor / ...)            │
└──────────────────┬───────────────────────────┘
                   │ stdio / HTTP
┌──────────────────▼───────────────────────────┐
│    server.py (stdio)  ·  http_server.py      │
│        12 MCP Tools  ·  APScheduler          │
├──────────────────────────────────────────────┤
│  handlers.py — 业务逻辑                       │
│  ┌─ Task ───────────────────────────────┐    │
│  │  create / update / get_task          │    │
│  │  handoff validation · quality score  │    │
│  └──────────────────────────────────────┘    │
│  ┌─ 写入 ──────────┐  ┌─ 读取 ──────────┐   │
│  │  resolve.py     │  │  retrieve.py    │   │
│  │  去重/矛盾消解   │  │  混合检索+评分   │   │
│  └─────────────────┘  └─────────────────┘   │
│  ┌─ 维护 ──────────┐  ┌─ 衰减 ──────────┐   │
│  │  consolidator   │  │  decay.py       │   │
│  │  聚类合并+剪枝   │  │  遗忘曲线+质量分  │   │
│  └─────────────────┘  └─────────────────┘   │
├──────────────────────────────────────────────┤
│  embedding.py (LRU向量编码)  │  graph.py (语义图谱) │
│                              │  batch_mode · user隔离│
├──────────────────────────────────────────────┤
│              db.py — DuckDB                  │
│  memories · tasks · session_outcomes         │
│  向量HNSW  ·  BM25 FTS  ·  CRUD             │
└──────────────────────────────────────────────┘
```

**数据目录** `~/.engram/`：

| 文件              | 说明                                                               |
| ----------------- | ------------------------------------------------------------------ |
| `memories.duckdb` | 向量数据库 — memories + tasks + session_outcomes（单文件，零运维） |
| `graph.json`      | 语义图谱（支持 batch_mode、多用户隔离）                            |
| `model_cache/`    | 嵌入模型缓存                                                       |

---

## 与其他 Memory 系统的区别

| 维度           | 普通 Memory       | Engram                                         |
| -------------- | ----------------- | ---------------------------------------------- |
| **核心抽象**   | 记忆条目          | **Task + 记忆**，以任务为中心组织              |
| **核心动作**   | 存文本 → 检索文本 | 存状态 → **驱动下一步执行**                    |
| **跨会话**     | 仅检索历史        | **结构化交接** + next_steps 执行验证           |
| **工程上下文** | 无                | **失败归因 + 进度追踪 + session outcome**      |
| **召回质量**   | 原样返回          | **quality_score + error-aware + handoff 置顶** |
| **数据增长**   | 无限膨胀          | **自动去重 / 衰减 / 剪枝**                     |
| **部署**       | 云端 API          | **本地优先**，零依赖                           |

---

## Benchmark

基于 [LoCoMo](https://github.com/snap-research/locomo)（Snap Research 长期对话记忆基准）评测：

| System     | Overall F1 | LLM           | 部署方式 |
| ---------- | ---------- | ------------- | -------- |
| MemMachine | 0.8487     | GPT-4o-mini   | 云端     |
| Memobase   | 0.7578     | GPT-4o-mini   | 云端     |
| Zep        | 0.7514     | GPT-4o-mini   | 云端     |
| Mem0       | 0.6688     | GPT-4o-mini   | 云端     |
| **Engram** | **0.4383** | DeepSeek-V3.2 | **本地** |

> 本地部署零云端依赖，四轮优化累计 **F1 +50.3%**，**Hit@5 +26.2pp**。

<details>
<summary>详细分类得分</summary>

| Category    |   Count |         F1 |     Hit@5 |
| ----------- | ------: | ---------: | --------: |
| Single-Hop  |     114 |     0.5121 |     76.3% |
| Temporal    |      63 |     0.4501 |     95.2% |
| Multi-Hop   |      43 |     0.3181 |     60.5% |
| Open-Domain |      13 |     0.1324 |     61.5% |
| **Overall** | **233** | **0.4383** | **77.7%** |

</details>

---

## 环境变量

<details>
<summary>完整配置项</summary>

| 变量                           | 默认值                  | 说明               |
| ------------------------------ | ----------------------- | ------------------ |
| `HF_ENDPOINT`                  | `https://hf-mirror.com` | HuggingFace 镜像   |
| `ENGRAM_MODEL`                 | `all-mpnet-base-v2`     | 嵌入模型           |
| `ENGRAM_MODEL_TIMEOUT`         | `30`                    | 模型加载超时（秒） |
| `ENGRAM_PRUNE_THRESHOLD`       | `0.05`                  | 剪枝强度阈值       |
| `ENGRAM_DEDUP_THRESHOLD`       | `0.65`                  | 去重相似度下限     |
| `ENGRAM_REINFORCE_THRESHOLD`   | `0.85`                  | 强化相似度阈值     |
| `ENGRAM_SIM_HIGH`              | `0.50`                  | 检索高阈值         |
| `ENGRAM_SIM_LOW`               | `0.20`                  | 检索低阈值         |
| `ENGRAM_REINFORCE_SIM`         | `0.75`                  | 检索时强化阈值     |
| `ENGRAM_W_BM25`                | `0.30`                  | BM25 权重          |
| `ENGRAM_W_VECTOR`              | `0.70`                  | 向量权重           |
| `ENGRAM_GRAPH_DEPTH`           | `3`                     | 图谱 BFS 深度      |
| `ENGRAM_EDGE_THRESHOLD`        | `0.40`                  | 图谱建边阈值       |
| `ENGRAM_MAX_EDGES`             | `5`                     | 每条记忆最大边数   |
| `ENGRAM_CONSOLIDATE_THRESHOLD` | `0.70`                  | 整合聚类阈值       |

</details>

---

## Roadmap

- [x] ~~Error-aware Memory~~ — 按 component 附带历史 failure 上下文 ✅
- [x] ~~Handoff Validation~~ — next_steps 执行状态检测 ✅
- [x] ~~Task Context~~ — Task 一等实体，跨会话任务全景视图 ✅
- [x] ~~Memory Quality Score~~ — 基于 importance + recall + outcome 动态评分 ✅
- [ ] Eval Pipeline — 自动化质量回归测试
- [ ] Multi-Agent Coordination — 多 Agent 并行任务分配与同步
- [ ] Coding Agent 深度集成 — IDE 原生 Task 面板

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

---

> **AI 不缺能力，缺的是"记住并持续执行"的能力。**
> Engram 让 Agent 的任务——不论换了几个 Agent、跨了几次会话——都可以不中断地往前走。
