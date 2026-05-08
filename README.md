# Engram

**Cognitive Continuation Layer for LLM Agents**

> Engram 恢复的是 agent 的 **cognition**，不是 machine 的 **execution**——这是与 Temporal / LangGraph 的根本差异。
>
> 跨会话的 **Memory** 解决"知道"，跨中断的 **Continuity** 解决"接着做"。
> Agent 可替换，任务不中断；按需召回上下文，不靠无限堆 token；数据始终保留在本机。
>
> 定位：Claude Code / Cursor 等 MCP 客户端的 **cognitive continuation layer**，不是又一个通用 agent runtime / workflow engine。

[![PyPI](https://img.shields.io/pypi/v/mcp-engram)](https://pypi.org/project/mcp-engram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 为什么需要 Memory + Continuity

AI Agent 的每次会话都是一座孤岛：

- **换个 Agent？** 从头来。
- **上下文窗口满了？** 丢掉历史，继续猜。
- **昨天踩过的坑？** 不知道，再踩一遍。
- **一个任务跨了三次会话？** 没人知道整体进度。

根本原因：**Agent 同时缺少两层基础设施**——

| 层 | 解决的问题 | Engram 的实现 |
| --- | --- | --- |
| **Memory** | 跨会话"知道"什么 | 混合检索 + 遗忘曲线 + 去重/矛盾消解 |
| **Continuity** | 跨中断"接着做"什么 | Task 状态 + 结构化 handoff + 行为级验证 |

Engram 是一个本地运行的 [MCP](https://modelcontextprotocol.io/) Server，把这两层一起交付给 Claude Code / Cursor 等已有客户端。

它**不**做：

- ❌ 通用 agent runtime / workflow orchestration（那是 LangGraph、Temporal 的位置）
- ❌ 自定义 agent loop / prompt 编排（让 MCP 客户端自己处理）
- ❌ 追求恢复时 LLM 行为完全一致（LLM 非确定性是物理限制，做的是 **constrained continuation**——用结构化状态收窄行动空间）

它**专门**做：

- ✅ 在会话中断后让任务可恢复
- ✅ 在 Agent 切换后让上下文可交接
- ✅ 在长任务中保留工程状态（失败、进度、约束）
- ✅ 用后续行为反向验证状态正确性（**Behavioral Handoff Verification**）

---

## Continuity Flow

**AI Agent Continuity 的核心体验：断点恢复。**

```
Agent A（Claude Code）                     Agent B（Cursor）
  │                                          │
  ├─ 创建任务，开始执行                        │
  ├─ 记录进度 + 失败经验                       │
  ├─ ━━━━━━━━━━━━━━━━━━━                     │
  │   ⚡ Session Interrupted                  │
  │   ━━━━━━━━━━━━━━━━━━━                     │
  ├─ session_handoff(交接摘要)                 │
  │        │                                  │
  │        ▼                                  │
  │   ┌─────────────────┐                     │
  │   │   Engram        │                     │
  │   │   Checkpoint    │                     │
  │   │   ┌───────────┐ │                     │
  │   │   │ 任务状态   │ │                     │
  │   │   │ 执行进度   │ │                     │
  │   │   │ 失败教训   │ │                     │
  │   │   │ 下一步计划 │ │                     │
  │   │   └───────────┘ │                     │
  │   └────────┬────────┘                     │
  │            │                              │
  │            ▼                              │
  │     Restore State ─────────────────────▶  │
  │                                          ├─ recall_memory(关键词)
  │                                          │    └─ handoff 自动置顶
  │                                          │    └─ 附带历史失败上下文
  │                                          │    └─ next_steps 执行验证
  │                                          ├─ 接着做，不从零开始
  │                                          └─ session_handoff(交接) ──▶ ...
```

> **无论换了几个 Agent、跨了几次会话，任务状态始终在。**

---

## MCP 工具一览

Engram 提供 **15 个 MCP 工具**，覆盖 Cognitive Continuity 的完整生命周期：

Engram 提供 **15 个 MCP 工具**：
## Checkpoint v2 — Constrained Continuation

把 `session_handoff` 升级为**版本化 cognitive checkpoint**：恢复时不强求新 Agent 复现同一个 action，而是给它一组**约束**收窄行动空间。

**Continuation 包的字段**

| 字段 | 作用 |
| --- | --- |
| `goal` / `completed` / `in_progress` / `blocked` / `preferred_next` | 任务状态主体 |
| `must_not_redo` | Negative memory — 已完成或已产生副作用、不可重做的动作 |
| `must_preserve` | 用户明令的 invariant（如"别动 main 分支"） |
| `working_set` | 中断前的工作集（file / tool / artifact） |
| `continuation_confidence` | 系统自评恢复可靠度（0~1） |

**Event-first 触发**（按认知事件保存，非时间周期；同 reason 60s debounce）

| Reason | 触发场景 |
| --- | --- |
| `MANUAL_HANDOFF` | 调用 `session_handoff` |
| `FAILURE` | 调用 `track_failure`（强触发，绕过 debounce） |
| `PLAN_UPDATE` | `in_progress` Jaccard < 0.7 |
| `WORKING_SET_SHIFT` | 工作集 Jaccard < 0.5 |
| `AUTO_SAVE` | 5 分钟无 checkpoint 兜底 |

**接口**

```python
# 一站式恢复（推荐）：get_task 自带 latest_checkpoint
get_task(task_id=42)["latest_checkpoint"]["continuation"]

# 完整恢复：带相关记忆 + 历史 failure 上下文
restore_checkpoint(task_id=42, memory_restore_mode="SELECTIVE")
# memory_restore_mode: FULL(全量) / SELECTIVE(默认, importance≥0.5 或 failure) / NONE

# checkpoint 历史
list_checkpoints(task_id=42, limit=10)
```

**向后兼容**：现有工具签名不变，新字段追加；老 task 无 checkpoint 时 `restore_checkpoint` 走 fallback。

---

---

## 安装

```bash
pip install mcp-engram
engram-setup          # 下载嵌入模型 + 初始化 DuckDB
```

MCP 客户端配置（Claude Code / Cursor）：

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

数据目录 `~/.engram/`：`memories.duckdb`（单文件 DB） + `graph.json`（语义图） + `model_cache/`（模型）。

### 推荐写入 CLAUDE.md 的 Agent 指令

```markdown
## Memory Rules
- 多步任务起点：create_task(name, goal)
- 任务开始：recall_memory(query) — handoff 自动置顶 + 历史 failure
- 接手任务：get_task(task_id) — 自带 latest_checkpoint
- 阶段进展：track_progress(feature, status, task_id=X)
- 遇到错误：track_failure(error, component, root_cause, task_id=X)
- 任务收尾：session_handoff(summary, completed, in_progress, blocked,
            next_steps, must_not_redo=[...], must_preserve=[...],
            working_set={...}, task_id=X)
```

支持 macOS / Linux / WSL2，Python 3.11+，约 500MB 模型缓存。

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
<summary>分类得分 + 记忆机制详解</summary>

### 分类得分

| Category    |   Count |         F1 |     Hit@5 |
| ----------- | ------: | ---------: | --------: |
| Single-Hop  |     114 |     0.5121 |     76.3% |
| Temporal    |      63 |     0.4501 |     95.2% |
| Multi-Hop   |      43 |     0.3181 |     60.5% |
| Open-Domain |      13 |     0.1324 |     61.5% |
| **Overall** | **233** | **0.4383** | **77.7%** |

### 记忆机制（关键算法摘要）

- **艾宾浩斯衰减**：`strength = importance × e^(-λ × days) × (1 + recall_count × 0.2)`，`failure` 半衰期 ~11 天，`strategy` ~38 天
- **去重**：相似度 ≥0.85 强化、0.65~0.84 检测矛盾后合并/覆盖、<0.65 新建
- **混合检索**：`0.3 × BM25 + 0.7 × (语义相似度 × 衰减强度) + 图谱加成`，HNSW + DuckDB FTS
- **Recall 增强**：handoff 自动置顶 + 关联 failure 上下文 + 动态 `quality_score`
- **自动维护**：每 12h 整合（≥0.70 聚类合并）+ 剪枝（strength<0.05）+ FTS 重建

### 重要性参考

`0.9–1.0` 核心身份/永久事实 · `0.7–0.8` 架构决策/强偏好 · `0.5` 普通事实 · `0.2–0.3` 临时上下文

### 环境变量（高频）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像 |
| `ENGRAM_MODEL` | `all-mpnet-base-v2` | 嵌入模型 |
| `ENGRAM_DEDUP_THRESHOLD` | `0.65` | 去重相似度下限 |
| `ENGRAM_REINFORCE_THRESHOLD` | `0.85` | 强化相似度阈值 |
| `ENGRAM_W_BM25` / `ENGRAM_W_VECTOR` | `0.30` / `0.70` | 检索权重 |
| `ENGRAM_PRUNE_THRESHOLD` | `0.05` | 剪枝强度阈值 |
| `ENGRAM_CONSOLIDATE_THRESHOLD` | `0.70` | 整合聚类阈值 |

完整变量列表见 `src/engram/config.py`。

</details>

---

## Roadmap

聚焦原则：**只做 Memory + Continuity 双层**，凡是滑向"通用 agent runtime / workflow orchestration"的需求一律延后，避免与 LangGraph / Temporal 重叠。

### 已交付

- [x] ~~Error-aware Memory~~ — 按 component 附带历史 failure 上下文 ✅
- [x] ~~Handoff Validation~~ — next_steps 执行状态检测 ✅
- [x] ~~Task Context~~ — Task 一等实体，跨会话任务全景视图 ✅
- [x] ~~Memory Quality Score~~ — 基于 importance + recall + outcome 动态评分 ✅
- [x] ~~Session Lifecycle~~ — 自动心跳、中断检测、atexit 兜底 ✅

### 已交付（Cognitive Continuation 第一层）

- [x] ~~**Checkpoint v2**~~ — 版本化 cognitive checkpoint，event-first 触发（6 类 reason），`restore_checkpoint` / `list_checkpoints` 上线，支持 constrained continuation（must_not_redo 作为 negative memory / must_preserve / preferred_next / working_set / continuation_confidence） ✅

### 进行中（Cognitive Continuation 强化）

- [ ] **Interruption Taxonomy** — 中断分类（overflow / user_away / tool_failure / crash / rate_limit / handoff），按类型路由恢复策略
- [ ] **Behavioral Verification 落表** — `handoff_verifications` 持久化，作为对外差异化能力
- [ ] **Continuity Metrics** — 6 维指标：Goal Retention / Action Consistency / Failure Recall / Working Set Stability / Replanning Rate / Redundant Exploration
- [ ] **Intent-Outcome 两阶段记录** — `track_intent` + `track_progress` 配对，缓解工具副作用不可逆问题
- [ ] **Chaos Continuity Test** — 真实 Coding Agent 执行中强杀 → 另一 Agent restore 接手，自动比对 6 维指标

### 延后 / 不做（Deferred）

- [ ] ~~Multi-Agent Coordination — 多 Agent 并行任务分配与同步~~ → **Deferred**：属于通用 orchestration 范畴，与项目定位冲突，让上层框架（LangGraph / AutoGen）解决。
- [ ] ~~跨模型 next_steps 中间表示~~ → **Deferred**：主流 MCP 客户端使用同档模型，自然语言 next_steps 已足够，过度工程化收益低。
- [ ] Coding Agent 深度集成 — IDE 原生 Task 面板（保留，依赖 Checkpoint v2 完成）

---

## 参与贡献

欢迎通过以下方式参与：

1. **提交 Issue** — 报告 Bug 或提出功能建议
2. **提交 PR** — Fork → 新建分支 → 提交 PR

```bash
git clone https://github.com/hugfeature/engram.git
cd engram
pip install -e ".[dev]"
pytest tests/ -v       # 确保测试通过
```

## 项目负责人

- [@hugfeature](https://github.com/hugfeature)

## License

[MIT](https://opensource.org/licenses/MIT)

---

> **Cognitive Continuation Layer — 我们恢复的是 agent 的 cognition，不是 machine 的 execution。**