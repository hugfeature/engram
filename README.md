# Engram

**Durable Agent Runtime — cross-session task continuity for MCP-aware coding agents**

> Engram 让 Agent 在中断、重启、跨 session 后能恢复**任务执行状态与工作上下文**。
> 不是又一个向量库 / 长期记忆库——主轴是 **runtime durability + execution continuity**。
> 定位：Claude Code / Cursor / OpenHands / Devin 类 runtime infra 的连续性层。

[![PyPI](https://img.shields.io/pypi/v/mcp-engram)](https://pypi.org/project/mcp-engram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 两条铁律 / Two Laws

> **Rule 1.** Event log is the only durability primitive.
> **Rule 2.** If it cannot be replayed, it is not critical state.

任何宣称"必须不丢"的数据，必须先写 `~/.engram/events/*.jsonl`（append-only, fsync），
DuckDB 只是它的 projection layer。

## 三层架构 / Tiered Architecture

```
Tier 1 — Runtime Continuity Layer  (Source of Truth, must never be lost)
  tasks · checkpoints · session lifecycle · handoff events
  → append-only event log (~/.engram/events/) + replay-recoverable

Tier 2 — Semantic Recall Layer    (Degradable, readonly-recoverable)
  memories.content · metadata · summaries · semantic graph
  → DuckDB projection from event log

Tier 3 — Derived Retrieval Cache  (Disposable, rebuildable)
  embeddings · FTS · vector index · rerank cache
  → never participates in recovery; rebuilt on demand
```

DB 损坏不会静默重置：进入 **readonly degraded mode**，用 `engram-setup recover` 显式重建。

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

### 已交付（Interruption Taxonomy）

- [x] ~~**Interruption Taxonomy**~~ — 6 类中断分类（overflow / user_away / tool_failure / crash / rate_limit / unknown），按类型路由恢复策略。新增 `report_interruption` MCP tool 供 LLM 主动上报；`cleanup_stale_sessions` 自动启发式分类；recall 的 `interrupted_sessions` hint 按中断类型给出针对性恢复建议 ✅

### 已交付（Chaos Continuity Test + Continuity Metrics）

- [x] ~~**Continuity Metrics**~~ — 6 维指标引擎（`continuity.py`）：Goal Retention / Action Consistency / Failure Recall / Working Set Stability / Replanning Rate / Redundant Exploration。`restore_checkpoint` 自动附带 `continuity_score`，新增 `evaluate_continuity` MCP tool 供主动评估 ✅
- [x] ~~**Chaos Continuity Test**~~ — 5 大中断场景自动化测试：正常 handoff（基准）/ SIGTERM / kill -9 crash / failure mid-session / working set drift，全部通过并量化恢复质量 ✅

### 已交付（P1-6 Event Log Gzip Rotate）

- [x] ~~**Event Log Gzip Rotate**~~ — 非当天的 `events-YYYYMMDD.jsonl` 在 boot 时自动 gzip 压缩为 `.jsonl.gz`，节省磁盘空间。recover / iter_events 透明读取 `.gz` 文件。压缩前验证行数一致，安全无损 ✅

### 进行中（Cognitive Continuation 强化）

- [ ] **Behavioral Verification 落表** — `handoff_verifications` 持久化，作为对外差异化能力

### 延后 / 不做（Deferred）

- [ ] ~~Multi-Agent Coordination — 多 Agent 并行任务分配与同步~~ → **Deferred**：属于通用 orchestration 范畴，与项目定位冲突，让上层框架（LangGraph / AutoGen）解决。
- [ ] ~~跨模型 next_steps 中间表示~~ → **Deferred**：主流 MCP 客户端使用同档模型，自然语言 next_steps 已足够，过度工程化收益低。
- [ ] Coding Agent 深度集成 — IDE 原生 Task 面板（保留，依赖 Checkpoint v2 完成）

---

## Changelog

### v0.13.1 — P1-6 Event Log Gzip Rotate

主轴：**永久保留事件日志也不爆盘**。

**新增**

- ✨ **Event log gzip rotate**（`event_log.py`）：`rotate_old_files()` 方法将非当天的 `.jsonl` 文件 gzip 压缩为 `.jsonl.gz`。压缩前后行数校验，确保零数据丢失。
- ✨ **透明读取 `.jsonl.gz`**（`event_log.py`）：`_sorted_event_files()` 同时识别 `.jsonl` 和 `.jsonl.gz`；`_iter_file()` 根据后缀自动选择 `open` 或 `gzip.open`。同日期同时存在两种格式时 `.jsonl` 优先。
- ✨ **Boot 自动 rotate**（`maintenance.py`）：`schedule_startup_maintenance()` 在 daemon 线程中自动调用 `rotate_event_logs()`，不阻塞启动。
- ✨ **Recover 透明兼容**：`recover()` 通过 `iter_events()` 间接受益，无需任何修改即可从 `.gz` 文件恢复。

**升级路径**

```bash
pip install -U mcp-engram      # 安装不变
```

- 完全向前兼容 v0.13.0：旧的 `.jsonl` 文件照常读取，首次启动后自动压缩历史文件

**回归**

- 456 tests passed（v0.13.0 的 441 + 新增 15：`test_event_log_rotate` 15）
- 0 lint error

### v0.13.0 — Chaos Continuity Test + Continuity Metrics

主轴：**量化 Agent 跨中断恢复的认知质量**，回答 "checkpoint restore 到底好不好" 这个核心问题。

**新增**

- ✨ **6 维 Continuity Metrics 引擎**（`continuity.py`）：每次 checkpoint restore 自动计算 6 维得分 + 加权 composite 评分。维度：Goal Retention（目标保持度）/ Action Consistency（行动一致性）/ Failure Recall（失败记忆召回率）/ Working Set Stability（工作集稳定度）/ Replanning Rate（重规划率）/ Redundant Exploration（冗余探索率）。
- ✨ **MCP tool `evaluate_continuity`**（`tools.py` / `handlers.py`）：LLM 可主动评估任意两个 checkpoint 版本之间的 continuity score。支持传入 `actions_taken_after_restore` 衡量 redundant exploration。
- ✨ **`restore_checkpoint` 自动附带 `continuity_score`**（`handlers.py`）：restore 时自动对比 parent_version，在响应中嵌入 6 维评分。LLM 可据此判断 "这次恢复的质量够不够好，是否需要额外补偿"。
- ✨ **Chaos Continuity Test 测试套件**（`test_chaos_continuity.py`）：5 大场景自动化验证 —— S1: Normal Handoff (baseline) / S2: SIGTERM (atexit fires) / S3: kill -9 Crash / S4: Failure Mid-Session / S5: Working Set Drift。

**升级路径**

```bash
pip install -U mcp-engram      # 安装不变
```

- 完全向前兼容 v0.12：`evaluate_continuity` 是新工具，不影响已有 client
- `restore_checkpoint` 的 `continuity_score` 是可选输出，旧 client 忽略即可

**回归**

- 441 tests passed（v0.12 的 404 + 新增 37：`test_continuity_metrics` 28 + `test_chaos_continuity` 9）
- 0 lint error

### v0.12.0 — Interruption Taxonomy

主轴：让下一个 Agent **知道上一个 Agent 是怎么中断的**，并据此选择最优恢复策略，而非千篇一律的 "session ended unexpectedly"。

**新增**

- ✨ **6 类中断分类**（`db.py`）：`overflow` / `user_away` / `tool_failure` / `crash` / `rate_limit` / `unknown`，每类对应一套恢复策略（restore_checkpoint + memory_restore_mode + hint）。
- ✨ **MCP tool `report_interruption`**（`tools.py` / `handlers.py`）：LLM 在检测到即将中断时（如 context window 快满、API 限流）主动调用，记录中断原因。该原因在进程退出时写入 `session_lifecycle`，下一个 Agent 可据此获得针对性恢复建议。
- ✨ **Stale session 自动分类**（`db.py`）：`cleanup_stale_sessions` 现在通过启发式规则自动分类中断原因：session < 2min → `crash`；≥ 2 条 failure 记忆 → `tool_failure`；其他 → `user_away`。
- ✨ **Taxonomy-aware recall hints**（`handlers.py`）：`recall_memory` 返回的 `interrupted_sessions` 不再是千篇一律的提示，而是按中断类型给出针对性恢复策略（`recovery_strategy` / `memory_restore_mode` / `hint`）。
- ✨ **atexit 中断感知**（`shared.py`）：`_on_exit` 现在检查 LLM 是否通过 `report_interruption` 预先报告了中断原因，有则写入 session_lifecycle，无则标记为正常 `process_exit`。

**Schema 变更**

- `session_lifecycle` 新增 `interruption_reason VARCHAR` + `interruption_context JSON` 两列
- 向前完全兼容：旧数据的 `interruption_reason = NULL` 自动视为 `unknown`；schema 迁移通过 `ALTER TABLE ADD COLUMN IF NOT EXISTS` 实现

**新增 Event 字段**

- `session.end` 事件新增可选字段：`interruption_reason` / `interruption_context`
- `engram recover` 的 `_replay_session_end` 已支持回放这两个字段

**升级路径**

```bash
pip install -U mcp-engram      # 安装不变
engram-setup doctor            # session_lifecycle 表自动加列
```

- 完全向前兼容 v0.11：旧 session 的 `interruption_reason` 为 NULL，recall hint 回退到 `unknown` 策略
- 新增的 `report_interruption` 工具是可选的；不调用时行为与 v0.11 完全一致

**回归**

- 404 tests passed（v0.11 的 387 + 新增 17：`test_interruption_taxonomy`）
- 0 lint error

### v0.11.0 — Operational Hardening

主轴：在 v0.10 的"两条铁律"基础上，把 **运维可见性** 和 **灾难性增长防护** 补齐。零配置默认开启，向前完全兼容 v0.10。

**新增**

- ✨ **周期性 Snapshot + Replay 加速**（`snapshot.py`）：每写入 N 条 event（默认 1000）或每 H 小时（默认 1）异步快照 DuckDB 文件到 `~/.engram/snapshots/snapshot-seq{N}-{ts}.duckdb`。`engram-setup recover` 优先从最新 snapshot 加载并只 replay `seq > snapshot_seq` 的事件，长期运行的 engram 不再因事件累积而拖慢恢复。
- ✨ **Backup 自动归档策略**（`maintenance.py`）：`~/.engram/backups/` 中受管文件（`memories-pre-recover-*` / `memories-pre-duckdb-upgrade-*`）超过 `ENGRAM_BACKUP_RETAIN`（默认 10）时，最旧的归档到 `backups/archive/`（**只移动不删除**，可恢复）。
- ✨ **DuckDB 版本升级自动备份**：检测到 `duckdb_version` 跨 minor/major 变化（如 `1.5 → 1.6`、`0.9 → 0.10`）时，启动前 `cp` 当前 DB 到 `backups/memories-pre-duckdb-upgrade-<old>-to-<new>-<ts>.duckdb`，并写一条 `runtime.duckdb_upgrade` event 锚定时间。
- ✨ **MCP tool `get_runtime_health`**：让 LLM（Claude Code / Cursor）能主动查询 engram 健康状态。返回 `advice` 数组（可读建议）+ 完整 `doctor()` 字段，degraded mode 时 LLM 可主动提示用户跑 `engram-setup recover`。
- ✨ **`engram-setup doctor` 输出增强**：新增 `backups`（`live_count` / `retain` / `archive_count` / `live_recent`）和 `snapshots`（`count` / `latest_seq` / `latest_size_bytes`）章节；超出 retention 时打印归档提示。
- ✨ **Recover 报告增强**：新增 `snapshot_used` / `snapshot_seq` 字段，可清楚看到本次 recover 是从哪个 snapshot 启动的。

**新增 Event Kinds**（不参与 Tier 1 replay，仅供运维审计）

```
snapshot.create            # {snapshot_path, seq, db_size_bytes}
runtime.duckdb_upgrade     # {old_version, new_version, backup_path}
maintenance.backup_pruned  # {archived: [...], kept, dir}
```

**新增 Env Vars**

| 变量 | 默认 | 说明 |
|---|---|---|
| `ENGRAM_BACKUP_RETAIN` | 10 | `backups/` 保留份数，超出归档到 `archive/` |
| `ENGRAM_SNAPSHOT_INTERVAL_EVENTS` | 1000 | 每写入多少 event 触发一次 snapshot |
| `ENGRAM_SNAPSHOT_INTERVAL_HOURS` | 1.0 | 距离上次 snapshot 最长间隔（小时） |
| `ENGRAM_SNAPSHOT_RETAIN` | 5 | snapshot 保留份数（旧的删除） |

**升级路径**

```bash
pip install -U mcp-engram      # 安装命令不变
engram-setup doctor            # 看到 backups + snapshots 新章节即升级成功
```

- 完全向前兼容 v0.10：snapshot 不存在时 recover 自动退化为全量 replay
- 无需修改 MCP client 配置；`get_runtime_health` 是新工具，老 client 不受影响
- 后台 maintenance 线程仅在主运行时进程启动（短期工具脚本如 doctor / recover 不触发）

**回归**

- 381 tests passed（v0.10 的 348 + 新增 33：`test_backup_pruner` / `test_duckdb_upgrade` / `test_runtime_health_tool` / `test_snapshot`）
- 0 lint error

### v0.10.0 — Durable Agent Runtime（架构重构）

定位升级：从 *AI Memory System* 转向 **Durable Agent Runtime**。
主轴：`runtime durability + execution continuity`，向量召回降级为辅助。

**两条铁律**

> Event log is the only durability primitive.
> If it cannot be replayed, it is not critical state.

**新增**

- ✨ **Append-only Event Log**：`~/.engram/events/events-YYYYMMDD.jsonl`，fsync 写入，按天滚动；Tier 1（task / checkpoint / session）写入路径全程经过日志。
- ✨ **Replay-based Recovery**：DuckDB 缺失/损坏时可从 event log 完整重建 Tier 1。
- ✨ **CLI**：`engram-setup doctor`（健康检查）、`engram-setup recover [--since YYYYMMDD] [--promote]`（dry-run 重建）。
- ✨ **`engram_meta` 表**：暴露 `schema_version` / `engram_version` / `duckdb_version` / `embedding_model` / `embedding_dim` / `embedding_stale` / `last_boot_at`，供 MCP 客户端读取做版本协商。
- ✨ **Readonly Degraded Mode**：DB 不可写时进入只读模式，写操作抛 `DegradedModeError`；HTTP 返回 503 + `recover_command`，MCP 返回 `{ok: false, code: "degraded_mode", recover_command: "engram recover"}`。
- ✨ `tasks` 表预留 `parent_task_id` / `retry_of_task_id` 列（暂不实现，避免未来破坏性迁移）。
- ✨ `/health` 增加 `db_readonly` / `embedding_stale` / `residue_files` / `engram_meta` 字段。

**⚠️ 行为变更（Breaking-ish）**

- **DuckDB 损坏不再静默重建空库**：原来的 `os.replace(db, db + ".corrupt")` + 自动建空库逻辑被移除。损坏时抛 `DatabaseCorruptionError`，原文件以 `<db>.corrupt.<timestamp>` 隔离到 `~/.engram/backups/`，由用户显式 `engram-setup recover` 处理。
  - 想保留旧行为：`ENGRAM_ALLOW_RESET=1 engram-server run`
- **Embedding 模型/维度变化不再自动 ALTER 列**：原来的"全表清零 + ALTER COLUMN"被移除，改为标记 `embedding_stale=true`，向量检索自动 fallback 到 BM25/FTS，写入路径不阻断。
- **WAL 启动恢复路径改进**：先尝试 `FORCE CHECKPOINT` 抢救数据，失败才将 WAL 隔离为 `<db>.wal-recovery.<timestamp>`（带时间戳，永不互相覆盖）。
- **Shutdown 自动 CHECKPOINT**：HTTP server 关闭时主动 flush WAL，避免下次启动有残留。

**升级路径**

```bash
pip install -U mcp-engram          # 安装命令不变
engram-setup doctor                # 升级后建议跑一次健康检查
```

- 现有 `~/.engram/memories.duckdb` 直接复用，schema 自动 `ALTER ... ADD COLUMN IF NOT EXISTS`。
- Event log 从此刻起累积；升级前写入的数据仍依赖 DB 文件本身（无 event log 可 replay）。
- MCP client 配置不需要改。

**回归**

- 348 tests passed（新增 17 个：`test_event_log` / `test_recover` / `test_degraded_mode`）。
- 0 lint error。

### v0.9.x（历史）

- Checkpoint v2 — 版本化 cognitive checkpoint，event-first 触发（6 类 reason），`restore_checkpoint` / `list_checkpoints` 上线。
- Task 一等实体；Session Lifecycle；Handoff Validation；Memory Quality Score；Error-aware Memory。

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