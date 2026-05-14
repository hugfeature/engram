# Engram

**AI 编码 Agent 的运行时连续性与中断恢复**

[![PyPI](https://img.shields.io/pypi/v/mcp-engram)](https://pypi.org/project/mcp-engram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-456%20passed-brightgreen)](https://github.com/hugfeature/engram)

> Engram 让 Claude Code、Cursor、OpenHands 拥有**可恢复执行**能力 ——
> 上下文坍塌、中断、会话终止后，恢复 Agent 的工作状态，接着做，不从零开始。

[English](./README.md) · [PyPI](https://pypi.org/project/mcp-engram/) · [Glama](https://glama.ai/mcp/servers/hugfeature/engram)

---

## 问题在哪里

AI Agent 的每次会话都是一座孤岛：

| 场景                          | 没有 Engram                        | 有 Engram                                    |
| ----------------------------- | ---------------------------------- | -------------------------------------------- |
| 上下文窗口满了                | 丢失历史，继续猜                   | Checkpoint 恢复，约束完整保留                |
| 进程被杀 / 服务重启           | 所有工作状态丢失                   | SIGTERM 自动存档，下次 recall 即可恢复       |
| 切换 Agent（Claude → Cursor） | 从头开始                           | 结构化交接，任务状态完整                     |
| 同一个 Bug 踩两遍             | 没有失败记录                       | failure 记忆自动浮现                         |
| 任务跨了三次会话              | 没人知道整体进度                   | 任务状态 + continuity 指标随时可查           |

---

## 快速开始

```bash
pip install mcp-engram
engram-setup          # 下载嵌入模型 + 初始化 DuckDB（约 500MB，一次性）
```

添加到 Claude Code / Cursor 配置：

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

完全本地运行，无需云端、无需 API Key、无遥测。

---

## 中断恢复

当会话意外终止——上下文溢出、进程被杀、IDE 崩溃——Engram 会在关闭时自动捕获结构化工作状态：

```
SIGTERM / 上下文坍塌
        ↓
Engram 自动存档
  goal · completed · in_progress · blocked
  modified_files (git diff) · last_tool_called · last_failure
        ↓
Agent 重启 → recall_memory()
        ↓
interrupt_recovery 注入到响应中
  → "调用 restore_checkpoint(task_id=X) 恢复"
```

无需手动干预。下次 Agent 会话首次调用 `recall_memory` 即自动获得恢复提示。

**根据当前状态生成 CLAUDE.md 片段：**

```bash
engram-prompt
```

输出（粘贴到项目的 `CLAUDE.md`）：

```markdown
## Engram Runtime State

**Active task:** 10 — 修复 login.py 的认证 Bug
**Checkpoint:** v3 (confidence: 0.71)

**已完成（不要重做）：**

- 在 test_auth.py 中复现了 Bug

**进行中：**

- 追踪 JWT 校验路径

**上次会话修改的文件：**

- `src/auth/validator.py`
- `tests/test_auth.py`

## Engram Session Rules

- 会话开始：recall_memory(query) — 中断状态自动置顶
- 恢复任务：restore_checkpoint(task_id=10)
- 上下文快满：report_interruption(reason="overflow") 然后 session_handoff(...)
```

---

## 工作原理

Engram 是一个本地 [MCP](https://modelcontextprotocol.io/) Server，提供 **18 个工具**，覆盖三个领域：

### 记忆（Memory）

混合语义检索（BM25 + 向量 + 图谱）+ 艾宾浩斯衰减，重要记忆保留更久。

| 工具                 | 作用                               |
| -------------------- | ---------------------------------- |
| `store_memory`       | 保存事实、决策或经验（自动去重）   |
| `recall_memory`      | 混合检索 — 每次任务开始时调用      |
| `update_memory`      | 按 ID 修正特定记忆                 |
| `consolidate_memory` | 合并相似记忆，裁剪弱记忆           |
| `memory_stats`       | 健康概览：数量、强度、上次维护时间 |
| `get_runtime_health` | 只读后端健康报告                   |

### 任务追踪（Task Tracking）

Task 是一等实体，跨会话持久存在。

| 工具             | 作用                                |
| ---------------- | ----------------------------------- |
| `create_task`    | 创建跨会话追踪任务                  |
| `get_task`       | 完整任务上下文 + 最新 checkpoint    |
| `update_task`    | 更新状态、目标或元数据              |
| `list_tasks`     | 查看所有任务，按状态过滤            |
| `track_progress` | 记录功能/任务进度快照               |
| `track_failure`  | 结构化记录失败（含根因 + 修复方案） |

### 连续性（Continuity）

基于 Checkpoint 的恢复机制，新 Agent 可以精确接手上一个 Agent 的工作。

| 工具                  | 作用                                           |
| --------------------- | ---------------------------------------------- |
| `session_handoff`     | 会话结束时记录结构化摘要                       |
| `session_outcome`     | 标记会话成功/失败（调整记忆权重）              |
| `restore_checkpoint`  | 获取接管用的约束续接包                         |
| `list_checkpoints`    | 查看任务的 checkpoint 历史                     |
| `report_interruption` | 退出前上报中断原因（overflow / rate_limit 等） |
| `evaluate_continuity` | 量化跨 checkpoint 的恢复质量                   |

---

## 断点恢复流程

```
Agent A（Claude Code）                  Agent B（Cursor）
  │                                        │
  ├─ create_task(name, goal)               │
  ├─ track_progress / track_failure        │
  ├─ ⚡ 中断（SIGTERM / 上下文溢出）      │
  ├─ [自动] 中断 checkpoint 已保存        │
  │        │                               │
  │   ┌────▼────────────────┐              │
  │   │  Engram Checkpoint  │              │
  │   │  goal               │              │
  │   │  completed          │              │
  │   │  must_not_redo  ────┼──────────▶   │
  │   │  modified_files     │             ├─ recall_memory() → interrupt_recovery 置顶
  │   │  working_set        │             ├─ restore_checkpoint(task_id)
  │   └─────────────────────┘             ├─ 接着做，不从零开始
  │                                       └─ session_handoff → 下一个 Agent...
```

**续接包字段**：`goal` · `completed` · `in_progress` · `blocked` · `preferred_next` · `must_not_redo`（负记忆）· `must_preserve`（不变量）· `working_set` · `continuation_confidence`

---

## 推荐写入 CLAUDE.md 的 Agent 指令

运行 `engram-prompt` 自动根据当前状态生成，或手动粘贴：

```markdown
## Engram 使用规则

- 多步任务起点：create_task(name, goal) → 保存 task_id
- 任务开始：recall_memory(query) — 最新 handoff + 中断状态自动置顶
- 接手任务：restore_checkpoint(task_id, memory_restore_mode="SELECTIVE")
- 阶段进展：track_progress(feature, status, task_id=<id>)
- 遇到错误：track_failure(error, component, root_cause, task_id=<id>)
- 上下文快满：report_interruption(reason="overflow") 然后 session_handoff(...)
- 任务收尾：session_handoff(summary, completed, in_progress, blocked, task_id=<id>)
```

---

## 架构设计

```
Tier 1 — Runtime Continuity（唯一可信源，可 replay 恢复）
  tasks · checkpoints · session lifecycle · handoff events
  → append-only 事件日志  ~/.engram/events/*.jsonl  (fsync，自动 gzip 轮转)

Tier 2 — Semantic Recall（可降级，只读可恢复）
  memories · metadata · 语义图
  → DuckDB 从事件日志投影

Tier 3 — Retrieval Cache（可丢弃，可重建）
  embeddings · FTS · 向量索引
  → 按需重建，不参与恢复路径
```

**两条铁律：**

1. 事件日志是唯一的持久化原语。
2. 不能 replay 的，就不是关键状态。

DB 损坏时进入 **readonly degraded mode**，永远不会静默重置。用 `engram-setup recover` 显式恢复。

---

## Benchmark

基于 [LoCoMo](https://github.com/snap-research/locomo)（Snap Research 长期对话记忆基准）评测：

| 系统       | Overall F1 | Hit@5     | LLM           | 部署方式 |
| ---------- | ---------- | --------- | ------------- | -------- |
| MemMachine | 0.8487     | —         | GPT-4o-mini   | 云端     |
| Memobase   | 0.7578     | —         | GPT-4o-mini   | 云端     |
| Zep        | 0.7514     | —         | GPT-4o-mini   | 云端     |
| Mem0       | 0.6688     | —         | GPT-4o-mini   | 云端     |
| **Engram** | **0.4383** | **77.7%** | DeepSeek-V3.2 | **本地** |

零云端依赖，四轮优化累计 **F1 +50.3%**，**Hit@5 +26.2pp**。

Engram 还通过 `evaluate_continuity` 追踪 **运行时连续性指标**：目标保持度 · 行动一致性 · 失败记忆召回率 · 工作集稳定度 · 重规划率 · 冗余探索率。

<details>
<summary>分类得分 + 记忆算法详解</summary>

### 分类得分

| Category    | Count   | F1         | Hit@5     |
| ----------- | ------- | ---------- | --------- |
| Single-Hop  | 114     | 0.5121     | 76.3%     |
| Temporal    | 63      | 0.4501     | 95.2%     |
| Multi-Hop   | 43      | 0.3181     | 60.5%     |
| Open-Domain | 13      | 0.1324     | 61.5%     |
| **Overall** | **233** | **0.4383** | **77.7%** |

### 记忆算法

- **艾宾浩斯衰减**：`strength = importance × e^(−λ × days) × (1 + recall_count × 0.2)`
  - `failure`：λ=0.35，半衰期约 11 天 · `strategy`：λ=0.10，半衰期约 38 天
- **去重**：≥0.85 → 强化 · 0.65–0.84 → 矛盾检测后合并/覆盖 · <0.65 → 新建
- **混合检索**：`0.3 × BM25 + 0.7 × (语义相似度 × 衰减强度) + 图谱加成`
- **自动维护**：每 12h 整合（≥0.70 聚类合并）+ 剪枝（strength < 0.05）+ FTS 重建

### 重要性参考

| 范围    | 适用场景           |
| ------- | ------------------ |
| 0.9–1.0 | 核心身份、永久事实 |
| 0.7–0.8 | 架构决策、强偏好   |
| 0.5     | 普通项目事实       |
| 0.2–0.3 | 临时会话上下文     |

### 常用环境变量

| 变量                                | 默认                    | 说明                         |
| ----------------------------------- | ----------------------- | ---------------------------- |
| `HF_ENDPOINT`                       | `https://hf-mirror.com` | HuggingFace 镜像（国内推荐） |
| `ENGRAM_MODEL`                      | `all-mpnet-base-v2`     | 嵌入模型                     |
| `ENGRAM_DEDUP_THRESHOLD`            | `0.65`                  | 去重相似度下限               |
| `ENGRAM_REINFORCE_THRESHOLD`        | `0.85`                  | 强化相似度阈值               |
| `ENGRAM_W_BM25` / `ENGRAM_W_VECTOR` | `0.30` / `0.70`         | 检索权重                     |
| `ENGRAM_PRUNE_THRESHOLD`            | `0.05`                  | 剪枝强度阈值                 |

完整变量列表：`src/engram/config.py`

</details>

---

## Engram 不做什么

- ❌ 保证恢复后 LLM 行为完全一致（LLM 非确定性是物理限制）
- ❌ 自定义 agent loop / prompt 编排（由 MCP 客户端负责）
- ❌ 多 Agent 协调或共享团队记忆（单用户、本地优先）

---

## 环境要求

- macOS / Linux / WSL2
- Python 3.11+
- 约 500MB 磁盘空间（嵌入模型，一次性下载）

---

## 参与贡献

```bash
git clone https://github.com/hugfeature/engram.git
cd engram
pip install -e ".[dev]"
pytest tests/ -v
```

欢迎提交 Issue 和 PR。

## License

[MIT](https://opensource.org/licenses/MIT) · 维护者 [@hugfeature](https://github.com/hugfeature)

---

> _Engram 恢复的是 Agent 的工作状态，而不仅是记忆。_
