# Engram

**AI 编码 Agent 的运行时连续性与中断恢复**

[![PyPI](https://img.shields.io/pypi/v/mcp-engram)](https://pypi.org/project/mcp-engram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-545%20passed-brightgreen)](https://github.com/hugfeature/engram)

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

## 为什么是 Engram

大多数记忆系统关注对话回忆。

Engram 关注运行时连续性：

- **Agent 在做什么** — 任务状态、工作集、修改的文件
- **执行停在哪里** — 包含目标、进度、阻塞的 checkpoint
- **如何安全恢复** — 负记忆（must_not_redo）+ 不变量（must_preserve）

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
Tier 1 — Event Journal（不可变，仅追加）
  唯一可信源 · 快照压缩 · 增量重放
  → ~/.engram/events/*.jsonl  (fsync，自动 gzip 轮转)
  → 定期快照，加速启动

Tier 2 — Runtime State Store（运行时持久）
  tasks · checkpoints · executions · sessions
  → SQLite WAL  ~/.engram/*.state.sqlite
  → 并发读取、快速恢复、恢复链路关键
  → 通过 ENGRAM_SQLITE_TIER2=1 启用（默认回退到 DuckDB）

Tier 3 — Runtime Intelligence Cache（可重建）
  memories · embeddings · FTS · 向量索引
  未来：drift vectors · recovery metrics · tool stats · continuity history
  → DuckDB（可丢弃并从 Tier 1 + 2 重建，无数据丢失）
  → 重建命令：engram-setup rebuild-cache
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
LoCoMo 衡量的是**检索**质量。但 Engram 真正的职责是**恢复**——熬过一次中断、把干净的状态交给下一个 agent。这需要专门的基准。

### 连续性基准（Core）

LoCoMo 告诉你记忆是否**找得到**,但不告诉你一次**恢复**好不好。跨会话连续性没有现成的标准基准,所以 Engram 自带一套。

**它回答的问题:** 给定一次中断,继续包（continuation package）能否让下一个 agent 在**不重做被禁止的工作**的前提下接手——而且选择性恢复是否真的比把全量历史灌回去更好?

```bash
python benchmark/continuity_bench.py --mode all --runs 5 --seed 42
```

**20 个场景 × 3 种恢复模式 × 5 次 = 300 次评测。** 完全确定性（std = 0）。

| 恢复模式      | 综合分 | 冗余探索 | 是什么 |
| ------------- | ------ | -------- | ------ |
| `NONE`        | 0.282  | 0.20     | 空包(失忆基线) |
| **`SELECTIVE`** | **0.912** | **1.00** | 真实 Engram 恢复(importance ≥ 0.5 + 失败记忆) |
| `FULL`        | 0.791  | 0.52     | 全量任务历史(上下文污染基线) |

**`SELECTIVE` 胜过 `FULL`** —— 独立复现了 Letta [Recovery-Bench](https://www.letta.com/blog/recovery-bench) 的上下文污染结论(其 `--message-mode full/summary/none` 正对应 Engram 的 `memory_restore_mode`)。两个真实恢复模式保存的**结构状态完全相同**,差距全部落在**冗余探索**上——`FULL` 把过期的失败路径重新喂回去,agent 又走了一遍。

场景覆盖三轴:**A — 中断类型**(SIGTERM、崩溃、上下文溢出、工具超时、网络故障、会话重启、长空闲、跨天暂停)、**B — 状态漂移**(目标突变、分支切换、依赖升级 scope 膨胀、工具权限变更、工作区清空)、**C — 失败召回**(重试风暴、人类交接、记忆污染、planner 重启)。

#### 给评测者 —— 诚实的边界

这是 **Core** 基准,评的是**继续包本身**,不是真实 agent 行为:

- **脚本化动作。** 恢复后的 agent 动作是每个场景声明好的（`agent_replay`）,不是真 LLM 跑出来的。Core 衡量的是「在假设的动作序列下,这个包能否让 agent 避免重做禁止项」。
- **结构指标是回归守卫。** `build_continuation` 在记忆模式分流之前执行,所以三档的目标/已完成/工作集完全相同(~1.0)。它们证明包忠实保存了 checkpoint,不是区分三档的判别器。
- **`redundant_exploration` 才是判别器**,由失败记忆是否被召回(真实、随模式变化)+ 脚本动作共同驱动。
- **只有 `SELECTIVE` 是真实的 `restore_checkpoint`。** `NONE`/`FULL` 是构造基线。`observed.related_memories` 列记录每档**真实**召回了多少——Core 中唯一完全真实的信号。
- 真正的因果问题(*真实 agent 是否真的少走弯路?*)是 **Live 基准(v2)**,刻意分开,让 Core 保持确定性、可进 CI。

不变式 `SELECTIVE > FULL > NONE` 由 CI 守护(`tests/test_continuity_bench.py`)——任何让 Engram 恢复质量退化的改动都会让构建失败。

Core 综合分权重(独立于工具内指标):目标 0.20 · 已完成 0.20 · 工作集 0.15 · 失败上下文 0.20 · **冗余 0.25**(最重——最贴近「少走弯路」)。运行时 `evaluate_continuity` 工具暴露相关的 6 维评分:目标保持度 · 行动一致性 · 失败记忆召回率 · 工作集稳定度 · 重规划率 · 冗余探索率。

### 连续性基准 —— Live（v2）

Core 在*脚本化*动作下打分。**Live** 闭合因果链:真实 LLM 仅凭**继续包**接手每个被中断的任务(它看不到禁止项清单),提出下一步动作,再由一个**独立的 judge** 对照 ground truth 打标。


- **Actor / judge 分离。** actor 只拿到渲染后的包;judge 拿到 actor 的动作 + ground truth（`forbidden_actions`、`must_preserve`）,标记每个重做 / 约束违反。同模型,不同 system prompt。
- **待验证的假设:** 拿到 `FULL` 全量历史的真实 LLM,是否会被*勾引去重走*过期的失败路径,而 `SELECTIVE` 让它干净地恢复?
- 设计上非确定性:报告 `--runs` 的均值 ± 标准差,**不**进 CI。复用 Core 场景数据集,Core 与 Live 直接可比。
- `--dry-run` 跑通完整流程(真实 `restore_checkpoint`、prompt 渲染、judge 流水线)但不花 API 调用——用于离线验证 harness。

#### 结果（3 个按轴代表的场景 × 3 模式 × 3 次）

`redundant_exploration`(越高 = 越少重做禁止项):

| 模型          | NONE  | SELECTIVE | FULL  |
| ------------- | ----- | --------- | ----- |
| DeepSeek-V3.2 | 0.956 | 1.000     | 1.000 |
| GLM-5.1       | 1.000 | 1.000     | 1.000 |

三档的真实召回数全程正确（NONE = 0、SELECTIVE ≈ 2.7、FULL = 3）。

**诚实结论 —— Live *没有*复现 Core 的 `SELECTIVE > FULL` 差距,而这正是结果本身。** 两层解读:

1. **一个弱但真实的因果信号。** 唯一低于 1.0 的是 DeepSeek 在*重试风暴*场景的 `NONE` 档(0.956):没有恢复包时,模型确实偶尔提出了接近已知坏路径的动作——这是脚本化的 Core 基准无法证明的。有包时则从不发生。
2. **强模型能抵抗上下文污染。** Core 的 `SELECTIVE > FULL`(0.912 vs 0.791）来自*脚本化* agent——它会重走包里浮现的任何东西。真实前沿模型会自己过滤 `FULL` 历史,于是污染差距消失。Recovery-Bench 报告的效应需要**更弱的模型、更长的历史、或更刁钻的禁止项设计**才能暴露——这*正是* Recovery-Bench 用弱模型制造失败的原因。

这把两个基准的边界划得很清楚:**Core 衡量包质量(与模型无关);Live 衡量真实 agent 行为(受模型能力干扰)。** Live 的负结果不削弱 Core——它界定了各自能下的结论范围。

> Live harness 还逼出了一个真 bug:GLM-5.1 的 judge 返回的是裸 JSON 数组而非 `{"verdicts": [...]}`,导致首次运行崩溃。已修复(`normalize_judge_output` 兼容四种输出形状)并在 `tests/test_continuity_bench_live.py` 中加了回归守卫。

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
| `ENGRAM_SQLITE_TIER2`               | _（未启用）_            | 设为 `1` 启用 SQLite WAL 运行时状态存储（Tier 2） |

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
