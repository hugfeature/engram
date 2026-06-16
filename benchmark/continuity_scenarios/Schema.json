# Continuity Benchmark — Scenario Schema

每个场景是一个声明式 JSON,描述「一次连续性断裂 + 恢复后 Agent 的假设行为」。
Runner 读它 → 真实写入 engram → 真实 checkpoint → 三档 restore → 6 维打分。

## 代码现实(决定了 Core 能诚实测什么)

`build_continuation` 在 mode 判断**之前**执行,所以 goal/completed/in_progress/
must_not_redo/working_set 这些**结构字段三档(NONE/SELECTIVE/FULL)返回完全一样**。
mode 只 gate `related_memories` / `related_failures` 的召回。

因此 Core bench 的诚实分工:
- **结构 4 维**(goal_retention / completed_preservation / working_set_overlap /
  failure_context)= **回归守卫**:证明 continuation 忠实保存了 checkpoint,抓
  序列化/恢复 bug。纯真实 restore 下三档恒等,应 ~1.0。
- **redundant_exploration** = **唯一真正区分三档的指标**:靠 related_failures
  是否被召回(SELECTIVE/FULL 召回、NONE 不召回)+ scripted agent_replay 驱动。
- **observed 信号**(每档真实召回的 related_memories 数 / continuation_confidence)
  = 真实系统行为证据,证明三档在真实 engram 里确实不同。

## 三轴设计(每个场景必须声明属于哪个轴,且说明压测哪条信号)

| axis | 压测信号 | restored_state_by_mode |
|------|----------|------------------------|
| `A_interruption` | interruption 分类 + 状态零损失;结构维趋同~1.0,差异只在 redundant | 缺省(三档忠实) |
| `B_state_drift`  | **脚本注入受控漂移,验证结构指标非死值(有区分度)**——不声称是 mode 导致 | 显式指定 |
| `C_failure_recall` | failure_context + redundant_exploration;related_failures 召回的三档差异 | 部分指定 |

## 字段

```jsonc
{
  "scenario_id": "a1_sigterm",
  "axis": "A_interruption",          // A_interruption | B_state_drift | C_failure_recall
  "description": "人类可读的一句话",
  "stresses": "这个场景专门压测哪条信号(写给评测读者看)",
  "interrupt_reason": "process_exit", // 写进 checkpoint 的中断分类标签

  // ── 第一层:中断前真实状态(会真实写进 engram + create_checkpoint) ──
  "pre_interrupt_state": {
    "goal": "...",
    "completed": ["..."],
    "in_progress": ["..."],
    "blocked": [],
    "preferred_next": ["..."],
    "must_not_redo": [{"action": "...", "reason": "failed_dont_retry"}],
    "must_preserve": ["..."],
    "working_set": {"files": ["..."], "tools": ["..."]}
  },

  // ── 关联记忆(真实写进 engram,驱动 SELECTIVE/FULL 召回差异)──
  "task_memories": [
    {"content": "...", "importance": 0.8, "category": "failure"},
    {"content": "...", "importance": 0.3, "category": "fact"}
  ],

  // ── 第二层:恢复后状态(可选)。
  //    缺省(A轴/C轴):runner 用真实 restore 的忠实 continuation 当 restored_state,
  //      三档结构维恒等~1.0(回归守卫)。
  //    显式指定(B轴):脚本注入受控漂移,验证结构指标有区分度(非死值)。
  //      key 可以是 "ALL"(三档同) 或分档 "NONE"/"SELECTIVE"/"FULL"。
  "restored_state_by_mode": null,

  // ── 第三层:评分参考 ──
  "ground_truth": {
    "expected_goal": "...",           // 恢复后目标应保持
    "must_preserve": ["..."],
    "forbidden_actions": ["..."]      // 恢复后不该重做的(小写匹配 redundant_exploration)
  },

  // ── mode-dependent 假设行为(scripted,Core 的诚实边界:这是假设不是真因果)──
  "agent_replay": {
    "NONE":      ["redo npm install", "inspect unrelated files"],
    "SELECTIVE": ["continue failing test"],
    "FULL":      ["reread long history", "redo npm install"]
  }
}
```

## 诚实边界(必须在报告里声明)

- `agent_replay` 是**假设的** Agent 行为,不是真跑出来的。Core bench 测的是
  "给定这组假设行为,恢复包能否如实反映恢复质量差异"。
- **真实因果(恢复包是否真让 Agent 少走弯路)是 Live bench(v2)的事。**
- 三档中只有 **SELECTIVE 走真实 engram restore_checkpoint**;NONE/FULL 是构造对照。
- runner 会额外记录每档真实召回的 `related_memories` 数量作为 observed 信号。
