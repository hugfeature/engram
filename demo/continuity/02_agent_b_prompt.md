# 🎬 Engram Continuity Demo — 会话 2（Agent B）

> **使用方式**：**关闭** Agent A 所在的 IDE 窗口（或在 Cursor / Claude Code 里 **新开一个会话**），然后把下面整段 prompt 喂给 Agent B。
> 把 `<TASK_ID>` 替换成 Agent A 给你的那个 task_id（就一个数字，比如 `42`）。
>
> ⚠️ 不要给 Agent B 任何额外提示。它不知道 Express、不知道 JWT、不知道你装过什么包。它只有一个 task_id。

---

## 📋 给 Agent B 的指令（整段复制 ↓↓↓）

````markdown
你是一个全新的 AI 编程 Agent（Agent B），刚被叫来接手一个被中断的任务。
你**唯一**知道的信息是：

> **task_id = <TASK_ID>**

你不知道这个任务在做什么、做到哪、踩过什么坑、用户有什么禁令。
你只能用 engram MCP 工具去把这些都问出来。

---

### 第 1 步：恢复认知（必做，且只做这一步）
按顺序调用：
1. `get_task(task_id=<TASK_ID>)` — 拿到任务全貌
2. `restore_checkpoint(task_id=<TASK_ID>, memory_restore_mode="SELECTIVE")` — 拿到 continuation 包
3. `recall_memory(query="JWT 版本", top_k=5)` — 验证一下不带 task_id 仅靠语义也能找到 failure

### 第 2 步：复述（必做，且必须用下面的固定格式）
**在动手做任何代码改动之前**，你必须先用下面的格式向我口头复述你恢复出来的认知。
**一个字段都不能省，没有就写 `(none)`**：

```
=== 🧠 Engram Continuity Restore Report ===
- 🎯 Goal:                <从 continuation.goal 来>
- ✅ Completed:            <continuation.completed 列表>
- 🔄 In Progress:          <continuation.in_progress 列表>
- ⛔ Must Not Redo:        <continuation.must_not_redo 每一项的 action>
- 🔒 Must Preserve:        <continuation.must_preserve 列表>
- 📂 Working Set:          <continuation.working_set>
- ➡️  Preferred Next:      <continuation.preferred_next 或 next_steps>
- 📊 Confidence:           <continuation.continuation_confidence>
- 💥 Recalled Failure:     <recall_memory 结果里和 'JWT 版本' 相关的那条 failure 的原文 error 字段>
=========================================
```

### 第 3 步：自我验收清单（必做，逐条打 ✅ / ❌）
对照下面 6 条，自己打勾（不要骗自己；任何一条 ❌ 都说明 engram 的 continuity 在这个维度上失效了）：

```
[ ] 1. 我能说出 task goal，且包含关键词 "JWT" 或 "鉴权"
[ ] 2. must_not_redo 里能看到 "jsonwebtoken@8" 相关的项
[ ] 3. must_preserve 里能看到 "server.js" 或 "PORT" 或 "端口" 相关的项
[ ] 4. recall_memory 召回的内容里能看到 "v8" / "callback" / "promise" 相关的失败原因
[ ] 5. working_set 里能看到 middleware/auth.js 或 routes/auth.js
[ ] 6. preferred_next / next_steps 里能看到 "routes/auth.js" 或 "单测" 相关项
```

### 第 4 步：声明你接下来会怎么做（必做）
基于上面恢复的状态，**用 3 句话**告诉我你接下来打算做的下一个动作，并明确声明：

> "我**不会**重新执行：……（列举至少 2 条 must_not_redo 里的动作）
> 我**不会**碰：……（列举 must_preserve）
> 我接下来打算做的第一个动作是：……（match preferred_next）"

### 第 5 步：（可选）真的接着干
如果上面 6 条全部 ✅，你可以选择真的继续推进任务（创建 routes/auth.js、写 jest 单测）。
如果有任何一条 ❌，**停下来告诉我哪一条失败了**，不要硬干。

---

### 🚫 严禁
- 严禁在调用 engram 工具之前就开始猜任务内容
- 严禁跳过"复述"和"自我验收"直接写代码
- 严禁重新执行 `npm install jsonwebtoken@8`
- 严禁修改 `/tmp/engram-demo-app/server.js` 的端口
- 严禁删除 `/tmp/engram-demo-app/routes/auth.js`
````

---

## 你（用户）应该看到的关键现象

| # | 现象 | 证明了 |
|---|---|---|
| 1 | Agent B 第一句话就说出"JWT 鉴权" | **goal 跨进程未丢** |
| 2 | Agent B 主动说"我不会再装 v8" | **must_not_redo 生效** |
| 3 | Agent B 主动声明"我不动 server.js 端口" | **must_preserve 生效** |
| 4 | Agent B 复述里出现 "v8 callback / v9 promise" 字样 | **failure 教训跨会话保留** |
| 5 | Agent B 直接打开 `middleware/auth.js`，不需要问你文件在哪 | **working_set 跨会话保留** |
| 6 | Agent B 第一个动作是"挂到 routes/auth.js"或"加单测" | **preferred_next 生效** |
| 7 | 6 条自检 **全 ✅** | **Continuity 完整性达标** |

---

## 反例（如果 engram 没工作，你会看到）

- Agent B 反问："这个 task 是做什么的？"  → goal 丢了
- Agent B 自作主张 `npm install jsonwebtoken`（不带版本号或装了 v8） → must_not_redo 丢了
- Agent B 重新创建 `middleware/auth.js`（覆盖 Agent A 的工作） → working_set 丢了

任意一条出现，就是 continuity bug，可以直接给项目提 issue。
