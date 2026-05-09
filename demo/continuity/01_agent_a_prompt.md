# 🎬 Engram Continuity Demo — 会话 1（Agent A）

> **使用方式**：在已经配好 `engram` MCP 的 Claude Code / Cursor 里**新开一个会话**，把下面**整段「给 Agent 的指令」**复制粘贴进去发送即可。
> Agent A 会按剧本演完前 5 步，然后**主动停下并提示你关闭 IDE 窗口**，模拟真实中断。

---

## 演示目标

让你亲眼看到：
- Agent A 把任务状态、失败教训、用户禁令、工作集，全部交给 engram 保管；
- 你**直接关掉整个 IDE 窗口**之后，会话 2 里一个全新的 Agent B 仍能"接着做"。

## 前置准备（30 秒）

在当前目录下随便建一个空文件夹当"项目"：
```bash
mkdir -p /tmp/engram-demo-app && cd /tmp/engram-demo-app && npm init -y > /dev/null && echo '{}' > package.json
```
然后把下面的 prompt 喂给 Agent A。

---

## 📋 给 Agent A 的指令（整段复制 ↓↓↓）

````markdown
你现在要配合我演示 engram MCP 的 "Cognitive Continuity"（任务不中断）能力。
你扮演 **Agent A**，按下面剧本严格执行。**每一步都必须真实调用对应的 engram 工具**，不要只是嘴上说。

工作目录：`/tmp/engram-demo-app`
任务背景：给一个 Express 项目加 JWT 鉴权中间件。

---

### Step 0：探活（必做）
先调用 `memory_stats`，确认 engram MCP 通了。如果失败，停下来告诉我"engram MCP 没连上，请先配置"，不要继续。

### Step 1：建任务
调用 `create_task`：
- name: `add-jwt-auth`
- goal: `给 /tmp/engram-demo-app 这个 Express 项目加 JWT 鉴权中间件`
- status: `in_progress`

把返回的 `task_id` **加粗显示给我看**，后面所有工具调用都要带上这个 task_id。

### Step 2：声明用户禁令（must_preserve 的预埋）
口头告诉我："收到，我会遵守一条硬约束：**不要修改 server.js 里的端口配置（PORT=3000）**。"
然后调用 `track_progress`：
- task_id: <上面的 task_id>
- feature: `add-jwt-auth`
- status: `in_progress`
- completion: 10
- notes: `用户硬约束：禁止修改 server.js 的端口配置 PORT=3000`

### Step 3：真实写代码 + 装包
1. 在 `/tmp/engram-demo-app` 下真实创建 `middleware/auth.js`，内容是一个用 `jsonwebtoken` 做 verify 的中间件骨架（5~10 行即可）。
2. 真实执行 `npm install jsonwebtoken@8` （**故意装老版本 v8**）。
3. 调用 `track_progress`：
   - task_id: <task_id>
   - feature: `add-jwt-auth`
   - status: `in_progress`
   - completion: 35
   - notes: `已创建 middleware/auth.js，已 npm install jsonwebtoken@8`

### Step 4：踩坑（关键！触发 FAILURE checkpoint）
打开 `middleware/auth.js`，假装你刚发现 v8 的 `jwt.verify` 异步签名跟你预期的 v9 promise 风格不一致。
调用 `track_failure`：
- task_id: <task_id>
- error: `jsonwebtoken@8 的 verify 是 callback-only，与项目其他代码 async/await 风格冲突`
- component: `middleware/auth.js`
- root_cause: `选错了主版本，v8 没有 promise API，v9 才原生支持`
- severity: `major`
- fix: `升级到 jsonwebtoken@9`

### Step 5：改计划（触发 PLAN_UPDATE checkpoint）
1. 真实执行 `npm install jsonwebtoken@9` 升级版本。
2. 调用 `track_progress`：
   - task_id: <task_id>
   - feature: `add-jwt-auth`
   - status: `in_progress`
   - completion: 60
   - notes: `已升级到 jsonwebtoken@9，middleware/auth.js 改用 async/await 风格`

### Step 6：主动交接（触发 MANUAL_HANDOFF checkpoint）
调用 `session_handoff`：
- task_id: <task_id>
- summary: `已完成 JWT 中间件骨架与版本选型，剩下接路由 + 写单测`
- completed: `["创建 middleware/auth.js 骨架", "确定使用 jsonwebtoken@9", "记录 v8 版本不可用的教训"]`
- in_progress: `["把 auth 中间件挂到 routes/auth.js"]`
- blocked: `[]`
- next_steps: `["把 auth 中间件挂到 routes/auth.js", "为 verify 函数加 jest 单测"]`
- must_not_redo: `[{"action": "npm install jsonwebtoken@8", "reason": "failed_dont_retry"}, {"action": "删除 routes/auth.js", "reason": "side_effect_emitted"}]`
- must_preserve: `["不要修改 server.js 的端口配置 PORT=3000"]`
- working_set: `{"files": ["/tmp/engram-demo-app/middleware/auth.js", "/tmp/engram-demo-app/routes/auth.js"]}`

### Step 7：模拟中断（停在这里）
**做完 Step 6 后，立刻停止响应**，并对我说：

> ✅ **Agent A 演完了。** 任务 `task_id=<X>` 的所有认知状态已经存进 engram。
>
> 👉 **现在请你 (用户) 直接关闭这个 IDE 窗口**（或新开一个会话），然后在新窗口里把 `02_agent_b_prompt.md` 喂给 Agent B，把 `task_id=<X>` 替换进去。
>
> 如果 engram 真的工作，Agent B 在不知道任何上下文的情况下，仅凭 task_id 就能接着干、不重装 v8、不动端口、不删 routes/auth.js。

**不要自己续演 Agent B 的剧情，必须停下让我换会话。**
````

---

## 你应该看到的现象（演完 Step 6 时）

- Agent A 至少调用了：`memory_stats`、`create_task`、`track_progress` ×3、`track_failure`、`session_handoff` —— 共 **7 次** engram 工具调用
- `/tmp/engram-demo-app/middleware/auth.js` **真实存在**
- `/tmp/engram-demo-app/node_modules/jsonwebtoken/package.json` 里 version 是 **9.x**
- `~/.engram/memories.duckdb` 文件 mtime 是刚才
- Agent A **明确告诉你停下来换会话**，没有自己往下演

确认这些都满足后，进入 `02_agent_b_prompt.md`。
