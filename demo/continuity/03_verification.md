# 🧪 Engram Continuity Demo — 验收清单

> 演完两个会话后，对照这份清单确认结果。可以肉眼看，也可以让 Agent B 自检。

---

## A. 物理证据（命令行验证，5 秒）

```bash
# 1. Agent A 真的写了文件
ls -la /tmp/engram-demo-app/middleware/auth.js

# 2. 装的是 v9（Agent A 升级生效，且 Agent B 没回退到 v8）
cat /tmp/engram-demo-app/node_modules/jsonwebtoken/package.json | grep '"version"'

# 3. engram 数据库刚被写过
ls -la ~/.engram/memories.duckdb

# 4. checkpoint 真的产生了（应至少有 3 个：FAILURE / PLAN_UPDATE / MANUAL_HANDOFF）
python3 -c "
import duckdb, os
db = duckdb.connect(os.path.expanduser('~/.engram/memories.duckdb'), read_only=True)
print(db.execute('SELECT version, reason, kind, created_at FROM checkpoints ORDER BY id DESC LIMIT 10').fetchall())
"
```

预期输出：
- `auth.js` 存在
- jsonwebtoken version 是 `9.x.x`
- duckdb mtime 是刚才
- checkpoints 表里能看到 `MANUAL_HANDOFF`、`FAILURE`、`PLAN_UPDATE` 至少各一行

---

## B. 行为证据（看 Agent B 的输出）

| # | 维度 | 通过标准 | Agent B 表现 |
|---|---|---|:-:|
| 1 | **Goal Retention** | 第一句话出现 "JWT" / "鉴权" | ☐ |
| 2 | **Action Avoidance** | 明确声明"不会装 jsonwebtoken@8" | ☐ |
| 3 | **Invariant Respect** | 明确声明"不会动 server.js / PORT" | ☐ |
| 4 | **Failure Recall** | 复述里出现 "v8" + "callback" / "promise" / "async" | ☐ |
| 5 | **Working Set** | 直接引用 `middleware/auth.js` 路径，不问 | ☐ |
| 6 | **Next Step Match** | 下一个动作是 routes/auth.js 或 jest 单测 | ☐ |
| 7 | **Self-check** | 自检 6 条全 ✅ | ☐ |

**6/7 通过 = continuity 基本工作；7/7 = 完整通过。**

---

## C. 反向对照（强烈推荐做一次）

为了证明这些表现**确实是 engram 的功劳**，而不是 Claude/Cursor 自己脑补出来的，可以再开一个会话做对照：

```
新开第三个会话，对 Agent C 说：

  "请帮我接手任务，task_id 是 99999。"
  （故意给一个不存在的 ID）
```

如果 engram 真在起作用：
- Agent C 调 `get_task(99999)` 会返回空 / 报错
- Agent C 应该如实告诉你"找不到这个任务"
- Agent C **不应该**编出"JWT 鉴权"、"server.js 端口"、"jsonwebtoken v8 失败" 这些细节

如果 Agent C 也能凭空说出这些细节，那就是大模型在脑补，不是 engram 在工作。

---

## D. 清理

```bash
rm -rf /tmp/engram-demo-app
# 如果想把 demo 期间的记忆也清掉（注意会清掉 default 用户的所有数据，慎用）：
# rm ~/.engram/memories.duckdb ~/.engram/graph.json
```

---

## 一句话结论

> 如果 B 表 ≥ 6/7，C 对照不脑补，且 A 表全过——
> 你刚刚验证了 engram 最核心的卖点：**Agent 可替换，任务不中断。**
