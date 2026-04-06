# Agent vs Agent2 对比

## 核心区别

| 特性 | agent (原版) | agent2 (增强版) |
|------|-------------|----------------|
| **工具系统** | 基础工具类 | 增强的 BaseTool 抽象类 |
| **权限检查** | ❌ 无 | ✅ ToolPermission 枚举 |
| **并发控制** | ❌ 无 | ✅ 防止同名工具并行 |
| **进度报告** | ❌ 无 | ✅ on_progress 回调 |
| **记忆管理** | 简单滑动窗口 | Token估算 + 智能修剪 |
| **多Agent** | CoordinatorAgent | MultiAgentCoordinator |
| **结果截断** | 固定长度 | 按角色差异化截断 |
| **Token估算** | ❌ 无 | ✅ 自动估算 |

## 工具对比

### agent (原版) - 11个工具
- search
- read_file
- find_pattern
- list_dir
- read_multiple
- write_file
- search_replace
- delete_file
- shell
- git
- python_run

### agent2 (增强版) - 7个精选工具
- search (查询扩展)
- read_file (智能截断)
- find_pattern (ReDoS保护)
- list_dir
- write_file (自动备份)
- search_replace (安全替换)
- shell (危险命令检测)

## 记忆系统对比

### agent
```python
class ShortTermMemory:
    def __init__(self, max_entries=20):
        self.max_entries = max_entries  # 只限制条目数
```

### agent2
```python
class ShortTermMemory:
    def __init__(self, max_entries=20, max_tokens=30000):
        self.max_entries = max_entries
        self.max_tokens = max_tokens  # 同时限制Token数
        # 自动Token估算
```

## 使用对比

### agent
```bash
snowcode agent "解释项目架构"
snowcode agent --coordinator --workers 2 "分析安全漏洞"
```

### agent2
```bash
snowcode agent2 "解释项目架构"
snowcode agent2 --multi-agent --workers 2 "分析安全漏洞"
```

## 输出对比

### agent 输出
```
Step 1 → search
  结果: ...

Step 2 → read_file
  结果: ...

Answer:
...
```

### agent2 输出
```
  LLM: gpt-4 | Steps: 5 | Planning: on
  Tools: search, read_file, find_pattern, ...

  Step 1 → search
  结果预览: ...

  Step 2 → read_file
  结果预览: ...

  ╭──────────────── Answer ─────────────────╮
  │ ...                                     │
  ╰─────────────────────────────────────────╯

  Execution Summary
  Steps: 2
  Tools used: search, read_file
  Time: 8500ms
```

## 何时使用哪个？

| 场景 | 推荐 |
|------|------|
| 快速简单查询 | agent2 (更快响应) |
| 需要 git 操作 | agent (有git工具) |
| 需要运行Python | agent (有python_run工具) |
| 复杂代码分析 | agent2 (更好的记忆管理) |
| 安全敏感操作 | agent2 (权限检查) |
| 多Agent协作 | agent2 (更完善的实现) |
