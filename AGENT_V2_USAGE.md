# Agent v2 使用指南

## 快速开始

### 1. 基础用法

```python
from codechat.agent_v2 import create_agent
from codechat.store import VectorStore
from pathlib import Path

# 创建向量存储（需要先建立索引）
store = VectorStore(Path("./your_project"))

# 创建增强版 Agent
agent = create_agent(
    store=store,
    project_root=Path("./your_project"),
    model="gpt-4",  # 或其他支持的模型
    max_steps=5,    # 最大执行步数
    use_planning=True  # 启用任务规划
)

# 运行 Agent
result = agent.run("解释一下这个项目的架构")

# 查看结果
print(result.answer)
print(f"执行步数: {result.steps_taken}")
print(f"使用工具: {result.tools_used}")
print(f"耗时: {result.total_elapsed_ms:.0f}ms")
```

### 2. 带进度回调的运行

```python
def on_step(step_num, tool_name, preview):
    """每步执行时的回调"""
    print(f"[Step {step_num}] {tool_name}")
    print(f"  结果预览: {preview[:100]}...")

def on_think(think_text):
    """Agent 思考时的回调"""
    print(f"[思考] {think_text}")

def on_progress(activity):
    """工具执行进度回调"""
    print(f"[进度] {activity}")

result = agent.run(
    question="找出所有 TODO 注释",
    on_step=on_step,
    on_think=on_think,
    on_progress=on_progress
)
```

### 3. 多 Agent 协作

```python
from codechat.agent_v2 import create_coordinator

# 创建 Coordinator（自动分解任务并行执行）
coordinator = create_coordinator(
    store=store,
    project_root=Path("./your_project"),
    model="gpt-4",
    num_workers=2  # 并行 Worker 数量
)

# 运行（自动分解为子任务）
answer = coordinator.coordinate(
    question="分析这个项目的安全漏洞并提供修复建议",
    on_progress=lambda msg: print(f"[{msg}]")
)
```

## 工具系统

### 可用工具列表

| 工具名 | 描述 | 权限 |
|--------|------|------|
| `search` | 语义搜索代码库 | 只读 |
| `read_file` | 读取文件内容 | 只读 |
| `find_pattern` | 正则搜索代码 | 只读 |
| `list_dir` | 列出目录结构 | 只读 |
| `write_file` | 写入文件 | 需确认 |
| `search_replace` | 搜索并替换 | 需确认 |
| `shell` | 执行命令 | 需确认 |

### 权限系统

```python
from codechat.agent_v2 import ToolPermission

# 工具权限级别
ToolPermission.ALLOWED    # 自动允许（只读操作）
ToolPermission.PROMPT     # 需要用户确认（写操作）
ToolPermission.DENIED     # 被策略阻止
ToolPermission.DANGEROUS  # 危险操作（需显式批准）
```

### 自定义工具

```python
from codechat.agent_v2 import BaseTool, ToolExecutionContext

class MyCustomTool(BaseTool):
    """自定义工具示例"""
    
    name = "my_tool"
    description = "我的自定义工具"
    search_hint = "custom operation"
    
    @property
    def parameters(self):
        return {
            "input": "输入参数",
            "option": "可选参数(可选)"
        }
    
    def is_read_only(self) -> bool:
        return False  # 标记为写操作
    
    def check_permission(self, params):
        # 自定义权限检查
        if params.get("option") == "dangerous":
            return ToolPermission.DENIED
        return ToolPermission.PROMPT
    
    def run(self, params: dict, ctx: ToolExecutionContext) -> str:
        # 实现工具逻辑
        input_val = params.get("input", "")
        return f"处理结果: {input_val}"
    
    def get_activity_description(self, params):
        return f"Processing: {params.get('input', '')[:30]}"

# 注册自定义工具
agent.tools.register(MyCustomTool())
```

## 记忆系统

### 短期记忆

```python
from codechat.agent_v2 import ShortTermMemory

# 创建短期记忆（自动管理上下文大小）
memory = ShortTermMemory(
    max_entries=20,    # 最大条目数
    max_tokens=30000   # 最大 Token 数
)

# 添加记忆
memory.add("user", "用户的问题")
memory.add("agent", "Agent 的思考")
memory.add("tool", "工具执行结果", tool_name="search")

# 获取上下文（自动截断）
context = memory.get_context(max_chars=30000)

# 获取最近的工具结果
recent_results = memory.get_recent_tool_results(n=3)
```

### 长期记忆

```python
from codechat.agent_v2 import LongTermMemory
from pathlib import Path

# 创建长期记忆（持久化到 .snowcode/memory.jsonl）
memory = LongTermMemory(Path("./project"))

# 存储问答记录
memory.store(
    question="如何运行测试？",
    answer="使用 pytest 命令",
    actions=[{"tool": "shell", "params": {"command": "pytest"}}]
)

# 检索相似问题
similar = memory.recall("怎么跑测试？", n=3)
print(similar)
```

## 执行计划

### 创建计划

```python
from codechat.agent_v2 import Planner, Plan, PlanStep

# Planner 自动将问题分解为步骤
planner = Planner(agent.llm, agent.tools.list_definitions())
plan = planner.create_plan("分析代码质量并生成报告")

# 查看计划
print(plan.to_context())
# 输出:
# Goal: 分析代码质量并生成报告
#
#   [ ] Step 1: 搜索所有 Python 文件
#   [ ] Step 2: 检查代码风格问题
#   [ ] Step 3: 运行静态分析工具
#   [ ] Step 4: 生成报告
```

### 手动创建计划

```python
plan = Plan(
    goal="重构认证模块",
    steps=[
        PlanStep(1, "搜索认证相关文件", tool_hint="search"),
        PlanStep(2, "读取主要认证逻辑", tool_hint="read_file"),
        PlanStep(3, "执行重构", tool_hint="search_replace"),
        PlanStep(4, "运行测试验证", tool_hint="shell"),
    ]
)
```

## 完整示例

### 示例 1: 代码审查 Agent

```python
from codechat.agent_v2 import create_agent
from codechat.store import VectorStore
from pathlib import Path

def review_code(project_path: str, model: str = "gpt-4"):
    """代码审查 Agent"""
    
    store = VectorStore(Path(project_path))
    agent = create_agent(
        store=store,
        project_root=Path(project_path),
        model=model,
        max_steps=10
    )
    
    result = agent.run(
        "审查这个项目的代码质量，找出潜在问题并给出改进建议",
        on_step=lambda n, tool, p: print(f"[{n}] {tool}: {p[:50]}...")
    )
    
    return result.answer

# 使用
print(review_code("./my_project"))
```

### 示例 2: Bug 修复 Agent

```python
def fix_bug(project_path: str, bug_description: str, model: str = "gpt-4"):
    """Bug 修复 Agent"""
    
    store = VectorStore(Path(project_path))
    agent = create_agent(
        store=store,
        project_root=Path(project_path),
        model=model,
        max_steps=15
    )
    
    prompt = f"""\
修复以下 Bug：

{bug_description}

步骤：
1. 搜索相关代码
2. 分析问题原因
3. 使用 search_replace 或 write_file 修复
4. 确认修复成功
"""
    
    result = agent.run(prompt)
    return result

# 使用
result = fix_bug(
    "./my_project",
    "用户登录时偶尔会报错 'NoneType has no attribute encode'"
)
print(result.answer)
print(f"修改了 {len([a for a in result.actions if a['tool'] in ['write_file', 'search_replace']])} 个文件")
```

### 示例 3: 测试生成 Agent

```python
def generate_tests(project_path: str, target_file: str, model: str = "gpt-4"):
    """测试生成 Agent"""
    
    store = VectorStore(Path(project_path))
    agent = create_agent(
        store=store,
        project_root=Path(project_path),
        model=model,
        max_steps=10
    )
    
    prompt = f"""\
为 {target_file} 生成单元测试。

要求：
1. 先读取目标文件了解其功能
2. 为每个公开函数/方法创建测试
3. 使用 pytest 框架
4. 将测试写入 tests/ 目录
"""
    
    result = agent.run(prompt)
    return result
```

## 配置

### 环境变量

```bash
# LLM 配置
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
export LLM_MODEL="gpt-4"  # 默认模型

# 或使用 Ollama
export OLLAMA_API_KEY="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="codellama"
```

### 代码配置

```python
from codechat.agent_v2 import LLMClient

# 使用指定模型
llm = LLMClient(model="gpt-4-turbo")

# 使用缓存
result = llm.complete(
    system="你是一个代码助手",
    user="解释这段代码",
    use_cache=True  # 相同请求会使用缓存
)
```

## 调试

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行 Agent
result = agent.run("...")
```

### 查看执行历史

```python
result = agent.run("...")

# 查看所有执行的动作
for action in result.actions:
    print(f"{action['tool']}: {action['success']} ({action['elapsed_ms']:.0f}ms)")
```

### 重置记忆

```python
# 清除所有记忆（短期 + 长期）
agent.reset_memory()
```

## 性能优化

### 1. 并行工具执行

只读工具可以并行执行：
- `search`
- `read_file`
- `find_pattern`
- `list_dir`

### 2. 缓存 LLM 响应

```python
result = llm.complete(
    system=system_prompt,
    user=user_prompt,
    use_cache=True  # 缓存相同请求
)
```

### 3. 限制搜索范围

```python
# 使用 file_glob 限制搜索范围
result = agent.run("搜索所有 Python 文件中的 TODO")
# Agent 会自动使用 find_pattern 的 file_glob 参数
```

## 常见问题

### Q: Agent 不响应怎么办？

检查 LLM 配置：
```python
from codechat.agent_v2 import LLMClient
llm = LLMClient()
print(f"Available: {llm.available}")
print(f"Model: {llm.model_name}")
```

### Q: 工具执行失败？

查看错误信息：
```python
result = agent.run("...")
for action in result.actions:
    if not action['success']:
        print(f"Failed: {action['tool']}")
```

### Q: 记忆太多导致 Token 超限？

调整记忆参数：
```python
agent = create_agent(
    store=store,
    project_root=path,
    max_steps=5
)

# 或手动调整
agent.memory_st.max_entries = 10
agent.memory_st.max_tokens = 20000
```

### Q: 如何禁用规划？

```python
agent = create_agent(
    store=store,
    project_root=path,
    use_planning=False  # 禁用任务规划
)
```
