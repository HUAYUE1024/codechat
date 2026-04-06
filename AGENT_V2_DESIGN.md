# Agent v2 设计文档

## 概述

Agent v2 是基于 Claude Code 优秀设计模式的增强版智能体实现。主要改进包括：

1. **工具系统增强** - 权限检查、并发控制、进度报告
2. **记忆管理优化** - 上下文压缩、智能截断、Token估算
3. **Agent架构改进** - 支持多Agent协作（Coordinator模式）
4. **系统提示优化** - 结构化提示工程
5. **执行机制改进** - 超时、重试、结果缓存

## 从 Claude Code 学到的关键设计

### 1. 工具系统架构

**Claude Code 设计**:
```typescript
type Tool<Input, Output, Progress> = {
  name: string
  call(args, context, canUseTool, parentMessage, onProgress): Promise<ToolResult<Output>>
  checkPermissions(input, context): Promise<PermissionResult>
  isReadOnly(input): boolean
  isConcurrencySafe(input): boolean
  maxResultSizeChars: number
}
```

**Snowcode 实现**:
```python
class BaseTool(ABC):
    name: str = ""
    description: str = ""
    max_result_size: int = 30000
    
    def check_permission(self, params: dict) -> ToolPermission:
        """权限检查"""
    
    def is_read_only(self) -> bool:
        """只读检查"""
    
    def is_concurrent_safe(self) -> bool:
        """并发安全检查"""
```

**关键改进**:
- ✅ 添加权限级别枚举（ALLOWED, PROMPT, DENIED, DANGEROUS）
- ✅ 支持并发控制，防止同名工具同时运行
- ✅ 添加结果大小限制和智能截断
- ✅ 支持进度回调

### 2. 记忆管理系统

**Claude Code 设计**:
- 短期记忆：滑动窗口 + Token限制
- 长期记忆：文件持久化 + 相似度检索
- 自动记忆：MEMORY.md 索引系统

**Snowcode 实现**:
```python
class ShortTermMemory:
    """智能上下文管理"""
    def __init__(self, max_entries: int = 20, max_tokens: int = 30000):
        self.entries: list[MemoryEntry] = []
        self.max_tokens = max_tokens
    
    def _prune(self):
        """基于Token估算的智能修剪"""
```

**关键改进**:
- ✅ 添加 Token 估算（1 token ≈ 4 chars）
- ✅ 智能修剪：保持首尾，删除中间旧条目
- ✅ 按角色差异化截断（tool: 2000, agent: 1000）
- ✅ 支持获取最近工具结果

### 3. 多Agent协作

**Claude Code 设计** (Coordinator模式):
```
Coordinator (指挥官)
├── Agent Tool → 派生 Worker
├── SendMessage → 继续 Worker
└── TaskStop → 停止 Worker

Worker (执行者)
└── 完整工具集
```

**Snowcode 实现**:
```python
class MultiAgentCoordinator:
    def coordinate(self, question: str) -> str:
        # 1. 分析问题
        # 2. 分解子任务
        # 3. 并行执行 Worker
        # 4. 综合结果
```

**关键改进**:
- ✅ 支持任务自动分解
- ✅ 并行执行子任务
- ✅ 结果自动综合
- ✅ Worker 完全独立（看不到原始问题）

### 4. 系统提示设计

**Claude Code 设计**:
```
Static (可缓存)
├── Introduction: 角色定义
├── System: 工具使用指南
├── Doing Tasks: 代码风格
├── Actions: 安全指令
└── Tone: 沟通风格

Dynamic (会话相关)
├── Session Guidance: 根据工具动态调整
├── Memory: 从 MEMORY.md 加载
└── MCP Instructions: MCP服务器说明
```

**Snowcode 实现**:
```python
AGENT_SYSTEM = """\
你是一个强大的代码 Agent...

## 输出格式
调用工具：{"think": "...", "tool": "...", "params": {...}}
给出结论：{"think": "...", "answer": "..."}

## 规则
1. 每次只调一个工具
2. 不要重复搜索
3. 读文件时一次读完整
...
"""
```

### 5. 执行机制

**Claude Code 设计**:
- 权限检查：validateInput → checkPermissions → canUseTool
- 并发控制：isConcurrencySafe 判断
- 进度报告：onProgress 回调
- 结果处理：大结果持久化到磁盘

**Snowcode 实现**:
```python
class ToolRegistry:
    def execute(self, name, params, ctx, on_progress) -> ToolResult:
        # 1. 检查并发
        # 2. 报告进度
        # 3. 执行工具
        # 4. 格式化输出
```

## 新增功能

### 1. 权限系统

```python
class ToolPermission(Enum):
    ALLOWED = "allowed"      # 自动允许
    PROMPT = "prompt"        # 需要用户确认
    DENIED = "denied"        # 被策略阻止
    DANGEROUS = "dangerous"  # 危险操作
```

### 2. 并发控制

```python
class ToolRegistry:
    def __init__(self):
        self._running_tools: set[str] = set()
        self._lock = threading.Lock()
    
    def execute(self, name, params, ctx):
        if not tool.is_concurrent_safe():
            with self._lock:
                if name in self._running_tools:
                    return ToolResult(False, "Tool already running", name)
                self._running_tools.add(name)
```

### 3. Token 估算

```python
@dataclass
class MemoryEntry:
    content: str
    token_estimate: int = 0
    
    def __post_init__(self):
        if not self.token_estimate:
            # Rough estimate: 1 token ≈ 4 chars
            self.token_estimate = len(self.content) // 4
```

### 4. 结果缓存

```python
class LLMClient:
    def __init__(self):
        self._cache: dict[str, str] = {}
    
    def complete(self, system, user, use_cache=False):
        if use_cache:
            cache_key = hashlib.md5(f"{system}:{user}".encode()).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]
```

## 使用示例

### 基础用法

```python
from codechat.agent_v2 import create_agent
from codechat.store import VectorStore
from pathlib import Path

# 创建 Agent
store = VectorStore(Path("./project"))
agent = create_agent(store, Path("./project"), model="gpt-4")

# 运行
result = agent.run("解释一下这个项目的架构")
print(result.answer)
print(f"Steps: {result.steps_taken}")
print(f"Tools used: {result.tools_used}")
```

### 多Agent协作

```python
from codechat.agent_v2 import create_coordinator

# 创建 Coordinator
coordinator = create_coordinator(store, Path("./project"), model="gpt-4")

# 运行（自动分解任务并行执行）
answer = coordinator.coordinate(
    "分析这个项目的安全漏洞并提供修复建议",
    on_progress=lambda msg: print(f"Progress: {msg}")
)
print(answer)
```

### 进度回调

```python
def on_step(step_num, tool_name, preview):
    print(f"Step {step_num}: {tool_name}")
    print(f"  Preview: {preview[:100]}...")

def on_think(think_text):
    print(f"Thinking: {think_text}")

result = agent.run(
    "找出所有TODO注释",
    on_step=on_step,
    on_think=on_think
)
```

## 性能优化

### 1. 并行工具执行

Claude Code 支持在单次响应中调用多个独立工具：
```typescript
// Claude Code
if (toolCalls.length > 1 && allConcurrencySafe) {
  await Promise.all(toolCalls.map(callTool))
}
```

Snowcode 可以在 Worker 中实现：
```python
# Future enhancement
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(registry.execute, name, params, ctx): name
        for name, params in independent_calls
    }
```

### 2. 提示缓存

Claude Code 使用静态/动态分离来优化缓存：
```typescript
// Static content (cacheable)
getSimpleIntroSection(),
getSimpleSystemSection(),
SYSTEM_PROMPT_DYNAMIC_BOUNDARY,  // <-- Boundary marker
// Dynamic content
...resolvedDynamicSections,
```

Snowcode 可以实现类似机制：
```python
# Static prompt (can be cached)
STATIC_PROMPT = """你是一个代码Agent..."""

# Dynamic prompt (changes per session)
def build_dynamic_prompt(tools, memory, plan):
    return f"""
## 工具列表
{tools}

## 记忆
{memory}

## 计划
{plan}
"""
```

### 3. 结果持久化

大结果应该持久化到磁盘而不是保留在内存中：
```python
class ToolResult:
    persisted_path: str | None = None  # 大结果的磁盘路径
    
    @property
    def output(self):
        if self.persisted_path:
            return Path(self.persisted_path).read_text()
        return self._output
```

## 未来改进方向

### 1. 做梦机制（Dream）

借鉴 Claude Code 的 KAIROS 系统：
- 自动整合会话记忆
- 四阶段流程：Orient → Gather → Consolidate → Prune
- 使用文件锁防止多进程冲突

### 2. 技能系统

```python
class Skill:
    """用户定义的技能"""
    name: str
    description: str
    prompt: str  # 展开后的完整提示
    
class SkillTool(BaseTool):
    """执行技能的工具"""
    def run(self, params, ctx):
        skill_name = params.get("skill")
        skill = self.skills[skill_name]
        return ctx.llm.complete(skill.prompt, params.get("input"))
```

### 3. 远程控制

借鉴 Claude Code 的 Bridge 系统：
- WebSocket 实时连接
- 从 Web 界面控制 CLI
- 状态同步

## 总结

Agent v2 从 Claude Code 借鉴了以下核心设计模式：

1. **工具系统**: 权限检查、并发控制、进度报告、结果管理
2. **记忆系统**: Token估算、智能修剪、上下文压缩
3. **Agent架构**: 多Agent协作、任务分解、并行执行
4. **系统提示**: 结构化设计、静态/动态分离
5. **执行机制**: 错误处理、重复检测、结果缓存

这些改进使 Snowcode 的 Agent 更加强大、安全和高效。
