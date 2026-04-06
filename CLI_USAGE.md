# Agent v2 CLI 使用指南

## 快速开始

### 1. 构建索引

首先需要为项目构建向量索引：

```bash
snowcode ingest --path ./your_project
```

### 2. 使用增强版 Agent

#### 基础用法

```bash
# 解释项目架构
snowcode agent2 "解释这个项目的架构"

# 指定项目路径
snowcode agent2 --path ./my_project "找出所有 TODO 注释"
```

#### 高级选项

```bash
# 设置最大执行步数
snowcode agent2 --steps 10 "分析代码质量并生成报告"

# 禁用任务规划
snowcode agent2 --no-plan "搜索所有 Python 文件"

# 使用多Agent协作
snowcode agent2 --multi-agent --workers 3 "分析安全漏洞并提供修复建议"

# 指定模型
snowcode agent2 --model gpt-4 "重构认证模块"
```

## 命令参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--path` | `-p` | 项目路径 | 自动检测 |
| `--model` | `-m` | LLM 模型 | 配置文件中的默认模型 |
| `--steps` | `-s` | 最大执行步数 | 5 |
| `--no-plan` | | 禁用任务规划 | false |
| `--multi-agent` | | 使用多Agent模式 | false |
| `--workers` | `-w` | Worker 数量（多Agent模式） | 2 |

## 示例

### 示例 1: 代码审查

```bash
snowcode agent2 "审查这个项目的代码质量，找出潜在问题"
```

输出示例：
```
  LLM: gpt-4 | Steps: 5 | Planning: on
  Tools: search, read_file, find_pattern, list_dir, write_file, search_replace, shell

  Step 1 → search
  搜索代码质量问题...

  Step 2 → read_file
  读取 src/auth.py...

  Step 3 → find_pattern
  搜索所有 TODO 注释...

  ╭──────────────────────────────────── Answer ────────────────────────────────────╮
  │                                                                               │
  │ ## 代码审查报告                                                               │
  │                                                                               │
  │ ### 发现的问题                                                               │
  │                                                                               │
  │ 1. **安全问题** (src/auth.py:45)                                              │
  │    - 密码未进行哈希处理                                                        │
  │                                                                               │
  │ 2. **TODO 注释** (共 5 处)                                                    │
  │    - src/main.py:23 - TODO: 添加错误处理                                      │
  │    - src/utils.py:67 - TODO: 优化性能                                         │
  │                                                                               │
  │ ### 建议                                                                       │
  │    - 使用 bcrypt 对密码进行哈希                                                │
  │    - 清理 TODO 注释或创建对应的 Issue                                          │
  │                                                                               │
  ╰───────────────────────────────────────────────────────────────────────────────╯

  Execution Summary
  Steps: 3
  Tools used: search, read_file, find_pattern
  Time: 12500ms
```

### 示例 2: 多Agent协作

```bash
snowcode agent2 --multi-agent --workers 2 "分析项目的安全漏洞并提供修复建议"
```

输出示例：
```
  LLM: gpt-4 | Mode: Multi-Agent Coordinator | Workers: 2
  Tools: search, read_file, find_pattern, list_dir, write_file, search_replace, shell

  [进度] Analyzing question...
  [进度] Executing 2 subtasks in parallel...
  [进度] Synthesizing results...

  ╭──────────────────────────────────── Answer ────────────────────────────────────╮
  │                                                                               │
  │ ## 安全漏洞分析报告                                                           │
  │                                                                               │
  │ ### Worker 1 发现                                                             │
  │ - SQL 注入风险 (src/db.py:89)                                                 │
  │ - XSS 漏洞 (src/templates/index.html:34)                                      │
  │                                                                               │
  │ ### Worker 2 发现                                                             │
  │ - 硬编码密码 (src/config.py:12)                                               │
  │ - 未验证的重定向 (src/auth.py:56)                                              │
  │                                                                               │
  │ ### 修复建议                                                                   │
  │ 1. 使用参数化查询防止 SQL 注入                                                 │
  │ 2. 对用户输入进行转义防止 XSS                                                  │
  │ 3. 使用环境变量存储敏感信息                                                     │
  │ 4. 验证重定向 URL                                                              │
  │                                                                               │
  ╰───────────────────────────────────────────────────────────────────────────────╯
```

### 示例 3: Bug 修复

```bash
snowcode agent2 --steps 10 "修复用户登录时的 NoneType 错误"
```

输出示例：
```
  Step 1 → search
  搜索 NoneType 错误相关代码...

  Step 2 → read_file
  读取 src/auth/login.py...

  Step 3 → search_replace
  修复 src/auth/login.py:42 的空值检查...

  ╭──────────────────────────────────── Answer ────────────────────────────────────╮
  │                                                                               │
  │ ## 修复完成                                                                   │
  │                                                                               │
  │ 问题: 用户登录时偶尔报错 'NoneType has no attribute encode'                    │
  │                                                                               │
  │ 原因: `get_user()` 函数在用户不存在时返回 None，但后续代码未做空值检查          │
  │                                                                               │
  │ 修复:                                                                         │
  │ - 文件: src/auth/login.py:42                                                  │
  │ - 添加空值检查: `if user is None: return None`                                 │
  │                                                                               │
  │ 已创建备份: src/auth/login.py.bak                                              │
  │                                                                               │
  ╰───────────────────────────────────────────────────────────────────────────────╯
```

### 示例 4: 测试生成

```bash
snowcode agent2 "为 src/utils.py 生成单元测试"
```

## 与原版 Agent 的对比

| 功能 | agent (原版) | agent2 (增强版) |
|------|-------------|----------------|
| 工具系统 | 基础 | 权限检查、并发控制、进度报告 |
| 记忆管理 | 简单 | Token估算、智能截断 |
| 多Agent | 有限 | 完整的 Coordinator 模式 |
| 工具数量 | 11 | 7（精选核心工具） |
| 安全性 | 基础 | 危险命令检测、路径验证 |
| 进度报告 | 无 | 实时进度回调 |

## 调试技巧

### 查看详细执行过程

```bash
# Agent 会显示每一步的思考和执行
snowcode agent2 "分析代码"
```

### 限制执行步数

```bash
# 只执行 3 步（适合快速查询）
snowcode agent2 --steps 3 "搜索 TODO 注释"
```

### 禁用规划（更快响应）

```bash
# 跳过任务规划，直接执行
snowcode agent2 --no-plan "读取 README.md"
```

## 常见问题

### Q: Agent 执行太慢？

尝试减少步数：
```bash
snowcode agent2 --steps 3 "你的问题"
```

### Q: 想要更深入的分析？

增加步数并启用多Agent：
```bash
snowcode agent2 --steps 10 --multi-agent "深入分析这个项目"
```

### Q: 如何查看可用工具？

Agent 启动时会显示可用工具列表：
```
Tools: search, read_file, find_pattern, list_dir, write_file, search_replace, shell
```

### Q: 多Agent模式下Worker数量设置多少合适？

- 简单任务：1-2 个 Worker
- 复杂任务：2-3 个 Worker
- 非常复杂的任务：3-4 个 Worker（注意 API 调用成本）

## 环境配置

### 设置默认模型

```bash
# 方法 1: 环境变量
export LLM_MODEL="gpt-4"

# 方法 2: 配置文件
snowcode config
```

### 设置 API Key

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### 使用 Ollama

```bash
export OLLAMA_API_KEY="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="codellama"
```

## 完整示例工作流

```bash
# 1. 进入项目目录
cd /path/to/your/project

# 2. 构建索引
snowcode ingest

# 3. 查看项目状态
snowcode status

# 4. 使用增强版 Agent 分析项目
snowcode agent2 "生成项目架构概览"

# 5. 代码审查
snowcode agent2 --steps 10 "审查代码质量"

# 6. Bug 修复
snowcode agent2 "修复所有 TODO 和 FIXME"

# 7. 生成测试
snowcode agent2 "为核心模块生成单元测试"

# 8. 安全审计
snowcode agent2 --multi-agent "进行安全审计"
```
