# Snowcode Agent v2 快速入门

## 三步开始

### 1. 安装

```bash
cd Snowcode
pip install -e .
```

### 2. 构建索引

```bash
snowcode ingest --path ./your_project
```

### 3. 运行 Agent

```bash
snowcode agent2 "解释这个项目的架构"
```

## 常用命令

```bash
# 代码审查
snowcode agent2 "审查代码质量"

# Bug 修复
snowcode agent2 "修复 TODO 和 FIXME"

# 生成测试
snowcode agent2 "为 src/main.py 生成单元测试"

# 安全审计（多Agent）
snowcode agent2 --multi-agent "分析安全漏洞"

# 快速查询（3步）
snowcode agent2 --steps 3 "搜索所有 Python 文件"
```

## 参数速查

| 参数 | 说明 |
|------|------|
| `--steps N` | 最大执行步数 |
| `--multi-agent` | 启用多Agent协作 |
| `--workers N` | Worker 数量 |
| `--no-plan` | 禁用任务规划 |
| `--model MODEL` | 指定 LLM 模型 |

## 完整示例

```bash
# 1. 进入项目
cd /path/to/project

# 2. 构建索引
snowcode ingest

# 3. 分析项目
snowcode agent2 "生成架构概览"

# 4. 代码审查
snowcode agent2 --steps 10 "找出潜在问题"

# 5. 修复问题
snowcode agent2 "修复所有发现的问题"
```

## 环境变量

```bash
# OpenAI
export OPENAI_API_KEY="your-key"
export LLM_MODEL="gpt-4"

# Ollama
export OLLAMA_API_KEY="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="codellama"
```

## 更多信息

- 详细使用指南: `CLI_USAGE.md`
- 设计文档: `AGENT_V2_DESIGN.md`
- API 文档: `AGENT_V2_USAGE.md`
