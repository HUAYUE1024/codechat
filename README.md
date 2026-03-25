<div align="center">

# codechat

**本地 RAG 代码库问答引擎 —— 终端里直接"对话"你的代码**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/Vector-NumPy-orange.svg)](https://numpy.org)
[![Local-First](https://img.shields.io/badge/Privacy-100%25_Local-red.svg)](#隐私与安全)

</div>

---

当你接手一个复杂的开源项目，或者面对自己几个月前写的庞大系统时，理解代码架构极其耗时。

**codechat** 在本地快速把整个项目向量化，让你直接在终端里 "对话" 代码库。不再需要逐个文件翻找，一句话就能定位鉴权逻辑在哪、数据流怎么走、某个函数被谁调用。

---

## 目录

- [功能概览](#功能概览)
- [架构](#架构)
- [快速开始](#快速开始)
- [命令详解](#命令详解)
- [Agent 模式](#agent-模式)
- [专业技能](#专业技能)
- [LLM 配置](#llm-配置)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [路线图](#路线图)

---

## 功能概览

| 命令 | 用途 |
|------|------|
| `codechat ingest` | 扫描项目，构建本地向量索引 |
| `codechat ask "问题"` | 终端提问，流式返回 AI 回答 |
| `codechat agent "问题"` | **多步 Agent**：自动规划 → 调工具 → 记忆 → 得出结论 |
| `codechat chat` | 交互式 REPL 对话模式 |
| `codechat explain "XX"` | 解释函数/类/文件 |
| `codechat review` | 代码审查（Bug/安全/性能） |
| `codechat find "XX"` | 搜索代码模式（TODO/正则/定义） |
| `codechat summary` | 项目架构概览 |
| `codechat trace "XX"` | 追踪调用链 |
| `codechat compare A B` | 对比两个文件 |
| `codechat test-suggest "XX"` | 生成测试用例建议 |
| `codechat status` | 查看索引状态 |
| `codechat clean` | 删除索引 |

**核心亮点：**
- 100% 本地运行，向量存储用 NumPy，Embedding 用 sentence-transformers，零数据外泄
- 支持 DashScope（通义千问）/ OpenAI / Ollama 多种 LLM 后端
- Agent 模式支持自动规划、5 种工具调用、短期+长期记忆
- 流式输出 + 思考过程显示
- Windows 中文路径全兼容

---

## 架构

```
┌──────────────────────────────────────────────────────┐
│                      CLI (Click + Rich)              │
├────────────┬──────────────────┬──────────────────────┤
│   ask      │     agent        │   skills             │
│            │                  │ explain/review/find  │
│            │  ┌────────────┐  │ summary/trace/       │
│            │  │  Planning  │  │ compare/test-suggest │
│            │  │  Memory    │  │                      │
│            │  │  Action    │  │                      │
│            │  │  Tools     │  │                      │
│            │  └────────────┘  │                      │
├────────────┴──────────────────┴──────────────────────┤
│                      RAG Engine                       │
│           (Embedding → Retrieve → Generate)           │
├──────────────────────────────────────────────────────┤
│   Scanner  →  Chunker  →  VectorStore  →  LLM Client│
│   (.gitignore    (函数级   (NumPy .npy    (DashScope │
│    感知)         分块)     + JSON)        OpenAI     │
│                                           Ollama)    │
└──────────────────────────────────────────────────────┘
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/yourname/codechat.git
cd codechat
pip install -e .
```

### 使用

```bash
# 1. 进入项目目录
cd /path/to/your-project

# 2. 建立向量索引
codechat ingest

# 3. 直接提问
codechat ask "这个项目怎么处理数据库连接池？"

# 4. 使用 Agent 深度探索
codechat agent "向量检索的完整流程是什么？从用户输入到返回结果"

# 5. 启动交互对话
codechat chat
```

### 配置 LLM

```bash
# Windows (cmd)
set DASHSCOPE_API_KEY=sk-xxx

# Linux / Mac
export DASHSCOPE_API_KEY=sk-xxx

# 然后直接用
codechat ask "你是谁"
```

---

## 命令详解

### `codechat ingest`

```bash
codechat ingest [OPTIONS]

Options:
  -p, --path PATH           项目路径（默认：自动检测 git 仓库）
  --reset                   清除已有索引后重建
  --chunk-size INTEGER      分块大小（默认 1000 字符）
  --chunk-overlap INTEGER   分块重叠（默认 200 字符）
  -m, --model TEXT          Embedding 模型（默认 all-MiniLM-L6-v2）
```

**智能分块策略：** 优先按 `def` / `class` / `func` 等函数边界切割，保持语义完整性；超长函数回退到行级重叠切割。

### `codechat ask`

```bash
codechat ask [OPTIONS] QUESTION

Options:
  -p, --path PATH           项目路径
  -k, --context INTEGER     检索片段数（默认 5）
  -m, --model TEXT          LLM 模型
  --show-sources            显示来源文件
  --show-thinking           显示 LLM 思考过程
```

**输出效果（有 LLM）：**
```
  LLM: qwen-flash @ https://dashscope.aliyuncs.com/compatible-mode/v1

  向量存储实现在 `codechat/store.py` 中：
  - `_save()`: 将 embeddings 保存为 `.npy` 文件
  - `_load()`: 从磁盘加载向量和元数据
  - `query()`: 余弦相似度检索，返回 Top-K 结果
```

### `codechat chat`

交互式对话模式，支持：
- 命令历史（上下箭头）
- 自动补全
- 斜杠命令：`/quit` `/reset` `/stats` `/help`

---

## Agent 模式

Agent 是 codechat 的高级功能，支持 **自动规划 → 工具调用 → 记忆 → 反思 → 得出结论** 的多步推理。

```bash
codechat agent "这个项目的鉴权逻辑是怎么实现的？从请求进入到验证完成的完整流程"
```

### 架构

```
用户问题
   ↓
┌──────────┐
│ Planning │ ← LLM 拆解为 2-5 个步骤
└────┬─────┘
     ↓
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Memory  │    │  Action  │    │  Tools   │
│          │    │          │    │          │
│ Short:   │───→│ execute()│───→│ search   │
│ 会话记忆 │    │ retry    │    │ read_file│
│          │    │ timeout  │    │ find_    │
│ Long:    │    │ log      │    │  pattern │
│ 持久记忆 │    └──────────┘    │ list_dir │
│ .jsonl   │         │         │ read_    │
└──────────┘         ↓         │  multiple│
              观察结果 → 反思 →└──────────┘
                    ↓
              最终回答
```

### 5 个工具

| 工具 | 功能 | Agent 使用场景 |
|------|------|----------------|
| `search` | 语义搜索代码 | 初步定位相关代码 |
| `read_file` | 读取文件/行范围 | 深入查看具体实现 |
| `find_pattern` | 正则搜索 | 查找函数定义、import、调用点 |
| `list_dir` | 浏览目录结构 | 了解项目组织方式 |
| `read_multiple` | 同时读取多个文件 | 对比不同模块的实现 |

### 记忆系统

- **短期记忆**：当前会话内的工具调用历史（滑动窗口，默认 20 条）
- **长期记忆**：每次问答结果持久化到 `.codechat/memory.jsonl`，下次遇到类似问题自动召回

### 参数

```bash
codechat agent "问题" -s 3          # 限制 3 步
codechat agent "问题" --no-plan     # 跳过规划阶段
```

---

## 专业技能

7 个预置的专业 prompt，比通用 `ask` 更精准：

| 命令 | 用途 | 示例 |
|------|------|------|
| `explain` | 解释函数/类/文件 | `codechat explain "VectorStore的query方法"` |
| `review` | 代码审查 | `codechat review` 或 `codechat review store.py` |
| `find` | 搜索代码模式 | `codechat find "所有的异常处理"` |
| `summary` | 项目架构概览 | `codechat summary` |
| `trace` | 追踪调用链 | `codechat trace "answer_question 的调用链"` |
| `compare` | 对比两个文件 | `codechat compare store.py chunker.py` |
| `test-suggest` | 测试用例建议 | `codechat test-suggest "chunk_file 函数"` |

每个 skill 有独立的 system prompt、检索策略和 `n_context` 配置。

---

## LLM 配置

### DashScope（通义千问）推荐

```bash
# Windows
set DASHSCOPE_API_KEY=sk-xxx

# Linux / Mac
export DASHSCOPE_API_KEY=sk-xxx
```

默认模型：`qwen-flash`（快速便宜）。

### OpenAI 兼容 API

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.openai.com/v1
```

### 国内 LLM 服务

```bash
# DeepSeek
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://api.deepseek.com/v1
set CODECHAT_MODEL=deepseek-chat

# 通义千问（OpenAI 兼容方式）
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
set CODECHAT_MODEL=qwen-plus
```

### Ollama 本地模型

```bash
ollama pull qwen2.5-coder:7b
set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=qwen2.5-coder
```

### 思考模式

DashScope 支持模型深度思考（reasoning），默认关闭：

```bash
set CODECHAT_THINKING=1
codechat ask "复杂问题" --show-thinking
```

### Embedding 模型

```bash
# 使用更精确的模型（更大，更慢）
codechat ingest -m all-mpnet-base-v2

# 多语言优化
codechat ingest -m paraphrase-multilingual-MiniLM-L12-v2
```

| 模型 | 维度 | 大小 | 特点 |
|------|------|------|------|
| `all-MiniLM-L6-v2` | 384 | 90MB | 默认，快速 |
| `all-mpnet-base-v2` | 768 | 420MB | 更精确 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470MB | 多语言 |

---

## 项目结构

```
codechat/
├── pyproject.toml         # 项目配置 & 依赖
├── README.md
├── .gitignore
└── codechat/
    ├── __init__.py        # 版本号
    ├── __main__.py        # python -m codechat 入口
    ├── cli.py             # CLI 命令 (Click + Rich)
    ├── config.py          # 常量、文件类型、目录过滤
    ├── scanner.py         # 文件扫描 (.gitignore 感知)
    ├── chunker.py         # 智能代码分块
    ├── store.py           # NumPy + JSON 向量存储
    ├── rag.py             # RAG 检索增强生成
    ├── agent.py           # Agent: Planning/Tools/Memory/Action/LLM
    ├── skills.py          # 7 个专业 skill
```

**使用后生成的数据目录：**
```
your-project/
├── .codechat/
│   ├── config.json        # 索引配置
│   ├── embeddings.npy     # 向量矩阵
│   ├── metadata.json      # 文件路径 + 行号
│   └── memory.jsonl       # Agent 长期记忆
```

### 技术栈

| 组件 | 技术 |
|------|------|
| CLI | Click + Rich + Prompt Toolkit |
| Embedding | sentence-transformers (all-MiniLM-L6-v2) |
| 向量存储 | NumPy .npy + JSON |
| 检索 | 余弦相似度 + 文件类型加权 + 多样化去重 |
| LLM | DashScope / OpenAI / Ollama |
| Agent | ReAct (Plan → Act → Observe → Reflect) |

---

## 常见问题

**Q: 没有配置 LLM 也能用吗？**
可以。纯检索模式直接返回相关代码片段和文件定位。

**Q: 支持中文提问吗？**
完全支持。Embedding 和 LLM 都支持中英文混合。

**Q: Windows 中文路径有兼容问题吗？**
已全部处理。使用 NumPy 替代 ChromaDB，UTF-8 强制编码。

**Q: 索引占多大空间？**
一般 1000 个文件的项目，索引约 20-50MB。

**Q: 更新代码后要重新 ingest 吗？**
是的。`codechat ingest --reset` 完全重建。

**Q: 怎么用国内 LLM？**
设置 `OPENAI_BASE_URL` 指向国内 API 端点即可，参见 [LLM 配置](#llm-配置)。

---

## 支持的文件类型

**代码：** `.py` `.js` `.ts` `.tsx` `.go` `.rs` `.java` `.kt` `.c` `.cpp` `.cs` `.rb` `.php` `.swift` `.sh` `.sql` `.proto` 等 40+ 种

**文档：** `.md` `.rst` `.txt`

**配置：** `.json` `.yaml` `.toml` `.xml` `.env`

**自动跳过：** `.git` `node_modules` `__pycache__` `.venv` `dist` `build` 等

---

## 路线图

- [x] RAG 问答（语义搜索 + LLM 生成）
- [x] Agent 多步推理（规划 + 工具 + 记忆）
- [x] 7 个专业 skill
- [x] 流式输出 + 思考过程
- [x] 长期记忆持久化
- [ ] 增量索引（只重建变更文件）
- [ ] AST 感知分块（Tree-sitter）
- [ ] 对话记忆（多轮上下文）
- [ ] `.codechatignore` 自定义忽略规则
- [ ] 导出问答结果为 Markdown
- [ ] VS Code / Neovim 插件

---

## 贡献

欢迎 Issue 和 PR。

```bash
git checkout -b feature/xxx
# 开发
git commit -m "feat: xxx"
git push origin feature/xxx
# 创建 Pull Request
```

## License

[MIT](LICENSE)
