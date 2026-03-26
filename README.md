<div align="center">

# codechat

**Local RAG codebase Q&A engine — chat with your code in the terminal**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Privacy](https://img.shields.io/badge/Privacy-100%25_Local-red.svg)](#privacy--security)

</div>

---

Take over a complex project or revisit your old codebase — understanding the architecture shouldn't take days. **codechat** vectorizes your project locally so you can "talk" to your code from the terminal.

## Quick Start

```bash
git clone https://github.com/HUAYUE1024/codechat.git
cd codechat
pip install -e .

cd /path/to/your-project
codechat ingest                              # build vector index
codechat ask "how does auth work?"           # ask questions
codechat agent "trace the request lifecycle" # multi-step exploration
codechat chat                                # interactive REPL
```

## Commands

| Command | Description |
|---------|-------------|
| `ingest` | Scan project, build vector index (incremental by default) |
| `ask "question"` | Ask about the codebase (streaming + LLM thinking) |
| `agent "question"` | Multi-step agent: Plan → Tools → Memory → Answer |
| `chat` | Interactive REPL with history and auto-complete |
| `explain "target"` | Explain a function, class, or file |
| `review` | Code review: bugs, security, performance |
| `find "pattern"` | Search code patterns (regex, definitions, imports) |
| `summary` | Generate project architecture overview |
| `trace "target"` | Trace function call chains |
| `compare A B` | Compare two files or modules |
| `test-suggest "target"` | Suggest test cases |
| `status` | Show index status |
| `clean` | Delete the vector index |

**Common options:** `-p PATH` project path, `-m MODEL` LLM model, `--show-sources` show source files

## Setup LLM

```cmd
:: Windows
set DASHSCOPE_API_KEY=sk-xxx

# Linux / Mac
export DASHSCOPE_API_KEY=sk-xxx
```

Default model: `qwen-flash`. See [LLM Config](#llm-config) for more options.

## Agent Mode

Multi-step agent with Planning, Tools, Memory, and Action.

```bash
codechat agent "how does the vector store persist data?"
codechat agent "question" -s 10       # limit to 10 steps
codechat agent "question" --no-plan   # skip planning
```

### Agent Tools (8 total — full CRUD)

| Tool | Operation | Description |
|------|-----------|-------------|
| `search` | Search | Semantic code search |
| `read_file` | **Read** | Read full file content (up to 2000 lines) |
| `find_pattern` | Search | Regex search across codebase |
| `list_dir` | Browse | Directory structure |
| `read_multiple` | Read | Read multiple files simultaneously |
| `write_file` | **Create/Update** | Write or overwrite file (auto .bak backup) |
| `search_replace` | **Update** | Find and replace specific code blocks |
| `delete_file` | **Delete** | Delete file (auto .deleted backup) |

All write/delete operations create backups before modifying.

### Agent Memory

- **Short-term**: Sliding window of tool calls (default 20 entries, 30K chars)
- **Long-term**: Q&A sessions persisted to `.codechat/memory.jsonl`
- **Repeat detection**: Auto-exits if same tool+params called repeatedly

### Agent Safety

- File operations restricted to project root (no path traversal)
- Regex patterns limited to prevent ReDoS
- Hard cap at 50 steps (override with `--steps`)
- All writes create `.bak` backups, deletes create `.deleted` backups

## Incremental Indexing

By default, `ingest` only processes changed files:

```bash
codechat ingest          # incremental: new/changed/deleted files only
codechat ingest --reset  # full rebuild
```

How it works:
1. File hashes (mtime + size) stored in `.codechat/file_hashes.json`
2. On subsequent runs, only changed/new files re-chunked and re-embedded
3. Chunks from deleted files automatically removed

## AST-Aware Chunking

Code is split using Tree-sitter AST parsing first, with regex and line-based fallback.

**Strategy:** AST (Tree-sitter) → regex function splitter → line-based

**Supported languages:** Python, JS/TS, Go, Rust, Java, C/C++, Ruby, PHP, C#, Kotlin, Swift, Lua, Bash, SQL, R, HTML, CSS — 20+ languages

## Skills

7 specialized prompts for specific analysis tasks:

| Command | Purpose |
|---------|---------|
| `explain` | Explain function/class/file |
| `review` | Code review (bugs, security, performance) |
| `find` | Search patterns (regex, definitions) |
| `summary` | Architecture overview |
| `trace` | Call chain tracing |
| `compare` | Compare two files |
| `test-suggest` | Test case suggestions |

## LLM Config

### DashScope (Recommended for China)

```cmd
set DASHSCOPE_API_KEY=sk-xxx
```

Default model: `qwen-flash`

### OpenAI Compatible

```cmd
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://api.openai.com/v1
```

### Ollama (Local)

```cmd
ollama pull qwen2.5-coder:7b
set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=qwen2.5-coder
```

### Thinking Mode

DashScope reasoning tokens, off by default:

```cmd
set CODECHAT_THINKING=1
codechat ask "complex question" --show-thinking
```

### Embedding Models

```bash
codechat ingest -m all-mpnet-base-v2           # default, best quality
codechat ingest -m all-MiniLM-L6-v2            # faster, lower quality
codechat ingest -m paraphrase-multilingual-MiniLM-L12-v2  # multilingual
```

| Model | Dimensions | Size |
|-------|-----------|------|
| `all-mpnet-base-v2` | 768 | 420MB |
| `all-MiniLM-L6-v2` | 384 | 90MB |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470MB |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   CLI (Click + Rich)                 │
├──────────┬────────────────┬──────────────────────────┤
│   ask    │    agent       │   skills                 │
│          │                │ explain/review/find      │
│          │ ┌────────────┐ │ summary/trace/           │
│          │ │ Planning   │ │ compare/test-suggest     │
│          │ │ Memory     │ │                          │
│          │ │ Action     │ │                          │
│          │ │ 8 Tools    │ │                          │
│          │ └────────────┘ │                          │
├──────────┴────────────────┴──────────────────────────┤
│                   RAG Engine                         │
├──────────────────────────────────────────────────────┤
│ Scanner → Chunker → VectorStore → LLM Client        │
│ (os.walk   (AST-first  (NumPy .npy  (DashScope     │
│  pruning)   + regex     + JSON +     OpenAI         │
│            fallback)    BM25 hybrid) Ollama)        │
└──────────────────────────────────────────────────────┘
```

## Project Structure

```
codechat/           ~4300 lines Python
├── cli.py          CLI commands (804 lines)
├── agent.py        Agent: Planning/Tools/Memory/Action (1129 lines)
├── store.py        Vector store + BM25 hybrid search (701 lines)
├── rag.py          RAG engine + LLM clients (390 lines)
├── tree_gen.py     Tree generation (338 lines)
├── ast_chunker.py  Tree-sitter AST chunking (297 lines)
├── skills.py       7 specialized skill prompts (239 lines)
├── chunker.py      Code chunking (260 lines)
├── scanner.py      File scanner (115 lines)
├── config.py       Constants and config (87 lines)
```

**Generated data:**
```
your-project/
├── .codechat/
│   ├── config.json        # Index config
│   ├── embeddings.npy     # Vector matrix
│   ├── metadata.json      # File paths + line numbers
│   ├── file_hashes.json   # File hashes for incremental indexing
│   ├── bm25.json          # BM25 index
│   └── memory.jsonl       # Agent long-term memory
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `click` | CLI framework |
| `numpy` | Vector storage |
| `sentence-transformers` | Embedding + reranking models |
| `tree-sitter` / `tree-sitter-languages` | AST parsing |
| `prompt-toolkit` | Interactive REPL |
| `rich` | Terminal output |
| `pathspec` | .gitignore parsing |
| `openai` | OpenAI-compatible API |
| `httpx` | Ollama API |

## Supported File Types

**Code:** `.py` `.js` `.ts` `.tsx` `.go` `.rs` `.java` `.kt` `.c` `.cpp` `.cs` `.rb` `.php` `.swift` `.sh` `.sql` `.proto` — 40+ languages

**Docs:** `.md` `.rst` `.txt`

**Config:** `.json` `.yaml` `.toml` `.xml` `.env`

**Auto-skipped:** `.git` `node_modules` `__pycache__` `.venv` `dist` `build` `.codechat`

## FAQ

**Q: Does it work without an LLM?**
Yes. Falls back to raw code retrieval.

**Q: Chinese support?**
Full support. Embedding and LLM both handle Chinese/English mixed input.

**Q: How to change models?**
```bash
codechat ingest -m all-mpnet-base-v2           # embedding model
codechat ask "question" -m qwen-plus           # LLM model
set CODECHAT_MODEL=deepseek-chat               # permanent LLM
```

**Q: What if the agent gets stuck in a loop?**
Repeat detection auto-exits after 2 identical tool calls. Hard cap at 50 steps.

## Privacy & Security

- All vector data stored locally in `.codechat/`
- Embedding runs locally via sentence-transformers
- Only LLM calls go to external API (DashScope/OpenAI/Ollama)
- File paths validated against project root (no path traversal)
- Regex patterns limited to prevent ReDoS
- Write/delete operations create backups

## Roadmap

- [x] RAG Q&A with semantic search
- [x] Agent with Planning + Tools + Memory (8 tools, CRUD)
- [x] 7 specialized skills
- [x] Streaming output + thinking mode
- [x] Long-term memory persistence
- [x] Incremental indexing (only changed files)
- [x] AST-aware chunking (Tree-sitter, 20+ languages)
- [x] BM25 hybrid search + cross-encoder reranking
- [x] Test suite
- [ ] Multi-turn conversation memory
- [ ] `.codechatignore` custom rules
- [ ] Export Q&A to Markdown

## License

[MIT](LICENSE)
