"""
CodeAgent — ReAct Agent with Planning / Tools / Memory / Action / LLM

Architecture:
    Planning  →  decompose user goal into executable steps
    Tools     →  external capabilities the agent can invoke
    Memory    →  short-term (session) + long-term (persisted)
    Action    →  execute tool calls with retry, timeout, error handling
    LLM       →  reasoning engine powering plan, decide, reflect
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_codechat_dir
from .rag import _get_llm_config, _call_llm
from .scanner import scan_files, read_file
from .store import VectorStore


# ═══════════════════════════════════════════════════════════════════════
#  LLM Client
# ═══════════════════════════════════════════════════════════════════════

class LLMClient:
    """Thin wrapper around LLM backends."""

    def __init__(self, model: str | None = None):
        self.model = model
        self.api_key, self.base_url, self.model_name, self.thinking = _get_llm_config(model)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        """Non-streaming completion."""
        if not self.available:
            return ""
            
        if self.api_key == "ollama":
            try:
                import httpx
                resp = httpx.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "stream": False,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    return resp.json()["message"]["content"]
                return f"[LLM Error] HTTP {resp.status_code}: {resp.text}"
            except Exception as e:
                return f"[LLM Error] {type(e).__name__}: {e}"
                
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            extra = {"enable_thinking": self.thinking} if self.thinking else {}
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=4096,
                extra_body=extra,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM Error] {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  Tools
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ToolResult:
    success: bool
    output: str
    tool_name: str
    elapsed_ms: float = 0


class Tool:
    """Base class for tools."""

    name: str = ""
    description: str = ""

    @property
    def parameters(self) -> dict[str, str]:
        return {}

    def run(self, params: dict, ctx: dict) -> str:
        raise NotImplementedError


class SearchTool(Tool):
    name = "search"
    description = "语义搜索代码库，返回最相关的代码片段"

    @property
    def parameters(self):
        return {"query": "搜索关键词", "n": "结果数量(默认5)"}

    def run(self, params: dict, ctx: dict) -> str:
        store: VectorStore = ctx["store"]
        query = params.get("query", "")
        n = int(params.get("n", 5))
        if not query:
            return "Error: query required"
            
        # Optional: Query Expansion via LLM to get better keywords
        llm = ctx.get("llm")
        expanded_query = query
        if llm and llm.available:
            prompt = "你是一个代码搜索专家。将以下用户问题转换为用于向量搜索的丰富关键词。包含可能的英文变量名、函数名和技术术语。不要解释，只返回关键词。"
            try:
                expanded = llm.complete(prompt, query).strip()
                if expanded and len(expanded) < 200:
                    expanded_query = f"{query} {expanded}"
            except Exception:
                pass
                
        results = store.query(expanded_query, n_results=n)
        if not results:
            return "No results."
        parts = []
        for i, r in enumerate(results, 1):
            m = r["metadata"]
            parts.append(f"[{i}] `{m['file_path']}` L{m['start_line']}-{m['end_line']}\n```\n{r['content']}\n```")
        return "\n\n".join(parts)


class ReadFileTool(Tool):
    name = "read_file"
    description = "读取文件内容（可指定行范围）"

    @property
    def parameters(self):
        return {"path": "文件路径", "start": "起始行(可选)", "end": "结束行(可选)"}

    def run(self, params: dict, ctx: dict) -> str:
        root: Path = ctx["root"]
        path = params.get("path", "")
        if not path:
            return "Error: path required"
        # Resolve and validate path is within project root
        full = (root / path).resolve()
        if not full.is_relative_to(root):
            return f"Access denied: path outside project root"
        if not full.exists():
            # try glob within root only
            name = Path(path).name
            matches = [m for m in root.rglob(name) if m.resolve().is_relative_to(root)]
            if matches:
                full = matches[0]
            else:
                return f"File not found: {path}"
        content = read_file(full)
        if content is None:
            return f"Cannot read: {path}"
        lines = content.splitlines()
        total = len(lines)
        s = max(1, int(params.get("start", 1)))
        e = min(total, int(params.get("end", total)))
        
        # Smart truncation to prevent token overflow (limit to max ~500 lines)
        MAX_LINES = 500
        if e - s + 1 > MAX_LINES:
            e = s + MAX_LINES - 1
            trunc_msg = f"\n... [Truncated for length. Only showing {MAX_LINES} lines] ..."
        else:
            trunc_msg = ""
            
        numbered = "\n".join(f"{i+s:>4} | {l}" for i, l in enumerate(lines[s-1:e]))
        rel = str(full.relative_to(root))
        return f"`{rel}` ({total} lines, showing {s}-{e})\n```\n{numbered}{trunc_msg}\n```"


import concurrent.futures

class FindPatternTool(Tool):
    name = "find_pattern"
    description = "正则搜索代码（找函数定义、import等）"

    @property
    def parameters(self):
        return {"pattern": "正则表达式", "file_glob": "文件过滤(可选)"}

    def _search_file(self, f: Path, root: Path, regex: re.Pattern, file_glob: str | None) -> list[str]:
        matches = []
        if file_glob:
            rel = str(f.relative_to(root))
            if not Path(rel).match(file_glob):
                return matches
        content = read_file(f)
        if content is None:
            return matches
        for i, line in enumerate(content.splitlines(), 1):
            search_line = line[:500]
            try:
                if regex.search(search_line):
                    rel = str(f.relative_to(root))
                    matches.append(f"`{rel}:{i}`  {line.strip()}")
            except Exception:
                continue
        return matches

    def run(self, params: dict, ctx: dict) -> str:
        root: Path = ctx["root"]
        pattern = params.get("pattern", "")
        if not pattern:
            return "Error: pattern required"
        if len(pattern) > 200:
            return "Error: pattern too long (max 200 chars)"
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex: {e}"
        file_glob = params.get("file_glob")
        
        files = scan_files(root)
        all_matches = []
        
        # Parallelize the file searching
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._search_file, f, root, regex, file_glob) for f in files]
            for future in concurrent.futures.as_completed(futures):
                matches = future.result()
                all_matches.extend(matches)
                if len(all_matches) >= 30:
                    break
                    
        return "\n".join(all_matches[:30]) if all_matches else "No matches."


class ListDirTool(Tool):
    name = "list_dir"
    description = "列出目录结构"

    @property
    def parameters(self):
        return {"path": "目录路径(留空=根目录)", "depth": "深度(默认2)"}

    def run(self, params: dict, ctx: dict) -> str:
        root: Path = ctx["root"]
        path = params.get("path", "")
        target = (root / path).resolve() if path else root
        
        if not target.is_relative_to(root):
            return f"Access denied: path outside project root"
            
        depth = int(params.get("depth", 2))
        if not target.exists():
            return f"Not found: {params.get('path', '.')}"
        lines = []
        self._walk(target, depth, lines, "")
        return "\n".join(lines) if lines else "(empty)"

    def _walk(self, d: Path, depth: int, lines: list, prefix: str):
        if depth <= 0:
            return
        skip = {".git","__pycache__","node_modules",".venv","venv",".codechat","dist","build"}
        try:
            entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return
        for e in entries:
            if e.name in skip or e.name.startswith("."):
                continue
            if e.is_dir():
                lines.append(f"{prefix}{e.name}/")
                self._walk(e, depth-1, lines, prefix+"  ")
            else:
                try:
                    sz = e.stat().st_size
                    ss = f"{sz:,}B" if sz < 1024 else f"{sz//1024:,}KB"
                except OSError:
                    ss = "?"
                lines.append(f"{prefix}{e.name}  [{ss}]")


class ReadMultipleTool(Tool):
    name = "read_multiple"
    description = "同时读取多个文件的关键行"

    @property
    def parameters(self):
        return {"files": "格式: 'a.py:10-30,b.py:5-15'"}

    def run(self, params: dict, ctx: dict) -> str:
        root: Path = ctx["root"]
        spec = params.get("files", "")
        if not spec:
            return "Error: files required"
        parts = []
        MAX_TOTAL_LINES = 1000
        total_lines_read = 0
        
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                fp, rs = item.rsplit(":", 1)
                try:
                    s, e = map(int, rs.split("-"))
                except ValueError:
                    s, e = 1, 9999
            else:
                fp, s, e = item, 1, 9999
            full = (root / fp).resolve()
            if not full.is_relative_to(root):
                parts.append(f"Access denied: {fp}")
                continue
            if not full.exists():
                parts.append(f"Not found: {fp}")
                continue
            content = read_file(full)
            if content is None:
                continue
            lines = content.splitlines()
            s, e = max(1, s), min(len(lines), e)
            
            lines_to_read = e - s + 1
            if total_lines_read + lines_to_read > MAX_TOTAL_LINES:
                allowed = max(0, MAX_TOTAL_LINES - total_lines_read)
                e = s + allowed - 1
                trunc_msg = f"\n... [Truncated due to total line limit]"
            else:
                trunc_msg = ""
                
            if e >= s:
                numbered = "\n".join(f"{i+s:>4} | {l}" for i, l in enumerate(lines[s-1:e]))
                parts.append(f"`{fp}` L{s}-{e}\n```\n{numbered}{trunc_msg}\n```")
                total_lines_read += (e - s + 1)
                
            if total_lines_read >= MAX_TOTAL_LINES:
                parts.append("... [Maximum line limit reached for this read_multiple call]")
                break
                
        return "\n\n".join(parts)


class WriteFileTool(Tool):
    name = "write_file"
    description = "写入或覆盖整个文件内容"

    @property
    def parameters(self):
        return {"path": "文件路径", "content": "要写入的完整文件内容"}

    def run(self, params: dict, ctx: dict) -> str:
        root: Path = ctx["root"]
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return "Error: path required"
            
        full = (root / path).resolve()
        if not full.is_relative_to(root):
            return f"Access denied: path outside project root"
            
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            return f"Successfully wrote to `{path}` ({len(content)} chars)."
        except Exception as e:
            return f"Error writing file: {e}"


class SearchReplaceTool(Tool):
    name = "search_replace"
    description = "搜索并替换文件中的指定代码块"

    @property
    def parameters(self):
        return {
            "path": "文件路径", 
            "old_str": "要被替换的原始代码块(需精确匹配)", 
            "new_str": "新的代码块"
        }

    def run(self, params: dict, ctx: dict) -> str:
        root: Path = ctx["root"]
        path = params.get("path", "")
        old_str = params.get("old_str", "")
        new_str = params.get("new_str", "")
        
        if not path or not old_str:
            return "Error: path and old_str required"
            
        full = (root / path).resolve()
        if not full.is_relative_to(root):
            return f"Access denied: path outside project root"
            
        if not full.exists():
            return f"Error: File not found `{path}`"
            
        try:
            content = full.read_text(encoding="utf-8")
            if old_str not in content:
                return "Error: old_str not found in the file. Ensure exact matching including whitespace."
                
            new_content = content.replace(old_str, new_str, 1) # Only replace first occurrence to be safe
            full.write_text(new_content, encoding="utf-8")
            return f"Successfully replaced code in `{path}`."
        except Exception as e:
            return f"Error modifying file: {e}"


class ToolRegistry:
    """Registry holding all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_definitions(self) -> str:
        lines = []
        for t in self._tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in t.parameters.items())
            lines.append(f"- **{t.name}**({params}): {t.description}")
        return "\n".join(lines)

    def execute(self, name: str, params: dict, ctx: dict) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(False, f"Unknown tool: {name}", name)
        start = time.time()
        try:
            output = tool.run(params, ctx)
            elapsed = (time.time() - start) * 1000
            return ToolResult(True, output, name, elapsed)
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ToolResult(False, f"{type(e).__name__}: {e}", name, elapsed)


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(SearchTool())
    reg.register(ReadFileTool())
    reg.register(FindPatternTool())
    reg.register(ListDirTool())
    reg.register(ReadMultipleTool())
    reg.register(WriteFileTool())
    reg.register(SearchReplaceTool())
    return reg


# ═══════════════════════════════════════════════════════════════════════
#  Memory
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    role: str          # "user", "agent", "tool", "system"
    content: str
    tool_name: str = ""
    timestamp: float = field(default_factory=time.time)


class ShortTermMemory:
    """In-session conversation memory with sliding window."""

    def __init__(self, max_entries: int = 20):
        self.entries: list[MemoryEntry] = []
        self.max_entries = max_entries

    def add(self, role: str, content: str, tool_name: str = ""):
        self.entries.append(MemoryEntry(role=role, content=content, tool_name=tool_name))
        if len(self.entries) > self.max_entries:
            # Keep first (goal) and last N
            self.entries = [self.entries[0]] + self.entries[-(self.max_entries - 1):]

    def get_context(self, max_chars: int = 8000) -> str:
        """Format memory into a string for the LLM, strictly bound by max_chars."""
        lines = []
        total = 0
        for entry in reversed(self.entries):
            # Truncate overly long single tool outputs to prevent token overflow
            content_str = entry.content
            if entry.role == "tool" and len(content_str) > 2000:
                content_str = content_str[:2000] + "\n...[truncated for length]..."
                
            line = f"[{entry.role}] {content_str}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        lines.reverse()
        return "\n---\n".join(lines)

    def clear(self):
        self.entries.clear()


class LongTermMemory:
    """Persistent memory stored in .codechat/memory.jsonl."""

    def __init__(self, project_root: Path):
        self.dir = get_codechat_dir(project_root)
        self.path = self.dir / "memory.jsonl"

    def store(self, question: str, answer: str, actions: list[dict]):
        """Save a Q&A session for future reference."""
        entry = {
            "ts": time.time(),
            "q": question[:200],
            "a": answer[:500],
            "actions": [a["tool"] for a in actions],
            "hash": hashlib.md5(question.encode()).hexdigest()[:8],
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def recall(self, question: str, n: int = 3) -> str:
        """Retrieve similar past Q&As."""
        if not self.path.exists():
            return ""
        q_words = set(question.lower().split())
        scored = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                past_q = entry.get("q", "")
                overlap = len(q_words & set(past_q.lower().split()))
                if overlap > 0:
                    scored.append((overlap, entry))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return ""
        lines = []
        for _, e in scored[:n]:
            lines.append(f"之前问过类似问题: \"{e['q']}\"\n回答概要: {e['a']}")
        return "\n\n".join(lines)

    def clear(self):
        self.path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  Planning
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PlanStep:
    index: int
    description: str
    tool_hint: str = ""      # suggested tool
    status: str = "pending"  # pending / running / done / failed
    result: str = ""


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    current: int = 0

    @property
    def done(self) -> bool:
        return all(s.status in ("done", "failed") for s in self.steps)

    @property
    def current_step(self) -> PlanStep | None:
        for s in self.steps:
            if s.status == "pending":
                return s
        return None

    def mark_current(self, status: str, result: str = ""):
        for s in self.steps:
            if s.status == "pending":
                s.status = status
                s.result = result[:200]
                break

    def to_context(self) -> str:
        lines = [f"Goal: {self.goal}", ""]
        for s in self.steps:
            status_icon = {"pending": "[ ]", "running": "[~]", "done": "[+]", "failed": "[-]"}
            lines.append(f"  {status_icon.get(s.status, '?')} Step {s.index}: {s.description}")
            if s.result:
                lines.append(f"      → {s.result[:100]}")
        return "\n".join(lines)


PLANNER_PROMPT = """\
你是一个任务规划器。根据用户目标，拆解为 2-5 个可执行步骤。

输出 JSON 数组，每个元素：
{{"index": 1, "description": "步骤描述", "tool_hint": "建议使用的工具名(可选)"}}

可用工具：
{tools}

规则：
1. 步骤要具体可执行，不要写"分析代码"这种模糊描述
2. 每个步骤对应一次工具调用
3. 最后一步通常是"综合信息给出回答"
4. 只输出 JSON 数组，不要其他内容
"""


class Planner:
    """Decompose user goals into executable plan steps."""

    def __init__(self, llm: LLMClient, tools_desc: str):
        self.llm = llm
        self.tools_desc = tools_desc

    def create_plan(self, goal: str) -> Plan:
        prompt = PLANNER_PROMPT.format(tools=self.tools_desc)
        user_msg = f"用户目标: {goal}"
        raw = self.llm.complete(prompt, user_msg)

        steps = self._parse_steps(raw)
        if not steps:
            # Fallback: single-step plan
            steps = [PlanStep(index=1, description=f"搜索并回答: {goal}", tool_hint="search")]

        return Plan(goal=goal, steps=steps)

    def refine_plan(self, plan: Plan, observation: str) -> Plan:
        """Adjust remaining steps based on new information."""
        # Simple refinement: if a step failed, add a retry with different approach
        failed = [s for s in plan.steps if s.status == "failed"]
        for s in failed:
            s.status = "pending"  # retry once
            s.tool_hint = "find_pattern" if s.tool_hint == "search" else "search"
        return plan

    def _parse_steps(self, raw: str) -> list[PlanStep]:
        # Extract JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
            return [
                PlanStep(
                    index=item.get("index", i+1),
                    description=item.get("description", ""),
                    tool_hint=item.get("tool_hint", ""),
                )
                for i, item in enumerate(data)
            ]
        except (json.JSONDecodeError, TypeError):
            return []


# ═══════════════════════════════════════════════════════════════════════
#  Action Executor
# ═══════════════════════════════════════════════════════════════════════

class ActionExecutor:
    """Execute tool calls with retry and error handling."""

    def __init__(self, registry: ToolRegistry, max_retries: int = 1):
        self.registry = registry
        self.max_retries = max_retries
        self.log: list[dict] = []

    def execute(self, tool_name: str, params: dict, ctx: dict) -> ToolResult:
        """Execute with retry on failure."""
        result = None
        for attempt in range(self.max_retries + 1):
            result = self.registry.execute(tool_name, params, ctx)
            self.log.append({
                "tool": tool_name,
                "params": {k: str(v)[:50] for k, v in params.items()},
                "success": result.success,
                "attempt": attempt + 1,
                "elapsed_ms": result.elapsed_ms,
            })
            if result.success:
                return result
            # On retry, try adjusting params slightly
            if attempt < self.max_retries and tool_name == "search":
                params["n"] = str(int(params.get("n", 5)) + 3)
        return result


# ═══════════════════════════════════════════════════════════════════════
#  Agent Orchestration
# ═══════════════════════════════════════════════════════════════════════

AGENT_SYSTEM = """\
你是一个强大的代码 Agent，通过工具查找、分析并修改代码。

## 输出格式（每次回复必须是 JSON）

调用工具：
{{"think": "思考过程", "tool": "工具名", "params": {{"参数": "值"}}}}

给出结论：
{{"think": "思考过程", "answer": "最终回答(Markdown)"}}

## 规则

1. 每次只调一个工具
2. 不要重复搜索相同关键词
3. 连续 3 次无结果则直接回答
4. 最多 {max_steps} 轮
5. 回答直接了当，不要"根据上下文"等废话
6. 精确标注代码位置(文件:行号)
7. 中文回答，代码术语英文
8. 当被要求修改代码、修复 Bug 或生成测试时，必须使用 write_file 或 search_replace 工具直接将代码写入项目，而不仅仅是提供代码建议！
"""


@dataclass
class AgentResult:
    answer: str
    plan: Plan | None
    actions: list[dict]
    steps_taken: int
    memory_entries: int


class CodeAgent:
    """
    ReAct Agent: Plan → Think → Act → Observe → Reflect → ... → Answer

    Components:
    - Planning:   decompose goal into steps
    - Memory:     short-term (session) + long-term (persisted)
    - Tools:      search, read_file, find_pattern, list_dir, read_multiple
    - Action:     execute tools with retry
    - LLM:        reasoning engine
    """

    def __init__(
        self,
        store: VectorStore,
        project_root: Path,
        model: str | None = None,
        max_steps: int = 5,
        use_planning: bool = True,
    ):
        self.store = store
        self.root = project_root.resolve()
        self.max_steps = max_steps
        self.use_planning = use_planning

        # Components
        self.llm = LLMClient(model)
        self.tools = build_default_registry()
        self.memory_st = ShortTermMemory(max_entries=20)
        self.memory_lt = LongTermMemory(self.root)
        self.executor = ActionExecutor(self.tools, max_retries=1)
        self.planner = Planner(self.llm, self.tools.list_definitions())

        self.ctx = {"store": store, "root": self.root, "llm": self.llm}

    def run(
        self,
        question: str,
        on_step: Callable[[int, str, str], None] | None = None,
        on_think: Callable[[str], None] | None = None,
        on_answer: Callable[[str], None] | None = None,
    ) -> AgentResult:
        """
        Run the agent loop.

        on_step(step_num, tool_name, result_preview)
        on_think(think_text)
        on_answer(answer_text)
        """
        # 1. Recall long-term memory
        past = self.memory_lt.recall(question)
        if past:
            self.memory_st.add("system", f"相关历史记忆:\n{past}")

        # 2. Plan
        plan = None
        if self.use_planning:
            plan = self.planner.create_plan(question)

        # 3. Execute loop
        answer = ""
        no_result_streak = 0

        for step in range(self.max_steps):
            # Build prompt
            system = AGENT_SYSTEM.format(max_steps=self.max_steps)
            tools_desc = self.tools.list_definitions()
            mem_ctx = self.memory_st.get_context()

            plan_ctx = ""
            if plan:
                plan_ctx = f"\n## 当前计划\n{plan.to_context()}\n"

            user_msg = f"""## 工具列表
{tools_desc}
{plan_ctx}
## 记忆
{mem_ctx}

## 用户问题
{question}

请思考下一步。输出 JSON:"""

            raw = self.llm.complete(system, user_msg, temperature=0.2)

            if not raw:
                # LLM unavailable, fallback
                results = self.store.query(question, n_results=5)
                if results:
                    from .rag import _format_context
                    answer = "LLM 不可用，相关代码：\n\n" + _format_context(results)
                else:
                    answer = "未找到相关代码，LLM 不可用。"
                break

            parsed = self._parse_json(raw)
            think = parsed.get("think", "")

            if on_think and think:
                on_think(think)
            self.memory_st.add("agent", f"Think: {think}")

            # Check for final answer
            if "answer" in parsed:
                answer = parsed["answer"]
                break

            # Execute tool
            tool_name = parsed.get("tool", "")
            params = parsed.get("params", {})

            if not tool_name:
                answer = parsed.get("answer", raw)
                break

            result = self.executor.execute(tool_name, params, self.ctx)

            preview = result.output[:150] + "..." if len(result.output) > 150 else result.output
            if on_step:
                on_step(step + 1, tool_name, preview)

            self.memory_st.add("tool", f"[{tool_name}] {result.output[:500]}", tool_name=tool_name)

            if plan:
                if result.success:
                    plan.mark_current("done", result.output[:100])
                else:
                    plan.mark_current("failed", result.output[:100])
                    self.planner.refine_plan(plan, result.output)

            # Track no-result streak
            if result.success and len(result.output) < 20:
                no_result_streak += 1
            else:
                no_result_streak = 0
            if no_result_streak >= 3:
                answer = "连续多次未找到有效信息，请尝试更具体的问题。"
                break

        if not answer:
            answer = "已达到最大步数。"

        # 4. Store to long-term memory
        self.memory_lt.store(question, answer, self.executor.log)

        if on_answer:
            on_answer(answer)

        return AgentResult(
            answer=answer,
            plan=plan,
            actions=self.executor.log,
            steps_taken=len(self.executor.log),
            memory_entries=len(self.memory_st.entries),
        )

    def _parse_json(self, raw: str) -> dict:
        """Extract JSON from LLM response."""
        text = raw.strip()
        if "```json" in text:
            m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                text = m.group(1)
        elif "```" in text:
            m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                text = m.group(1)
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"think": "parse failed", "answer": text}

    def reset_memory(self):
        """Clear all memory."""
        self.memory_st.clear()
        self.memory_lt.clear()
