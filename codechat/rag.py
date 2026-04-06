"""RAG query engine - retrieves context and generates answers."""

from __future__ import annotations

import os
import sys
import json
import time
from collections.abc import Callable
from pathlib import Path

from .config import get_snowcode_dir, get_llm_config_from_file
from .store import VectorStore


SYSTEM_PROMPT = """\
你是 snowcode，一位资深的代码架构分析师。你的任务是基于提供的代码上下文，准确回答用户关于代码库的问题。

## 核心原则

1. **只基于上下文回答**：严格依据提供的代码片段作答，不编造任何信息。
2. **直接回答**：不要说"根据上下文..."、"我找到了..."之类的废话，直接给出答案。
3. **精确定位**：引用代码时必须标注文件路径和行号，格式为 `文件路径:起始行-结束行`。
4. **展示关键代码**：用代码块展示与答案直接相关的代码片段（不超过 15 行），避免贴大段无关代码。
5. **信息不足时明确告知**：如果上下文无法回答问题，直接说"在提供的上下文中未找到相关信息"，并建议用户需要查看哪些类型的文件。

## 回答格式要求

- 使用 Markdown 格式
- 文件名用行内代码 `file.py` 标注
- 函数名、变量名用行内代码 `function_name` 标注
- 代码片段用 fenced code block，并标注语言
- 列表使用 `-` 而非 `*`
- 不要使用表格（终端渲染效果差）

## 不同问题类型的回答策略

**"XX 在哪个文件？"类问题**：
→ 直接给出文件路径和关键行号，附上 1-2 行核心代码。

**"XX 怎么实现的？"类问题**：
→ 1. 一句话概括实现思路
→ 2. 按执行顺序列出关键步骤，每个步骤标注文件和行号
→ 3. 展示核心代码片段

**"XX 和 XX 有什么区别？"类问题**：
→ 直接列出差异点，每个差异点标注对应的代码位置。

**"这个项目是做什么的？"类问题**：
→ 基于代码中的入口文件、核心模块、配置文件推断项目用途，不要依赖 README。

## 回答语言

- 用户用中文提问，用中文回答
- 用户用英文提问，用英文回答
- 代码术语保持英文（如 function、class、import）
"""


def _format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context string."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        file_path = meta["file_path"]
        start = meta["start_line"]
        end = meta["end_line"]
        content = r["content"]
        # Detect language for syntax highlighting
        ext = Path(file_path).suffix.lstrip(".")
        lang_map = {
            "py": "python", "js": "javascript", "ts": "typescript",
            "tsx": "tsx", "jsx": "jsx", "go": "go", "rs": "rust",
            "java": "java", "kt": "kotlin", "c": "c", "cpp": "cpp",
            "h": "c", "hpp": "cpp", "rb": "ruby", "php": "php",
            "sh": "bash", "sql": "sql", "md": "markdown",
            "yaml": "yaml", "yml": "yaml", "json": "json", "toml": "toml",
        }
        lang = lang_map.get(ext, ext)
        parts.append(
            f"[{i}] `{file_path}` (第 {start}-{end} 行)\n```{lang}\n{content}\n```"
        )
    return "\n\n".join(parts)


def _get_llm_config(model: str | None = None) -> tuple[str, str, str, bool]:
    """
    Resolve LLM backend config.
    Returns (api_key, base_url, model_name, enable_thinking).
    Priority:
      1. Environment variables (DASHSCOPE_API_KEY, OPENAI_API_KEY, OLLAMA_URL)
      2. Project config file (.snowcode/config.json) for model name and thinking mode
      3. Hardcoded defaults
    """
    # Try to get project root from current directory
    cwd = Path.cwd()
    project_root = cwd
    for parent in [cwd, *cwd.parents]:
        if (parent / ".snowcode").exists() or (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            project_root = parent
            break
    
    # Load config from file (for model name and thinking mode)
    file_config = get_llm_config_from_file(project_root)
    
    # Environment variables override file config
    # DashScope (Alibaba Qwen)
    ds_key = os.environ.get("DASHSCOPE_API_KEY")
    if ds_key:
        ds_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        ds_model = model or os.environ.get("CODECHAT_MODEL", file_config["default_model"])
        thinking = os.environ.get("CODECHAT_THINKING", "1" if file_config["thinking_enabled"] else "0") == "1"
        return ds_key, ds_url, ds_model, thinking
    
    # OpenAI-compatible
    oa_key = os.environ.get("OPENAI_API_KEY")
    if oa_key:
        oa_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        oa_model = model or os.environ.get("CODECHAT_MODEL", file_config["default_model"])
        thinking = file_config["thinking_enabled"]
        return oa_key, oa_url, oa_model, thinking
    
    # Ollama (local)
    ollama_url = os.environ.get("OLLAMA_URL")
    if ollama_url:
        ollama_model = model or os.environ.get("OLLAMA_MODEL", file_config["default_model"])
        return "ollama", ollama_url, ollama_model, file_config["thinking_enabled"]
    
    # No environment variables, use file config only if API key is set there?
    # Since we don't store API keys in config file (for security), we can't proceed.
    # Return empty config.
    return "", "", "", False

    return "", "", "", False


def _call_llm(prompt: str, model: str | None = None, _system_override: str | None = None, history: list[dict] | None = None, max_retries: int = 3) -> str:
    """Non-streaming LLM call (used as fallback)."""
    api_key, base_url, llm_model, thinking = _get_llm_config(model)
    system_msg = _system_override or SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_msg}]
    if history:
        # Use configurable history limit, fallback to 10
        limit = int(os.environ.get("CODECHAT_HISTORY_LIMIT", "10"))
        messages.extend(history[-limit:])
    messages.append({"role": "user", "content": prompt})

    if not api_key:
        return ""

    for attempt in range(max_retries):
        # Ollama uses httpx, not openai SDK
        if api_key == "ollama":
            try:
                import httpx
                resp = httpx.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": llm_model,
                        "messages": messages,
                        "stream": False,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    return resp.json()["message"]["content"]
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"\n[LLM Error] {type(e).__name__}: {e}", file=sys.stderr)
            return ""

        # OpenAI-compatible API
        try:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            
            # Conditionally set extra_body instead of passing empty dict when not needed
            kwargs = {
                "model": llm_model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 4096,
            }
            if thinking:
                kwargs["extra_body"] = {"enable_thinking": True}
                
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"\n[LLM Error] {type(e).__name__}: {e}", file=sys.stderr)

    return ""


def stream_llm(
    prompt: str,
    model: str | None = None,
    on_think: Callable[[str], None] | None = None,
    on_answer: Callable[[str], None] | None = None,
    history: list[dict] | None = None,
    max_retries: int = 3,
) -> str:
    """
    Streaming LLM call with thinking/reasoning support.

    Calls on_think(token) for reasoning tokens, on_answer(token) for answer tokens.
    Returns the full answer text.
    """
    api_key, base_url, llm_model, thinking = _get_llm_config(model)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # Use configurable history limit, fallback to 10
        limit = int(os.environ.get("CODECHAT_HISTORY_LIMIT", "10"))
        messages.extend(history[-limit:])
    messages.append({"role": "user", "content": prompt})

    if not api_key:
        # Fallback to non-streaming
        answer = _call_llm(prompt, model=model, history=history)
        if answer and on_answer:
            on_answer(answer)
        return answer

    for attempt in range(max_retries):
        # Ollama: use httpx streaming
        if api_key == "ollama":
            try:
                import httpx
                answer_parts: list[str] = []
                with httpx.stream(
                    "POST",
                    f"{base_url}/api/chat",
                    json={
                        "model": llm_model,
                        "messages": messages,
                        "stream": True,
                    },
                    timeout=120,
                ) as resp:
                    for line in resp.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    answer_parts.append(content)
                                    if on_answer:
                                        on_answer(content)
                            except json.JSONDecodeError:
                                continue
                return "".join(answer_parts)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"\n[LLM Error] {type(e).__name__}: {e}", file=sys.stderr)
                return ""

        # OpenAI-compatible streaming
        try:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            
            kwargs = {
                "model": llm_model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 4096,
                "stream": True,
            }
            if thinking:
                kwargs["extra_body"] = {"enable_thinking": True}
                
            stream = client.chat.completions.create(**kwargs)
            answer_parts: list[str] = []
            is_answering = False
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                # Reasoning / thinking tokens
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if on_think:
                        on_think(delta.reasoning_content)
                # Answer tokens
                if delta.content:
                    if not is_answering:
                        is_answering = True
                    answer_parts.append(delta.content)
                    if on_answer:
                        on_answer(delta.content)
            return "".join(answer_parts)
        except Exception as e:
            if not locals().get("is_answering", False) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"\n[LLM Error] {type(e).__name__}: {e}", file=sys.stderr)
            return "".join(locals().get("answer_parts", []))

    # Fallback: non-streaming
    answer = _call_llm(prompt, model=model)
    if answer and on_answer:
        on_answer(answer)
    return answer


def answer_question(
    store: VectorStore,
    question: str,
    n_context: int = 5,
    model: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    Answer a question (non-streaming).

    Returns dict with:
    - answer: str
    - sources: list[dict]
    - context: str
    """
    results = store.query(question, n_results=n_context)

    if not results:
        return {
            "answer": "No relevant code found. Has the project been ingested? Run `snowcode ingest` first.",
            "sources": [],
            "context": "",
        }

    context = _format_context(results)
    prompt = _build_prompt(context, question, history=history)
    answer = _call_llm(prompt, model=model, history=history)

    if not answer:
        answer = (
            "No LLM configured. Here are the most relevant code sections:\n\n"
            + context
            + "\n\n---\nSet DASHSCOPE_API_KEY / OPENAI_API_KEY or configure Ollama."
        )

    return {"answer": answer, "sources": results, "context": context}


def answer_question_stream(
    store: VectorStore,
    question: str,
    n_context: int = 5,
    model: str | None = None,
    on_think: Callable[[str], None] | None = None,
    on_answer: Callable[[str], None] | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    Answer a question with streaming output.

    Returns dict with answer, sources, context.
    """
    results = store.query(question, n_results=n_context)

    if not results:
        msg = "No relevant code found. Has the project been ingested? Run `snowcode ingest` first."
        if on_answer:
            on_answer(msg)
        return {"answer": msg, "sources": [], "context": ""}

    context = _format_context(results)
    prompt = _build_prompt(context, question, history=history)
    answer = stream_llm(prompt, model=model, on_think=on_think, on_answer=on_answer, history=history)

    if not answer:
        fallback = (
            "No LLM configured. Here are the most relevant code sections:\n\n"
            + context
            + "\n\n---\nSet DASHSCOPE_API_KEY / OPENAI_API_KEY or configure Ollama."
        )
        if on_answer:
            on_answer(fallback)
        answer = fallback

    return {"answer": answer, "sources": results, "context": context}


def _build_prompt(context: str, question: str, history: list[dict] | None = None) -> str:
    # If history is present, we might want to adjust the prompt to be more conversational
    # but currently history is handled by messages list in _call_llm/stream_llm.
    # This function creates the *last* user message content.
    return f"""## 代码上下文

以下是与用户问题相关的代码片段，每个片段标注了文件路径和行号：

{context}

## 用户问题

{question}

请基于以上代码上下文直接回答用户问题。要求：
- 直接给出答案，不要说"根据上下文"等前缀
- 引用代码时标注 `文件路径:行号`
- 展示关键代码片段（不超过 15 行）
- 如果上下文不足以回答，明确说明需要查看哪些类型的文件"""
