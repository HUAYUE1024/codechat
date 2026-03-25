"""Specialized skills for code analysis."""

from __future__ import annotations

from .rag import _format_context, _call_llm, stream_llm, _build_prompt
from .store import VectorStore


# ================================================================== prompts

EXPLAIN_PROMPT = """\
你是一位代码解读专家。根据以下代码，用通俗易懂的语言解释这段代码的功能。

要求：
1. 一句话说清楚这个文件/函数是做什么的
2. 列出关键函数/类的作用（标注行号）
3. 说明数据流向或调用关系
4. 如果有设计模式或特殊技巧，点明
5. 用中文回答，代码术语保留英文
"""

REVIEW_PROMPT = """\
你是一位资深代码审查专家。请对以下代码进行 Code Review。

审查维度：
1. **Bug 风险**：潜在的空指针、边界越界、资源泄漏、竞态条件
2. **安全问题**：SQL 注入、路径遍历、敏感信息泄露、不安全的反序列化
3. **性能问题**：不必要的重复计算、内存泄漏风险、低效算法
4. **代码质量**：命名规范、错误处理、代码重复、过度复杂

输出格式：
- 按严重程度排序（严重 > 中等 > 建议）
- 每个问题标注文件路径和行号
- 给出修复建议
- 如果代码质量良好，也请说明优点
- 用中文回答
"""

FIND_PROMPT = """\
你是一位代码搜索专家。根据用户的搜索意图，在提供的代码上下文中查找匹配项。

要求：
1. 列出所有匹配的位置（文件路径 + 行号）
2. 每个匹配附上 1-2 行关键代码
3. 按相关度排序
4. 如果没有找到匹配，说明可能的原因和替代搜索方向
5. 用中文回答
"""

SUMMARY_PROMPT = """\
你是一位软件架构分析师。根据提供的代码上下文，生成项目的架构概览。

输出结构：
1. **项目定位**：一句话说明这个项目是什么、解决什么问题
2. **核心模块**：列出主要模块及其职责
3. **数据流**：关键的数据流向（从输入到输出）
4. **技术栈**：使用的主要框架、库、协议
5. **目录结构**：关键目录的作用

要求：
- 基于代码推断，不要依赖 README
- 标注关键文件路径
- 用中文回答
"""

TRACE_PROMPT = """\
你是一位代码追踪专家。根据上下文，追踪指定函数/方法的调用链。

输出格式：
1. **调用链**：从入口到目标的完整调用路径，格式为 `A() → B() → C()`
2. **每层说明**：每个调用步骤标注文件路径和行号，一句话说明做了什么
3. **被调用方**：这个函数还调用了哪些其他函数
4. **关键参数**：调用链中传递的关键参数和返回值

要求：
- 如果上下文不足以还原完整调用链，明确说明哪些部分缺失
- 用中文回答
"""

COMPARE_PROMPT = """\
你是一位代码对比分析专家。根据提供的两段代码上下文，分析它们的异同。

输出结构：
1. **功能对比**：两者分别做什么
2. **共同点**：共享的设计思路或实现模式
3. **差异点**：逐项列出不同之处，标注对应的代码位置
4. **适用场景**：各自适合什么场景

要求：
- 标注文件路径和行号
- 用中文回答
"""

GENERATE_TEST_PROMPT = """\
你是一位测试工程师。根据以下代码，生成测试用例建议。

输出结构：
1. **核心功能测试**：覆盖主要功能路径的测试用例
2. **边界条件测试**：空值、超长输入、非法参数等
3. **异常处理测试**：错误路径和异常恢复
4. **集成测试建议**：与外部依赖的交互测试

要求：
- 使用 pytest 风格的伪代码
- 每个测试用例标注被测函数和文件路径
- 用中文回答
"""


# ================================================================== skill queries

def _build_skill_query(question: str, extra_context: str = "") -> str:
    """Build a search query for skill-based analysis."""
    return question + extra_context


SKILL_QUERIES = {
    "explain": {
        "prompt": EXPLAIN_PROMPT,
        "query_suffix": " 实现 原理 逻辑",
        "n_context": 8,
    },
    "review": {
        "prompt": REVIEW_PROMPT,
        "query_suffix": " 实现 逻辑 错误",
        "n_context": 10,
    },
    "find": {
        "prompt": FIND_PROMPT,
        "query_suffix": "",
        "n_context": 10,
    },
    "summary": {
        "prompt": SUMMARY_PROMPT,
        "query_suffix": " 项目 结构 模块 入口",
        "n_context": 12,
    },
    "trace": {
        "prompt": TRACE_PROMPT,
        "query_suffix": " 调用 函数 方法",
        "n_context": 10,
    },
    "compare": {
        "prompt": COMPARE_PROMPT,
        "query_suffix": " 对比 区别",
        "n_context": 10,
    },
    "test": {
        "prompt": GENERATE_TEST_PROMPT,
        "query_suffix": " 函数 参数 返回",
        "n_context": 8,
    },
}


def run_skill(
    store: VectorStore,
    skill_name: str,
    question: str,
    model: str | None = None,
) -> dict:
    """Run a skill synchronously."""
    skill = SKILL_QUERIES[skill_name]
    query = _build_skill_query(question, skill["query_suffix"])
    results = store.query(query, n_results=skill["n_context"])

    if not results:
        return {
            "answer": "未找到相关代码。请先运行 `codechat ingest` 建立索引。",
            "sources": [],
            "context": "",
        }

    context = _format_context(results)
    prompt = f"""{skill['prompt']}

## 代码上下文

{context}

## 分析目标

{question}"""

    answer = _call_llm(prompt, model=model)

    if not answer:
        answer = (
            "未配置 LLM，以下是相关代码：\n\n"
            + context
            + "\n\n---\n设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 启用 AI 分析。"
        )

    return {"answer": answer, "sources": results, "context": context}


def run_skill_stream(
    store: VectorStore,
    skill_name: str,
    question: str,
    model: str | None = None,
    on_think=None,
    on_answer=None,
) -> dict:
    """Run a skill with streaming output."""
    skill = SKILL_QUERIES[skill_name]
    query = _build_skill_query(question, skill["query_suffix"])
    results = store.query(query, n_results=skill["n_context"])

    if not results:
        msg = "未找到相关代码。请先运行 `codechat ingest` 建立索引。"
        if on_answer:
            on_answer(msg)
        return {"answer": msg, "sources": [], "context": ""}

    context = _format_context(results)
    prompt = f"""{skill['prompt']}

## 代码上下文

{context}

## 分析目标

{question}"""

    answer = stream_llm(prompt, model=model, on_think=on_think, on_answer=on_answer)

    if not answer:
        fallback = (
            "未配置 LLM，以下是相关代码：\n\n"
            + context
            + "\n\n---\n设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 启用 AI 分析。"
        )
        if on_answer:
            on_answer(fallback)
        answer = fallback

    return {"answer": answer, "sources": results, "context": context}
