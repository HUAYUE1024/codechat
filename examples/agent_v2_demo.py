#!/usr/bin/env python
"""
Agent v2 快速示例

使用方法:
    python examples/agent_v2_demo.py [项目路径] [问题]

示例:
    python examples/agent_v2_demo.py ./my_project "解释这个项目的架构"
"""

import sys
import os

# 确保可以导入 codechat
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from codechat.agent_v2 import create_agent, create_coordinator, build_default_registry


def demo_basic_usage(project_path: str, question: str):
    """基础用法示例"""
    print("=" * 60)
    print("Agent v2 基础用法示例")
    print("=" * 60)
    
    from codechat.store import VectorStore
    
    root = Path(project_path).resolve()
    store = VectorStore(root)
    
    # 创建 Agent
    agent = create_agent(
        store=store,
        project_root=root,
        max_steps=5,
        use_planning=True
    )
    
    print(f"\n📁 项目: {root}")
    print(f"❓ 问题: {question}")
    print(f"🔧 可用工具: {[t.name for t in agent.tools.list_tools()]}")
    print("\n" + "-" * 60)
    
    # 定义回调函数
    def on_step(step_num, tool_name, preview):
        print(f"\n[步骤 {step_num}] 🔧 {tool_name}")
        print(f"  📄 结果: {preview[:100]}...")
    
    def on_think(think_text):
        print(f"\n[思考] 💭 {think_text[:200]}...")
    
    # 运行 Agent
    result = agent.run(
        question=question,
        on_step=on_step,
        on_think=on_think
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("📊 执行统计")
    print("=" * 60)
    print(f"✅ 答案: {result.answer}")
    print(f"📈 执行步数: {result.steps_taken}")
    print(f"🔧 使用工具: {result.tools_used}")
    print(f"⏱️  耗时: {result.total_elapsed_ms:.0f}ms")
    
    if result.plan:
        print(f"\n📋 执行计划:\n{result.plan.to_context()}")


def demo_tool_registry():
    """工具注册表示例"""
    print("=" * 60)
    print("工具注册表示例")
    print("=" * 60)
    
    registry = build_default_registry()
    
    print(f"\n📦 已注册工具: {len(registry.list_tools())}")
    print("\n工具列表:")
    for tool in registry.list_tools():
        perms = "只读" if tool.is_read_only() else "写操作"
        concurrent = "✓" if tool.is_concurrent_safe() else "✗"
        print(f"  - {tool.name:15} | {perms:6} | 并发安全: {concurrent}")
        print(f"    {tool.description}")
        print(f"    参数: {tool.parameters}")
        print()


def demo_memory_system():
    """记忆系统示例"""
    print("=" * 60)
    print("记忆系统示例")
    print("=" * 60)
    
    from codechat.agent_v2 import ShortTermMemory, MemoryEntry
    
    # 创建短期记忆
    memory = ShortTermMemory(max_entries=5, max_tokens=1000)
    
    # 添加记忆条目
    memory.add("user", "什么是 Python 装饰器？")
    memory.add("agent", "装饰器是一种修改函数行为的设计模式...")
    memory.add("tool", "[search] 找到 3 个装饰器示例:\n1. @staticmethod\n2. @classmethod\n3. @property")
    memory.add("agent", "基于搜索结果，装饰器使用 @ 语法...")
    
    print(f"\n📝 记忆条目数: {len(memory.entries)}")
    print(f"📊 估算 Token 数: {sum(e.token_estimate for e in memory.entries)}")
    
    # 获取上下文
    context = memory.get_context(max_chars=500)
    print(f"\n📄 上下文 (截断到 500 字符):\n{context[:300]}...")
    
    # 获取最近的工具结果
    recent = memory.get_recent_tool_results(2)
    print(f"\n🔧 最近的工具结果: {len(recent)} 个")


def demo_plan_system():
    """计划系统示例"""
    print("=" * 60)
    print("计划系统示例")
    print("=" * 60)
    
    from codechat.agent_v2 import Plan, PlanStep
    
    # 创建计划
    plan = Plan(
        goal="重构认证模块",
        steps=[
            PlanStep(1, "搜索认证相关文件", tool_hint="search", status="done", result="找到 5 个文件"),
            PlanStep(2, "读取主要认证逻辑", tool_hint="read_file", status="done", result="已读取 auth.py"),
            PlanStep(3, "执行重构", tool_hint="search_replace", status="pending"),
            PlanStep(4, "运行测试验证", tool_hint="shell", status="pending"),
        ]
    )
    
    print(f"\n📋 目标: {plan.goal}")
    print(f"✅ 完成: {plan.done}")
    print(f"📌 当前步骤: {plan.current_step.description if plan.current_step else 'N/A'}")
    
    print(f"\n计划详情:\n{plan.to_context()}")


def demo_multi_agent(project_path: str, question: str):
    """多 Agent 协作示例"""
    print("=" * 60)
    print("多 Agent 协作示例")
    print("=" * 60)
    
    from codechat.store import VectorStore
    
    root = Path(project_path).resolve()
    store = VectorStore(root)
    
    # 创建 Coordinator
    coordinator = create_coordinator(
        store=store,
        project_root=root,
        num_workers=2
    )
    
    print(f"\n📁 项目: {root}")
    print(f"❓ 问题: {question}")
    print(f"👥 Worker 数量: {coordinator.num_workers}")
    print("\n" + "-" * 60)
    
    # 运行
    answer = coordinator.coordinate(
        question=question,
        on_progress=lambda msg: print(f"[进度] {msg}")
    )
    
    print("\n" + "=" * 60)
    print("✅ 综合答案:")
    print("=" * 60)
    print(answer)


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Agent v2 - 增强版代码智能体                          ║
║           基于 Claude Code 优秀设计                            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python agent_v2_demo.py <项目路径> [问题]")
        print("\n示例:")
        print("  python agent_v2_demo.py ./my_project")
        print("  python agent_v2_demo.py ./my_project '解释这个项目的架构'")
        print("\n运行演示模式（无需参数）:")
        print("  python agent_v2_demo.py demo")
        sys.exit(1)
    
    if sys.argv[1] == "demo":
        # 演示模式
        print("\n🔧 运行工具注册表示例...")
        demo_tool_registry()
        
        print("\n" + "=" * 60 + "\n")
        
        print("📝 运行记忆系统示例...")
        demo_memory_system()
        
        print("\n" + "=" * 60 + "\n")
        
        print("📋 运行计划系统示例...")
        demo_plan_system()
        
        print("\n" + "=" * 60)
        print("✅ 演示完成！")
        print("=" * 60)
    else:
        # 实际运行模式
        project_path = sys.argv[1]
        question = sys.argv[2] if len(sys.argv) > 2 else "解释这个项目的架构"
        
        demo_basic_usage(project_path, question)


if __name__ == "__main__":
    main()
