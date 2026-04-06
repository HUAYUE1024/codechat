#!/usr/bin/env python
"""Quick test for agent_v2 module."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing agent_v2 import...")

try:
    from codechat.agent_v2 import (
        CodeAgent,
        ToolRegistry,
        build_default_registry,
        ShortTermMemory,
        Plan,
        PlanStep,
    )
    print("✓ Import successful!")
    
    # Test ToolRegistry
    reg = build_default_registry()
    tools = [t.name for t in reg.list_tools()]
    print(f"✓ Tools registered: {tools}")
    
    # Test ShortTermMemory
    memory = ShortTermMemory(max_entries=5, max_tokens=1000)
    memory.add("user", "Hello")
    memory.add("agent", "Hi there")
    context = memory.get_context()
    print(f"✓ Memory works: {len(context)} chars")
    
    # Test Plan
    plan = Plan(
        goal="test",
        steps=[
            PlanStep(1, "step1", status="done"),
            PlanStep(2, "step2", status="pending"),
        ]
    )
    print(f"✓ Plan works: done={plan.done}, current_step={plan.current_step.description if plan.current_step else None}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
