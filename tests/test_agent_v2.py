"""
Tests for Agent v2 - Enhanced agent with Claude Code-inspired features.
"""

import pytest
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codechat.agent_v2 import (
    BaseTool,
    ToolResult,
    ToolRegistry,
    ToolPermission,
    ToolExecutionContext,
    ShortTermMemory,
    LongTermMemory,
    MemoryEntry,
    Plan,
    PlanStep,
    LLMClient,
    SearchTool,
    ReadFileTool,
    FindPatternTool,
    ListDirTool,
    WriteFileTool,
    SearchReplaceTool,
    ShellTool,
    build_default_registry,
    CodeAgent,
    MultiAgentCoordinator,
    AgentResult,
)


class TestToolResult:
    """Test ToolResult dataclass."""
    
    def test_basic_result(self):
        result = ToolResult(True, "output", "test_tool")
        assert result.success is True
        assert result.output == "output"
        assert result.tool_name == "test_tool"
        assert result.elapsed_ms == 0
    
    def test_preview_truncation(self):
        long_output = "x" * 500
        result = ToolResult(True, long_output, "test_tool")
        assert len(result.preview) == 203  # 200 + "..."
        assert result.preview.endswith("...")
    
    def test_preview_no_truncation(self):
        short_output = "short"
        result = ToolResult(True, short_output, "test_tool")
        assert result.preview == short_output


class TestToolPermission:
    """Test ToolPermission enum."""
    
    def test_permission_values(self):
        assert ToolPermission.ALLOWED.value == "allowed"
        assert ToolPermission.PROMPT.value == "prompt"
        assert ToolPermission.DENIED.value == "denied"
        assert ToolPermission.DANGEROUS.value == "dangerous"


class TestShortTermMemory:
    """Test ShortTermMemory class."""
    
    def test_add_and_get_context(self):
        memory = ShortTermMemory(max_entries=5, max_tokens=1000)
        memory.add("user", "Hello")
        memory.add("agent", "Hi there")
        
        context = memory.get_context()
        assert "Hello" in context
        assert "Hi there" in context
    
    def test_prune_by_entries(self):
        memory = ShortTermMemory(max_entries=3, max_tokens=10000)
        
        # Add more than max_entries
        for i in range(5):
            memory.add("user", f"Message {i}")
        
        # Should keep first and last (max_entries - 1)
        assert len(memory.entries) == 3
        assert memory.entries[0].content == "Message 0"
        assert memory.entries[-1].content == "Message 4"
    
    def test_prune_by_tokens(self):
        memory = ShortTermMemory(max_entries=10, max_tokens=100)
        
        # Add large entries
        for i in range(5):
            memory.add("tool", "x" * 100)  # ~25 tokens each
        
        # Should prune to stay under token limit
        total_tokens = sum(e.token_estimate for e in memory.entries)
        assert total_tokens <= 100
    
    def test_clear(self):
        memory = ShortTermMemory()
        memory.add("user", "test")
        memory.clear()
        assert len(memory.entries) == 0
    
    def test_get_recent_tool_results(self):
        memory = ShortTermMemory()
        memory.add("user", "question")
        memory.add("tool", "result1", tool_name="search")
        memory.add("agent", "thinking")
        memory.add("tool", "result2", tool_name="read_file")
        
        results = memory.get_recent_tool_results(2)
        assert len(results) == 2
        assert "result1" in results[0]
        assert "result2" in results[1]


class TestMemoryEntry:
    """Test MemoryEntry dataclass."""
    
    def test_token_estimate(self):
        entry = MemoryEntry(role="user", content="x" * 100)
        assert entry.token_estimate == 25  # 100 / 4
    
    def test_custom_token_estimate(self):
        entry = MemoryEntry(role="user", content="x" * 100, token_estimate=50)
        assert entry.token_estimate == 50


class TestPlan:
    """Test Plan class."""
    
    def test_done_property(self):
        plan = Plan(
            goal="test",
            steps=[
                PlanStep(1, "step1", status="done"),
                PlanStep(2, "step2", status="done"),
            ]
        )
        assert plan.done is True
    
    def test_not_done(self):
        plan = Plan(
            goal="test",
            steps=[
                PlanStep(1, "step1", status="done"),
                PlanStep(2, "step2", status="pending"),
            ]
        )
        assert plan.done is False
    
    def test_current_step(self):
        plan = Plan(
            goal="test",
            steps=[
                PlanStep(1, "step1", status="done"),
                PlanStep(2, "step2", status="pending"),
            ]
        )
        assert plan.current_step is not None
        assert plan.current_step.description == "step2"
    
    def test_mark_current(self):
        plan = Plan(
            goal="test",
            steps=[
                PlanStep(1, "step1", status="pending"),
            ]
        )
        plan.mark_current("done", "completed successfully")
        assert plan.steps[0].status == "done"
        assert plan.steps[0].result == "completed successfully"
    
    def test_to_context(self):
        plan = Plan(
            goal="test",
            steps=[
                PlanStep(1, "step1", status="done", result="ok"),
                PlanStep(2, "step2", status="pending"),
            ]
        )
        context = plan.to_context()
        assert "Goal: test" in context
        assert "[+]" in context  # done icon
        assert "[ ]" in context  # pending icon


class TestToolRegistry:
    """Test ToolRegistry class."""
    
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = SearchTool()
        registry.register(tool)
        
        assert registry.get("search") is tool
        assert registry.get("nonexistent") is None
    
    def test_list_tools(self):
        registry = build_default_registry()
        tools = registry.list_tools()
        assert len(tools) > 0
        assert all(isinstance(t, BaseTool) for t in tools)
    
    def test_list_definitions(self):
        registry = build_default_registry()
        definitions = registry.list_definitions()
        assert "**search**" in definitions
        assert "**read_file**" in definitions
    
    def test_check_permission_unknown_tool(self):
        registry = ToolRegistry()
        allowed, msg = registry.check_permission("unknown", {})
        assert allowed is False
        assert "Unknown tool" in msg
    
    def test_check_permission_denied(self):
        registry = ToolRegistry()
        
        class DeniedTool(BaseTool):
            name = "denied_tool"
            description = "test"
            
            def check_permission(self, params):
                return ToolPermission.DENIED
            
            def run(self, params, ctx):
                return "ok"
        
        registry.register(DeniedTool())
        allowed, msg = registry.check_permission("denied_tool", {})
        assert allowed is False
        assert "denied" in msg.lower()


class TestBaseTool:
    """Test BaseTool features."""
    
    def test_format_output_truncation(self):
        class TestTool(BaseTool):
            name = "test"
            description = "test"
            max_result_size = 100
            
            def run(self, params, ctx):
                return "x" * 200
        
        tool = TestTool()
        output = tool.format_output("x" * 200)
        assert len(output) < 200
        assert "Truncated" in output
    
    def test_format_output_no_truncation(self):
        class TestTool(BaseTool):
            name = "test"
            description = "test"
            
            def run(self, params, ctx):
                return "short"
        
        tool = TestTool()
        output = tool.format_output("short")
        assert output == "short"


class TestSearchTool:
    """Test SearchTool."""
    
    def test_is_read_only(self):
        tool = SearchTool()
        assert tool.is_read_only() is True
    
    def test_is_concurrent_safe(self):
        tool = SearchTool()
        assert tool.is_concurrent_safe() is True
    
    def test_parameters(self):
        tool = SearchTool()
        assert "query" in tool.parameters
        assert "n" in tool.parameters


class TestReadFileTool:
    """Test ReadFileTool."""
    
    def test_is_read_only(self):
        tool = ReadFileTool()
        assert tool.is_read_only() is True
    
    def test_max_lines_per_read(self):
        tool = ReadFileTool()
        assert tool.max_lines_per_read == 2000
    
    def test_run_with_temp_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            test_file = root / "test.py"
            test_file.write_text("line1\nline2\nline3\n")
            
            ctx = ToolExecutionContext(root=root)
            tool = ReadFileTool()
            result = tool.run({"path": "test.py"}, ctx)
            
            assert "test.py" in result
            assert "line1" in result
            assert "line2" in result
            assert "line3" in result
    
    def test_run_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = ToolExecutionContext(root=Path(tmpdir))
            tool = ReadFileTool()
            result = tool.run({"path": "nonexistent.py"}, ctx)
            
            assert "not found" in result.lower() or "Error" in result


class TestListDirTool:
    """Test ListDirTool."""
    
    def test_is_read_only(self):
        tool = ListDirTool()
        assert tool.is_read_only() is True
    
    def test_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "dir1").mkdir()
            (root / "file1.txt").write_text("content")
            
            ctx = ToolExecutionContext(root=root)
            tool = ListDirTool()
            result = tool.run({}, ctx)
            
            assert "dir1/" in result
            assert "file1.txt" in result


class TestShellTool:
    """Test ShellTool."""
    
    def test_is_read_only(self):
        tool = ShellTool()
        assert tool.is_read_only() is False
    
    def test_check_permission(self):
        tool = ShellTool()
        
        # Normal command
        perm = tool.check_permission({"command": "ls"})
        assert perm == ToolPermission.PROMPT
        
        # Dangerous command
        perm = tool.check_permission({"command": "rm -rf /"})
        assert perm == ToolPermission.DENIED
    
    def test_run_simple_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = ToolExecutionContext(root=Path(tmpdir))
            tool = ShellTool()
            
            # Platform-specific command
            if sys.platform == "win32":
                result = tool.run({"command": "echo hello"}, ctx)
            else:
                result = tool.run({"command": "echo hello"}, ctx)
            
            assert "hello" in result.lower() or "no output" in result.lower()


class TestWriteFileTool:
    """Test WriteFileTool."""
    
    def test_is_read_only(self):
        tool = WriteFileTool()
        assert tool.is_read_only() is False
    
    def test_check_permission(self):
        tool = WriteFileTool()
        perm = tool.check_permission({})
        assert perm == ToolPermission.PROMPT
    
    def test_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ctx = ToolExecutionContext(root=root)
            tool = WriteFileTool()
            
            result = tool.run({
                "path": "new_file.py",
                "content": "print('hello')"
            }, ctx)
            
            assert "success" in result.lower()
            assert (root / "new_file.py").exists()
            assert (root / "new_file.py").read_text() == "print('hello')"


class TestCodeAgent:
    """Test CodeAgent class."""
    
    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create a minimal vector store mock
            class MockStore:
                def query(self, q, n_results=5):
                    return []
            
            agent = CodeAgent(MockStore(), root, max_steps=3)
            
            assert agent.root == root
            assert agent.max_steps == 3
            assert agent.tools is not None
            assert agent.memory_st is not None
            assert agent.memory_lt is not None
    
    def test_parse_json_markdown_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class MockStore:
                def query(self, q, n_results=5):
                    return []
            
            agent = CodeAgent(MockStore(), Path(tmpdir))
            
            # Test markdown JSON block
            raw = '```json\n{"think": "test", "answer": "result"}\n```'
            parsed = agent._parse_json(raw)
            assert parsed["think"] == "test"
            assert parsed["answer"] == "result"
    
    def test_parse_json_direct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class MockStore:
                def query(self, q, n_results=5):
                    return []
            
            agent = CodeAgent(MockStore(), Path(tmpdir))
            
            # Test direct JSON
            raw = '{"think": "test", "answer": "result"}'
            parsed = agent._parse_json(raw)
            assert parsed["answer"] == "result"
    
    def test_parse_json_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class MockStore:
                def query(self, q, n_results=5):
                    return []
            
            agent = CodeAgent(MockStore(), Path(tmpdir))
            
            # Test fallback to plain text
            raw = "This is not JSON, but Answer: This is the answer"
            parsed = agent._parse_json(raw)
            assert "answer" in parsed
            assert "This is the answer" in parsed["answer"]


class TestMultiAgentCoordinator:
    """Test MultiAgentCoordinator class."""
    
    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class MockStore:
                def query(self, q, n_results=5):
                    return []
            
            coordinator = MultiAgentCoordinator(MockStore(), Path(tmpdir))
            assert coordinator.root == Path(tmpdir)
            assert coordinator.tasks == {}
            assert coordinator.results == {}


class TestBuildDefaultRegistry:
    """Test build_default_registry function."""
    
    def test_creates_all_tools(self):
        registry = build_default_registry()
        
        expected_tools = [
            "search",
            "read_file",
            "find_pattern",
            "list_dir",
            "write_file",
            "search_replace",
            "shell",
        ]
        
        for tool_name in expected_tools:
            assert registry.get(tool_name) is not None, f"Missing tool: {tool_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
